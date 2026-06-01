"""
Ollama-based inference for ABSA — 0-shot and few-shot evaluation.

Uses the ollama HTTP API to run inference with locally-hosted models
(command-r, qwen2.5-coder, gemma2, etc.). Reuses prompt construction,
demo retrieval, and evaluation from gemma_pipeline.
"""

import json
import time
import random
import requests
from pathlib import Path

from src.data.data import load_aste_file, load_acos_jsonl, load_files, enrich_syntax
from src.eval.eval import parse_output, evaluate, save_results, save_metrics_table
from src.pipelines.gemma_pipeline import (
    _get_system_prompt,
    _build_input_text,
    _build_few_shot_messages,
    DemoRetriever,
)


OLLAMA_URL = "http://localhost:11434/api/chat"


def _ollama_chat(model: str, messages: list[dict], max_tokens: int = 256) -> str:
    """Call ollama chat API and return the assistant response text."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.0,
        },
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def run_ollama_inference(cfg: dict, output_dir: Path):
    """Run 0-shot or few-shot inference via ollama API and evaluate."""
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_name = model_cfg.get("name", "gemma2:27b")
    n_shots = model_cfg.get("n_shots", 0)
    output_format = model_cfg.get("output_format", "structured")
    syntax_enrichment = data_cfg.get("syntax_enrichment", None)
    max_new_tokens = model_cfg.get("max_new_tokens", 256)
    seed = cfg.get("seed", 42)
    demo_selection = model_cfg.get("demo_selection", "hybrid")

    test_cfg = cfg.get("test", {})
    datasets = test_cfg.get("datasets", [])
    scopes = test_cfg.get("scopes", [{"keys": ["aspect", "sentiment", "polarity"], "metrics": ["micro_f1"]}])

    first_ds = datasets[0] if datasets else {}
    task_keys_str = first_ds.get("tasks", ["aspect", "sentiment", "polarity"])
    keys = task_keys_str if isinstance(task_keys_str, list) else list(task_keys_str)
    task_type = "quad" if "category" in keys and "aspect" in keys else "catpol" if "category" in keys else "triplet"

    print(f"\n{'='*60}")
    print(f"  Ollama Inference: {model_name}")
    print(f"  Mode: {n_shots}-shot | Format: {output_format} | Syntax: {syntax_enrichment or 'none'}")
    print(f"  Demo selection: {demo_selection}")
    print(f"  Task: {task_type} ({', '.join(keys)})")
    print(f"{'='*60}\n")

    # verify ollama is running
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        available = [m["name"] for m in r.json().get("models", [])]
        # check model is available (allow partial match)
        matched = [m for m in available if model_name in m or m.startswith(model_name)]
        if not matched:
            print(f"  WARNING: model '{model_name}' not found in ollama. Available: {available}")
        else:
            print(f"  Model found: {matched[0]}")
    except Exception as e:
        raise RuntimeError(f"Cannot connect to ollama at {OLLAMA_URL}: {e}")

    # load global training data for few-shot (fallback pool)
    global_train_examples = []
    global_demo_retriever = None
    global_few_shot_messages = []
    if n_shots > 0:
        train_files = data_cfg.get("train_file", [])
        if isinstance(train_files, str):
            train_files = [train_files]
        spacy_model = data_cfg.get("spacy_model", "en_core_web_sm")
        for f in train_files:
            global_train_examples.extend(load_files(f))
        if syntax_enrichment:
            global_train_examples = enrich_syntax(global_train_examples, syntax_enrichment, spacy_model=spacy_model)

        if demo_selection == "fixed":
            global_few_shot_messages = _build_few_shot_messages(
                global_train_examples, n_shots, output_format, keys,
                syntax_enrichment=syntax_enrichment, seed=seed,
            )
        else:
            print(f"  Building {demo_selection} retriever over {len(global_train_examples)} training examples...")
            global_demo_retriever = DemoRetriever(
                global_train_examples, method=demo_selection, n_shots=n_shots,
                output_format=output_format, keys=keys, syntax_enrichment=syntax_enrichment,
            )
            print("  Retriever ready.")

    # run on each test dataset
    system_prompt = _get_system_prompt(task_type, output_format)
    test_metrics_history = []

    for ds_cfg in datasets:
        data_path = ds_cfg["data"]
        ds_keys = ds_cfg.get("tasks", keys)
        if isinstance(ds_keys, list) and all(isinstance(k, str) for k in ds_keys):
            pass
        ds_scopes = ds_cfg.get("scopes", scopes)

        # per-dataset demo source override
        demo_source = ds_cfg.get("demo_source")
        if demo_source and n_shots > 0:
            ds_train = load_files(demo_source)
            if syntax_enrichment:
                ds_train = enrich_syntax(ds_train, syntax_enrichment, spacy_model=data_cfg.get("spacy_model", "en_core_web_sm"))
            if demo_selection == "fixed":
                demo_retriever = None
                few_shot_messages = _build_few_shot_messages(
                    ds_train, n_shots, output_format, keys,
                    syntax_enrichment=syntax_enrichment, seed=seed,
                )
            else:
                print(f"  Building {demo_selection} retriever over {len(ds_train)} examples (from {demo_source})...")
                demo_retriever = DemoRetriever(
                    ds_train, method=demo_selection, n_shots=n_shots,
                    output_format=output_format, keys=keys, syntax_enrichment=syntax_enrichment,
                )
                few_shot_messages = []
                print("  Retriever ready.")
        else:
            demo_retriever = global_demo_retriever
            few_shot_messages = global_few_shot_messages

        print(f"\nEvaluating: {data_path}")

        test_examples = load_files(data_path)

        if syntax_enrichment:
            spacy_model = data_cfg.get("spacy_model", "en_core_web_sm")
            test_examples = enrich_syntax(test_examples, syntax_enrichment, spacy_model=spacy_model)

        # build golds
        all_golds = []
        for ex in test_examples:
            gold_triplets = []
            for ann in ex["annotations"]:
                d = {k: ann.get(k, "NULL") for k in ds_keys}
                gold_triplets.append(d)
            all_golds.append(gold_triplets)

        # inference
        all_preds = []
        t0 = time.time()
        for i, ex in enumerate(test_examples):
            syntax = ex.get("_syntax_compact") if syntax_enrichment else None
            input_text = _build_input_text(ex["sentence"], syntax)

            demos = demo_retriever.get_demos(ex["sentence"]) if demo_retriever else few_shot_messages

            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(demos)
            messages.append({"role": "user", "content": input_text})

            decoded = _ollama_chat(model_name, messages, max_tokens=max_new_tokens)
            preds = parse_output(decoded, ds_keys, output_format)
            all_preds.append(preds)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = elapsed / (i + 1)
                remaining = rate * (len(test_examples) - i - 1)
                print(f"  {i+1}/{len(test_examples)} ({rate:.2f}s/ex, ~{remaining/60:.1f}min left)")

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed/len(test_examples):.2f}s/example)")

        # evaluate
        metrics = evaluate(all_preds, all_golds, ds_scopes)
        entry = {"data": data_path, **metrics}
        test_metrics_history.append(entry)

        for scope_key, scope_metrics in metrics.items():
            if isinstance(scope_metrics, dict):
                for metric_name, values in scope_metrics.items():
                    if isinstance(values, dict) and "f1" in values:
                        print(f"  {scope_key}/{metric_name}: P={values['precision']:.4f} R={values['recall']:.4f} F1={values['f1']:.4f}")

    # save results
    import yaml
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    for i, entry in enumerate(test_metrics_history):
        data_label = Path(entry["data"]).stem
        metrics = {k: v for k, v in entry.items() if k != "data"}
        save_metrics_table(metrics, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")

    save_results([], test_metrics_history, [], [], str(results_dir))
