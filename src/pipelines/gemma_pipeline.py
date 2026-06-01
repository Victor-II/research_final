"""
Gemma 4B inference for ABSA — 0-shot and few-shot evaluation.

Loads google/gemma-4-E4B-it in 4-bit quantization and prompts it for
aspect sentiment triplet/quad extraction. Supports both structured and
NL output formats, with optional syntax enrichment in the prompt.
"""

import json
import random
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.data import load_aste_file, load_acos_jsonl, enrich_syntax, to_generative_format
from src.eval.eval import parse_output, evaluate, save_results, save_metrics_table
from constants import Task, TASK_KEY_MAP


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_STRUCTURED = """You are an aspect-based sentiment analysis system. Given a review sentence, extract all (aspect, opinion, polarity) triplets.

Output format: one triplet per line as [aspect, opinion, polarity]
- aspect: the entity or feature being discussed (exact span from text)
- opinion: the opinion expression about the aspect (exact span from text)
- polarity: positive, negative, or neutral

If there are no triplets, output: none
Do not explain your reasoning. Only output the triplets."""

_SYSTEM_PROMPT_NL = """You are an aspect-based sentiment analysis system. Given a review sentence, extract all aspect-opinion-polarity triplets and express each one in natural language.

Output format: one triplet per line using this template:
"<aspect> is described as <opinion>, expressing a <polarity> sentiment"

- aspect: the entity or feature being discussed (exact span from text)
- opinion: the opinion expression about the aspect (exact span from text)
- polarity: positive, negative, or neutral

If there are no triplets, output: none"""

_SYSTEM_PROMPT_QUAD_STRUCTURED = """You are an aspect-based sentiment analysis system. Given a review sentence, extract all (aspect, opinion, polarity, category) quads.

Output format: one quad per line as [aspect, opinion, polarity, category]
- aspect: the entity or feature being discussed (exact span from text, or "NULL" if implicit)
- opinion: the opinion expression (exact span from text, or "NULL" if implicit)
- polarity: positive, negative, or neutral
- category: the aspect category (e.g. food quality, service general, etc.)

If there are no quads, output: none
Do not explain your reasoning. Only output the quads."""

_SYSTEM_PROMPT_QUAD_NL = """You are an aspect-based sentiment analysis system. Given a review sentence, extract all (aspect, opinion, polarity, category) quads and express each one in natural language.

Output format: one quad per line using this template:
"<aspect>, related to <category>, is described as <opinion>, expressing a <polarity> sentiment"

- aspect: the entity or feature being discussed (exact span from text, or "NULL" if implicit)
- opinion: the opinion expression (exact span from text, or "NULL" if implicit)
- polarity: positive, negative, or neutral
- category: the aspect category

If there are no quads, output: none"""

_SYSTEM_PROMPT_CATPOL = """You are a sentiment analysis system for phone reviews written in Romanian. Given a review, extract all (category, polarity) pairs that are discussed.

Output format: one pair per line as [category, polarity]
- category: Must be one of: experience, durability, performance, battery, price_quality, design, camera, audio, software, brand, service
- polarity: Must be one of: positive, negative, neutral

Category descriptions (for reference):
  experience = overall satisfaction, durability = build quality/longevity,
  performance = speed/responsiveness, battery = battery life/charging,
  price_quality = value for money, design = appearance/form factor,
  camera = photo/video quality, audio = speakers/microphone,
  software = OS/apps/updates, brand = manufacturer reputation,
  service = delivery/customer support/warranty

If there are no pairs, output: none
Do not explain your reasoning. Only output the pairs."""


def _get_system_prompt(task_type: str, output_format: str) -> str:
    if task_type == "catpol":
        return _SYSTEM_PROMPT_CATPOL
    if task_type == "quad":
        return _SYSTEM_PROMPT_QUAD_NL if output_format == "nl" else _SYSTEM_PROMPT_QUAD_STRUCTURED
    return _SYSTEM_PROMPT_NL if output_format == "nl" else _SYSTEM_PROMPT_STRUCTURED


def _format_example_structured(ex: dict, keys: list[str]) -> str:
    """Format a canonical example as structured output for few-shot."""
    parts = []
    for ann in ex["annotations"]:
        values = []
        for k in keys:
            v = ann.get(k)
            values.append(v if v else "NULL")
        parts.append(f"[{', '.join(values)}]")
    return "\n".join(parts) if parts else "none"


def _format_example_nl(ex: dict, keys: list[str]) -> str:
    """Format a canonical example as NL output for few-shot."""
    parts = []
    for ann in ex["annotations"]:
        aspect = ann.get("aspect", "NULL")
        sentiment = ann.get("sentiment", "NULL")
        polarity = ann.get("polarity", "neutral")
        category = ann.get("category")
        if category and "category" in keys:
            parts.append(f"{aspect}, related to {category}, is described as {sentiment}, expressing a {polarity} sentiment")
        else:
            parts.append(f"{aspect} is described as {sentiment}, expressing a {polarity} sentiment")
    return "\n".join(parts) if parts else "none"


def _build_input_text(sentence: str, syntax: str = None) -> str:
    text = f"Input: {sentence}"
    if syntax:
        text += f"\nSyntax: {syntax}"
    return text


def _build_few_shot_messages(
    examples: list[dict],
    n_shots: int,
    output_format: str,
    keys: list[str],
    syntax_enrichment: str = None,
    seed: int = 42,
) -> list[dict]:
    """Build fixed few-shot example messages (same for all test sentences)."""
    rng = random.Random(seed)
    sorted_exs = sorted(examples, key=lambda x: len(x["annotations"]), reverse=True)
    pool = sorted_exs[:min(len(sorted_exs), n_shots * 3)]
    selected = rng.sample(pool, min(n_shots, len(pool)))
    return _format_demos(selected, output_format, keys, syntax_enrichment)


def _format_demos(selected: list[dict], output_format: str, keys: list[str], syntax_enrichment: str = None) -> list[dict]:
    """Convert selected examples into chat messages."""
    messages = []
    format_fn = _format_example_nl if output_format == "natural-language" else _format_example_structured
    for ex in selected:
        syntax = ex.get("_syntax_compact") if syntax_enrichment else None
        input_text = _build_input_text(ex["sentence"], syntax)
        output_text = format_fn(ex, keys)
        messages.append({"role": "user", "content": input_text})
        messages.append({"role": "assistant", "content": output_text})
    return messages


# ---------------------------------------------------------------------------
# Dynamic demonstration retrieval (per test sentence)
# ---------------------------------------------------------------------------

class DemoRetriever:
    """Retrieves demonstrations per test sentence using BM25, SimCSE, or hybrid."""

    def __init__(self, train_examples: list[dict], method: str = "hybrid", n_shots: int = 5,
                 output_format: str = "structured", keys: list[str] = None,
                 syntax_enrichment: str = None):
        self.train_examples = train_examples
        self.method = method
        self.n_shots = n_shots
        self.output_format = output_format
        self.keys = keys or ["aspect", "sentiment", "polarity"]
        self.syntax_enrichment = syntax_enrichment

        sentences = [ex["sentence"] for ex in train_examples]

        # BM25 index
        if method in ("bm25", "hybrid"):
            from rank_bm25 import BM25Okapi
            tokenized = [s.lower().split() for s in sentences]
            self.bm25 = BM25Okapi(tokenized)

        # SimCSE / sentence-transformers index
        if method in ("simcse", "hybrid"):
            from sentence_transformers import SentenceTransformer
            self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embeddings = self.st_model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

    def get_demos(self, test_sentence: str) -> list[dict]:
        """Retrieve demonstrations for a single test sentence."""
        if self.method == "bm25":
            selected = self._bm25_select(test_sentence, self.n_shots)
        elif self.method == "simcse":
            selected = self._simcse_select(test_sentence, self.n_shots)
        else:  # hybrid
            n_each = self.n_shots // 2
            n_bm25 = n_each
            n_sim = self.n_shots - n_each
            bm25_picks = self._bm25_select(test_sentence, n_bm25)
            sim_picks = self._simcse_select(test_sentence, n_sim, exclude=bm25_picks)
            # interleave
            selected = []
            for a, b in zip(bm25_picks, sim_picks):
                selected.extend([a, b])
            selected.extend(bm25_picks[len(sim_picks):])
            selected.extend(sim_picks[len(bm25_picks):])

        return _format_demos(selected, self.output_format, self.keys, self.syntax_enrichment)

    def _bm25_select(self, query: str, n: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = scores.argsort()[-n:][::-1]
        return [self.train_examples[i] for i in top_idx]

    def _simcse_select(self, query: str, n: int, exclude: list[dict] = None) -> list[dict]:
        import torch
        q_emb = self.st_model.encode(query, convert_to_tensor=True)
        scores = torch.nn.functional.cosine_similarity(q_emb.unsqueeze(0), self.embeddings)
        exclude_sents = {ex["sentence"] for ex in (exclude or [])}
        top_idx = scores.argsort(descending=True).cpu().tolist()
        selected = []
        for i in top_idx:
            if self.train_examples[i]["sentence"] not in exclude_sents:
                selected.append(self.train_examples[i])
                if len(selected) >= n:
                    break
        return selected


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def _infer_on_device(
    model,
    tokenizer,
    device: str,
    device_id: int,
    examples: list[dict],
    ds_keys: list[str],
    output_format: str,
    system_prompt: str,
    few_shot_messages: list[dict],
    syntax_enrichment: str,
    max_new_tokens: int,
    demo_retriever=None,
) -> list[list[dict]]:
    """Run inference on a pre-loaded model. Designed to be called from a thread."""
    results = []
    for i, ex in enumerate(examples):
        syntax = ex.get("_syntax_compact") if syntax_enrichment else None
        input_text = _build_input_text(ex["sentence"], syntax)

        # per-sentence demos if retriever is available, otherwise fixed
        demos = demo_retriever.get_demos(ex["sentence"]) if demo_retriever else few_shot_messages

        messages = [{"role": "user", "content": system_prompt}]
        messages.append({"role": "assistant", "content": "I understand. I'll extract aspect-based sentiment analysis triplets from the sentences you provide."})
        messages.extend(demos)
        messages.append({"role": "user", "content": input_text})

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        preds = parse_output(decoded, ds_keys, output_format)
        results.append(preds)

        if (i + 1) % 50 == 0:
            print(f"  [GPU {device_id}] {i+1}/{len(examples)}...")

    return results


def _load_models(model_name: str, n_gpus: int):
    """Load one model per GPU sequentially (not thread-safe)."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    models = []
    for gpu_id in range(n_gpus):
        device = f"cuda:{gpu_id}"
        print(f"  Loading model on {device}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device,
            dtype=torch.bfloat16,
        )
        model.eval()
        models.append(model)
        print(f"  [GPU {gpu_id}] ready — {torch.cuda.memory_allocated(gpu_id)/1e9:.1f} GB")
    return models, tokenizer


def run_gemma_inference(cfg: dict, output_dir: Path):
    """Run 0-shot or few-shot Gemma inference and evaluate."""
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_name = model_cfg.get("name", "google/gemma-4-E4B-it")
    n_shots = model_cfg.get("n_shots", 0)
    output_format = model_cfg.get("output_format", "structured")  # "structured" or "nl"
    syntax_enrichment = data_cfg.get("syntax_enrichment", None)
    max_new_tokens = model_cfg.get("max_new_tokens", 256)
    seed = cfg.get("seed", 42)
    n_gpus = min(torch.cuda.device_count(), model_cfg.get("n_gpus", 2))
    demo_selection = model_cfg.get("demo_selection", "hybrid")  # fixed, bm25, simcse, hybrid

    # determine task type from config
    test_cfg = cfg.get("test", {})
    datasets = test_cfg.get("datasets", [])
    scopes = test_cfg.get("scopes", [{"keys": ["aspect", "sentiment", "polarity"], "metrics": ["micro_f1"]}])

    # detect if quad task
    first_ds = datasets[0] if datasets else {}
    task_keys_str = first_ds.get("tasks", ["aspect", "sentiment", "polarity"])
    keys = task_keys_str if isinstance(task_keys_str, list) else list(task_keys_str)
    task_type = "quad" if "category" in keys else "triplet"

    print(f"\n{'='*60}")
    print(f"  Gemma Inference: {model_name}")
    print(f"  Mode: {n_shots}-shot | Format: {output_format} | Syntax: {syntax_enrichment or 'none'}")
    print(f"  Demo selection: {demo_selection}")
    print(f"  Task: {task_type} ({', '.join(keys)})")
    print(f"  GPUs: {n_gpus}")
    print(f"{'='*60}\n")

    # load training data for few-shot
    train_examples = []
    demo_retriever = None
    few_shot_messages = []
    if n_shots > 0:
        train_files = data_cfg.get("train_file", [])
        if isinstance(train_files, str):
            train_files = [train_files]
        spacy_model = data_cfg.get("spacy_model", "en_core_web_sm")
        for f in train_files:
            if f.endswith(".jsonl"):
                train_examples.extend(load_acos_jsonl(f))
            else:
                train_examples.extend(load_aste_file(f))
        if syntax_enrichment:
            train_examples = enrich_syntax(train_examples, syntax_enrichment, spacy_model=spacy_model)

        if demo_selection == "fixed":
            few_shot_messages = _build_few_shot_messages(
                train_examples, n_shots, output_format, keys,
                syntax_enrichment=syntax_enrichment, seed=seed,
            )
        else:
            print(f"  Building {demo_selection} retriever over {len(train_examples)} training examples...")
            demo_retriever = DemoRetriever(
                train_examples, method=demo_selection, n_shots=n_shots,
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
            pass  # already a list of key names
        ds_scopes = ds_cfg.get("scopes", scopes)

        print(f"\nEvaluating: {data_path}")

        # load test data
        if data_path.endswith(".jsonl"):
            test_examples = load_acos_jsonl(data_path)
        else:
            test_examples = load_aste_file(data_path)

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

        # parallel inference across GPUs
        if n_gpus > 1:
            from concurrent.futures import ThreadPoolExecutor
            import time

            # load models sequentially (not thread-safe), then infer in parallel
            if not hasattr(run_gemma_inference, "_models"):
                print("Loading models on all GPUs...")
                run_gemma_inference._models, run_gemma_inference._tokenizer = _load_models(model_name, n_gpus)
            models = run_gemma_inference._models
            tokenizer = run_gemma_inference._tokenizer

            # split examples across GPUs
            chunk_size = len(test_examples) // n_gpus
            chunks = []
            for g in range(n_gpus):
                start = g * chunk_size
                end = start + chunk_size if g < n_gpus - 1 else len(test_examples)
                chunks.append(test_examples[start:end])

            print(f"  Splitting {len(test_examples)} examples across {n_gpus} GPUs...")
            t0 = time.time()

            with ThreadPoolExecutor(max_workers=n_gpus) as executor:
                futures = []
                for gpu_id, chunk in enumerate(chunks):
                    f = executor.submit(
                        _infer_on_device,
                        models[gpu_id], tokenizer, f"cuda:{gpu_id}", gpu_id,
                        chunk, ds_keys, output_format,
                        system_prompt, few_shot_messages, syntax_enrichment, max_new_tokens,
                        demo_retriever,
                    )
                    futures.append(f)
                # collect results in order
                all_preds = []
                for f in futures:
                    all_preds.extend(f.result())

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s ({elapsed/len(test_examples):.2f}s/example)")
        else:
            # single GPU path
            import time
            print("Loading model...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="cuda:0",
                dtype=torch.bfloat16,
            )
            model.eval()
            print(f"Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            all_preds = []
            t0 = time.time()
            for i, ex in enumerate(test_examples):
                syntax = ex.get("_syntax_compact") if syntax_enrichment else None
                input_text = _build_input_text(ex["sentence"], syntax)

                demos = demo_retriever.get_demos(ex["sentence"]) if demo_retriever else few_shot_messages

                messages = [{"role": "user", "content": system_prompt}]
                messages.append({"role": "assistant", "content": "I understand. I'll extract aspect-based sentiment analysis triplets from the sentences you provide."})
                messages.extend(demos)
                messages.append({"role": "user", "content": input_text})

                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                    )

                generated = output_ids[0][inputs["input_ids"].shape[1]:]
                decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
                preds = parse_output(decoded, ds_keys, output_format)
                all_preds.append(preds)

                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(test_examples)}...")

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s ({elapsed/len(test_examples):.2f}s/example)")

        # evaluate
        metrics = evaluate(all_preds, all_golds, ds_scopes)
        entry = {"data": data_path, **metrics}
        test_metrics_history.append(entry)

        # print summary
        for scope_key, scope_metrics in metrics.items():
            if isinstance(scope_metrics, dict):
                for metric_name, values in scope_metrics.items():
                    if isinstance(values, dict) and "f1" in values:
                        print(f"  {scope_key}/{metric_name}: P={values['precision']:.4f} R={values['recall']:.4f} F1={values['f1']:.4f}")

    # save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    import yaml
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    for i, entry in enumerate(test_metrics_history):
        data_label = Path(entry["data"]).stem
        metrics = {k: v for k, v in entry.items() if k != "data"}
        save_metrics_table(metrics, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")

    save_results([], test_metrics_history, [], [], str(results_dir))
    print(f"\nResults saved to: {results_dir}")
