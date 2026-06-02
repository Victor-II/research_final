import logging
from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.data import split_by_task, load_aste_file, load_silviolima_domain, load_acos_jsonl, load_emag_csv, to_generative_format, filter_implicit_aspects, enrich_syntax, extract_categories, mvp_multiply_orderings
from src.model.t5_model import T5ABSAModel
from src.eval.eval import save_results, save_metrics_table
from src.augment.registry import apply_augmentations
from constants import TASK_KEY_MAP, Task


def _resolve_tasks(task_keys: list[str]) -> list[Task]:
    return [Task[k.upper()] for k in task_keys]


def _load_data(file_path: str, filter_implicit: bool = False, syntax_enrichment: str = None, spacy_model: str = "en_core_web_sm") -> list[dict]:
    if file_path.endswith(".jsonl"):
        examples = load_acos_jsonl(file_path)
    elif file_path.endswith(".json"):
        examples = load_silviolima_domain(file_path)
    elif file_path.endswith(".csv"):
        examples = load_emag_csv(file_path)
    else:
        examples = load_aste_file(file_path)
    if filter_implicit:
        examples = filter_implicit_aspects(examples)
    if syntax_enrichment:
        examples = enrich_syntax(examples, syntax_enrichment, spacy_model=spacy_model)
    return examples


def _prepare_data(cfg: dict):
    import random as _random
    data_cfg = cfg["data"]
    tasks_partition = {
        TASK_KEY_MAP[k]: v
        for k, v in data_cfg["tasks_partition"].items()
        if v > 0
    }

    train_files = data_cfg["train_file"]
    if isinstance(train_files, str):
        train_files = [train_files]
    canonical_all = []
    fi = data_cfg.get("filter_implicit", False)
    se = data_cfg.get("syntax_enrichment", None)
    spacy_model = data_cfg.get("spacy_model", "en_core_web_sm")
    language = data_cfg.get("language", "en")
    for f in train_files:
        if isinstance(f, dict):
            file_path = f["path"]
            fraction = f.get("fraction", 1.0)
            examples = _load_data(file_path, filter_implicit=fi, syntax_enrichment=se, spacy_model=spacy_model)
            if fraction < 1.0:
                rng_f = _random.Random(cfg.get("seed", 42))
                n_keep = max(1, int(len(examples) * fraction))
                examples = rng_f.sample(examples, n_keep)
            canonical_all.extend(examples)
        else:
            canonical_all.extend(_load_data(f, filter_implicit=fi, syntax_enrichment=se, spacy_model=spacy_model))

    # train_fraction: subsample training data
    train_fraction = data_cfg.get("train_fraction", 1.0)
    if train_fraction < 1.0:
        rng_frac = _random.Random(cfg["seed"])
        n_keep = max(1, int(len(canonical_all) * train_fraction))
        canonical_all = rng_frac.sample(canonical_all, n_keep)

    # val split from training data
    val_split = cfg["eval"].get("val_split", 0)
    if val_split > 0:
        rng = _random.Random(cfg["seed"])
        indices = list(range(len(canonical_all)))
        rng.shuffle(indices)
        n_val = int(len(canonical_all) * val_split)
        val_indices = set(indices[:n_val])
        canonical_train = [canonical_all[i] for i in range(len(canonical_all)) if i not in val_indices]
        canonical_val = [canonical_all[i] for i in indices[:n_val]]
    else:
        canonical_train = canonical_all
        canonical_val = None

    shuffle_tasks = data_cfg.get("shuffle_tasks", False)
    needs_resplit = len(tasks_partition) > 1
    aug_cfg = data_cfg.get("augmentation", {})
    has_augmentation = any(v for v in aug_cfg.values() if v)

    nl_fraction = data_cfg.get("natural_language_fraction", 0.0)

    augmented_train = apply_augmentations(canonical_train, cfg)

    # category set extraction (auto from training data if not specified)
    category_set = data_cfg.get("category_set", None)
    if category_set is None:
        cats = extract_categories(canonical_all)
        category_set = cats if cats else None

    # curriculum config
    curriculum = data_cfg.get("curriculum", None)

    bare_prompt = data_cfg.get("bare_prompt", False)

    include_categories = data_cfg.get("include_categories", False)

    train_examples = [
        ex
        for part in split_by_task(
            train_files[0],
            tasks_partition,
            shuffle_tasks=shuffle_tasks,
            examples=augmented_train,
            nl_fraction=nl_fraction,
            infer_implicit=data_cfg.get("infer_implicit", False),
            category_set=category_set,
            bare_prompt=bare_prompt,
            language=language,
            include_categories=include_categories,
        ).values()
        for ex in part
    ]

    # mix in auxiliary syntax prediction examples
    syntax_aux_fraction = data_cfg.get("syntax_auxiliary_fraction", 0.0)
    syntax_aux_tasks = data_cfg.get("syntax_auxiliary_tasks", ["dep"])
    if syntax_aux_fraction > 0 and se:
        from src.data.data import to_syntax_auxiliary, enrich_syntax_auxiliary
        aux_canonical = enrich_syntax_auxiliary(list(canonical_train))
        n_aux = int(len(train_examples) * syntax_aux_fraction)
        rng_aux = _random.Random(cfg["seed"])
        for _ in range(n_aux):
            src_ex = rng_aux.choice(aux_canonical)
            task = rng_aux.choice(syntax_aux_tasks)
            aux_ex = to_syntax_auxiliary(src_ex, task=task)
            if aux_ex:
                train_examples.append(aux_ex)

    # mix in auxiliary opinion-prediction examples
    opinion_aux_fraction = data_cfg.get("opinion_auxiliary_fraction", 0.0)
    if opinion_aux_fraction > 0:
        from src.augment.masking import opinion_prediction_aux
        aux_opinion = opinion_prediction_aux(
            canonical_train,
            fraction=opinion_aux_fraction,
            seed=cfg["seed"],
        )
        train_examples.extend(aux_opinion)

    # STAR-style data multiplication: sub-task variants as additional training examples
    star_fraction = data_cfg.get("star_multiply_fraction", 0.0)
    if star_fraction > 0:
        from src.augment.star import star_multiply
        star_examples = star_multiply(
            canonical_train,
            fraction=star_fraction,
            output_format="natural-language" if nl_fraction > 0 else "structured",
            language=language,
            seed=cfg["seed"],
        )
        train_examples.extend(star_examples)

    # MvP top-k ordering multiplication (replaces normal training when mvp-markers format)
    mvp_top_k = data_cfg.get("mvp_top_k", 0)
    output_format = data_cfg.get("output_format", None)
    if mvp_top_k > 0 and output_format == "mvp-markers":
        # resolve the main task tuple from partition (use the largest weight group)
        main_task_key = max(tasks_partition, key=lambda k: tasks_partition[k])
        main_tasks = list(main_task_key)
        mvp_examples = mvp_multiply_orderings(
            augmented_train,
            tasks=main_tasks,
            output_format="mvp-markers",
            top_k=mvp_top_k,
            language=language,
            seed=cfg["seed"],
        )
        # replace training examples with MvP orderings (tagged _level="main")
        train_examples = mvp_examples

    # STAR pairwise relation examples
    star_pairwise_flag = data_cfg.get("star_pairwise", False)
    if star_pairwise_flag:
        from src.augment.star import star_pairwise
        pairwise_format = output_format if output_format == "mvp-markers" else ("natural-language" if nl_fraction > 0 else "structured")
        pairwise_examples = star_pairwise(
            canonical_train,
            output_format=pairwise_format,
            language=language,
            seed=cfg["seed"],
        )
        train_examples.extend(pairwise_examples)

    # pass config so model can re-split (and re-augment) each epoch
    task_split_cfg = None
    if needs_resplit or has_augmentation or nl_fraction > 0 or curriculum or mvp_top_k > 0 or star_pairwise_flag:
        task_split_cfg = {
            "file_path": train_files[0],
            "tasks_partition": tasks_partition,
            "shuffle_tasks": shuffle_tasks,
            "canonical": canonical_train,
            "seed": cfg["seed"],
            "aug_cfg": aug_cfg if has_augmentation else None,
            "nl_fraction": nl_fraction,
            "infer_implicit": data_cfg.get("infer_implicit", False),
            "curriculum": curriculum,
            "category_set": category_set,
            "syntax_auxiliary_fraction": syntax_aux_fraction,
            "syntax_auxiliary_tasks": syntax_aux_tasks,
            "syntax_enrichment": se,
            "bare_prompt": bare_prompt,
            "language": language,
            "include_categories": include_categories,
            "opinion_auxiliary_fraction": opinion_aux_fraction,
            "star_multiply_fraction": star_fraction,
            "mvp_top_k": mvp_top_k,
            "output_format": output_format,
            "star_pairwise": star_pairwise_flag,
        }

    val_tasks = _resolve_tasks(cfg["eval"].get("tasks", ["aspect", "sentiment", "polarity"]))
    val_format = cfg["eval"].get("output_format", data_cfg.get("output_format") or "structured")
    if canonical_val is not None:
        val_examples = [to_generative_format(ex, val_tasks, output_format=val_format, category_set=category_set, bare_prompt=bare_prompt, language=language, include_categories=include_categories, nl_fraction=nl_fraction) for ex in canonical_val]
    else:
        from src.data.data import load_files
        canonical_val = load_files(cfg["eval"]["data"], filter_implicit=fi, syntax_enrichment=se, spacy_model=spacy_model)
        val_examples = [to_generative_format(ex, val_tasks, output_format=val_format, category_set=category_set, bare_prompt=bare_prompt, language=language, include_categories=include_categories, nl_fraction=nl_fraction) for ex in canonical_val]

    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")
    return train_examples, val_examples, task_split_cfg


def _build_model(cfg: dict, train_examples: list[dict], val_examples: list[dict]) -> T5ABSAModel:
    m = cfg["model"]
    g = cfg.get("generation", {})
    return T5ABSAModel(
        model_name=m["name"],
        learning_rate=m["learning_rate"],
        weight_decay=m.get("weight_decay", 0.01),
        max_length=m["max_length"],
        batch_size=m["batch_size"],
        val_batch_size=m["val_batch_size"],
        warmup_ratio=m["warmup_ratio"],
        lr_scheduler=m.get("lr_scheduler", "cosine"),
        max_new_tokens=m["max_new_tokens"],
        num_beams=g.get("num_beams", 1),
        repetition_penalty=g.get("repetition_penalty", 1.0),
        length_penalty=g.get("length_penalty", 1.0),
        do_sample=g.get("do_sample", False),
        temperature=g.get("temperature", 1.0),
        num_return_sequences=g.get("num_return_sequences", 1),
        vote_threshold=g.get("vote_threshold", 1),
        penalty_alpha=g.get("penalty_alpha", 0.0),
        top_k=g.get("top_k", 0),
        num_beam_groups=g.get("num_beam_groups", 1),
        diversity_penalty=g.get("diversity_penalty", 0.0),
        constrained_decoding=g.get("constrained_decoding", False),
        structured_attention=cfg.get("data", {}).get("structured_attention", False),
        focal_gamma=m.get("focal_gamma", 0.0),
        optimizer=cfg.get("model", {}).get("optimizer", "adamw"),
        label_smoothing=m.get("label_smoothing", 0.0),
        num_workers=cfg["trainer"].get("num_workers", 11),
        language=cfg.get("data", {}).get("language", "en"),
        train_examples=train_examples,
        val_examples=val_examples,
        eval_scopes=cfg["eval"]["scopes"],
    )


def _setup_output(cfg: dict, output_dir: Path):
    # output_dir is pre-created by main.py (before DDP spawning)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    config_path = output_dir / "config.yaml"
    if not config_path.exists():
        with open(config_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return ckpt_dir, results_dir


def run(cfg: dict, output_dir: Path):
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg["seed"], workers=True)

    print(f"\n{'='*60}")
    print(f"  Experiment: {cfg.get('name', 'unnamed')}")
    print(f"{'='*60}\n")

    train_examples, val_examples, task_split_cfg = _prepare_data(cfg)
    model = _build_model(cfg, train_examples, val_examples)
    if task_split_cfg is not None:
        model._task_split_cfg = task_split_cfg
    model._star_balanced_loss = cfg.get("data", {}).get("star_balanced_loss", False)
    ckpt_dir, results_dir = _setup_output(cfg, output_dir)
    model._results_dir = str(results_dir)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )

    t = cfg["trainer"]
    callbacks = [checkpoint_cb]
    es_patience = t.get("early_stopping_patience", 0)
    if es_patience > 0:
        callbacks.append(EarlyStopping(monitor="val_f1", mode="max", patience=es_patience))
    resume_from = t.get("from_checkpoint") or None

    strategy = t.get("strategy", "auto")
    if strategy == "fsdp":
        from pytorch_lightning.strategies import FSDPStrategy
        from transformers.models.t5.modeling_t5 import T5Block
        strategy = FSDPStrategy(
            auto_wrap_policy={T5Block},
            activation_checkpointing_policy={T5Block},
            state_dict_type="full",
            limit_all_gathers=True,
        )

    trainer = pl.Trainer(
        max_epochs=t["max_epochs"],
        precision=t["precision"],
        accumulate_grad_batches=t["accumulate_grad_batches"],
        gradient_clip_val=t.get("gradient_clip_val", 1.0) if strategy == "auto" else None,
        log_every_n_steps=t["log_every_n_steps"],
        limit_train_batches=t["limit_train_batches"],
        num_sanity_val_steps=t["num_sanity_val_steps"],
        reload_dataloaders_every_n_epochs=t["reload_dataloaders_every_n_epochs"],
        deterministic=t["deterministic"],
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
        strategy=strategy,
    )

    trainer.fit(model, ckpt_path=resume_from)

    # save training history immediately (before test phase may interfere with DDP state)
    _history_file = results_dir / "_history.json"
    if _history_file.exists():
        import json as _json
        with open(_history_file) as f:
            _train_history = _json.load(f)
    else:
        _train_history = {
            "val": list(model.val_metrics_history),
            "train_loss": list(model.train_loss_history),
            "val_loss": list(model.val_loss_history),
        }

    # test phase (after training)
    if "test" in cfg:
        model._eval_implicit_split = cfg["test"].get("eval_implicit_split", False)
        model._implicit_mode = cfg["test"].get("implicit_mode", None)
        _run_test(cfg, model, ckpt_dir / "best.ckpt", results_dir, checkpoint_cb=checkpoint_cb)

    save_results(
        _train_history["val"],
        model.test_metrics_history,
        _train_history["train_loss"],
        _train_history["val_loss"],
        str(results_dir),
    )


def test(cfg: dict, checkpoint: str, output_dir: Path):
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("medium")

    if "test" not in cfg:
        raise ValueError("No 'test' block found in config.")

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_cfg = cfg["test"]
    default_scopes = test_cfg["scopes"]
    first = test_cfg["datasets"][0]
    first_tasks = _resolve_tasks(first.get("tasks", ["aspect", "sentiment", "polarity"]))
    fi = cfg.get("data", {}).get("filter_implicit", False)
    se = cfg.get("data", {}).get("syntax_enrichment", None)
    spacy_model = cfg.get("data", {}).get("spacy_model", "en_core_web_sm")
    language = cfg.get("data", {}).get("language", "en")
    include_categories = cfg.get("data", {}).get("include_categories", False)
    nl_fraction = cfg.get("data", {}).get("natural_language_fraction", 0.0)
    test_format = test_cfg.get("output_format", "structured")
    bp = cfg.get("data", {}).get("bare_prompt", False)

    def _test_category_set(ds_cfg):
        cs = ds_cfg.get("category_set") or test_cfg.get("category_set")
        if cs:
            return cs
        # auto-extract from test data
        test_data = _load_data(ds_cfg["data"], filter_implicit=fi, syntax_enrichment=se, spacy_model=spacy_model)
        cats = extract_categories(test_data)
        return cats if cats else None

    first_cats = _test_category_set(first)
    model = T5ABSAModel.load_from_checkpoint(
        str(ckpt_path),
        test_examples=[to_generative_format(ex, first_tasks, output_format=test_format, category_set=first_cats, bare_prompt=bp, language=language, include_categories=include_categories, nl_fraction=nl_fraction) for ex in _load_data(first["data"], filter_implicit=fi, syntax_enrichment=se, spacy_model=spacy_model)],
        test_scopes=first.get("scopes", default_scopes),
        **{k: v for k, v in cfg.get("generation", {}).items() if v is not None},
    )
    model._current_test_data = first["data"]
    model._category_set = first_cats
    model._eval_implicit_split = test_cfg.get("eval_implicit_split", False)
    model._implicit_mode = test_cfg.get("implicit_mode", None)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    trainer = pl.Trainer(devices=1, num_nodes=1, enable_model_summary=False)
    trainer.test(model)

    for ds in test_cfg["datasets"][1:]:
        scopes = ds.get("scopes", default_scopes)
        ds_tasks = _resolve_tasks(ds.get("tasks", ["aspect", "sentiment", "polarity"]))
        ds_cats = _test_category_set(ds)
        model.set_test_data(
            [to_generative_format(ex, ds_tasks, output_format=test_format, category_set=ds_cats, bare_prompt=bp, language=language, include_categories=include_categories, nl_fraction=nl_fraction) for ex in _load_data(ds["data"], filter_implicit=fi, syntax_enrichment=se, spacy_model=spacy_model)],
            scopes,
            ds["data"],
            category_set=ds_cats,
        )
        trainer.test(model)

    for i, entry in enumerate(model.test_metrics_history):
        data_label = Path(entry["data"]).stem
        metrics = {k: v for k, v in entry.items() if k != "data"}
        save_metrics_table(metrics, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")

    save_results([], model.test_metrics_history, [], [], str(results_dir))


def _run_test(cfg: dict, model: T5ABSAModel, ckpt_path: Path, results_dir: Path, checkpoint_cb=None):
    t = cfg["trainer"]
    test_cfg = cfg["test"]
    default_scopes = test_cfg["scopes"]
    fi = cfg.get("data", {}).get("filter_implicit", False)
    se = cfg.get("data", {}).get("syntax_enrichment", None)
    spacy_model = cfg.get("data", {}).get("spacy_model", "en_core_web_sm")
    language = cfg.get("data", {}).get("language", "en")
    include_categories = cfg.get("data", {}).get("include_categories", False)
    nl_fraction = cfg.get("data", {}).get("natural_language_fraction", 0.0)
    test_format = test_cfg.get("output_format", "structured")
    bare_prompt = cfg.get("data", {}).get("bare_prompt", False)

    callbacks = [checkpoint_cb] if checkpoint_cb else []
    test_trainer = pl.Trainer(
        devices=1, num_nodes=1,
        precision=t["precision"],
        enable_model_summary=False,
        callbacks=callbacks,
    )
    ckpt = str(ckpt_path)
    for ds in test_cfg["datasets"]:
        scopes = ds.get("scopes", default_scopes)
        ds_tasks = _resolve_tasks(ds.get("tasks", ["aspect", "sentiment", "polarity"]))
        test_data = _load_data(ds["data"], filter_implicit=fi, syntax_enrichment=se, spacy_model=spacy_model)
        ds_cats = ds.get("category_set") or test_cfg.get("category_set")
        if not ds_cats:
            cats = extract_categories(test_data)
            ds_cats = cats if cats else None
        model.set_test_data(
            [to_generative_format(ex, ds_tasks, output_format=test_format, category_set=ds_cats, bare_prompt=bare_prompt, language=language, include_categories=include_categories, nl_fraction=nl_fraction) for ex in test_data],
            scopes,
            ds["data"],
            category_set=ds_cats,
        )
        test_trainer.test(model, ckpt_path=ckpt)
        ckpt = None  # only load checkpoint on first call

    for i, entry in enumerate(model.test_metrics_history):
        data_label = Path(entry["data"]).stem
        metrics = {k: v for k, v in entry.items() if k != "data"}
        save_metrics_table(metrics, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")


def test_mvp(cfg: dict, checkpoint: str, output_dir: Path):
    """
    MvP-style multi-prompt voting test.

    For each test example, runs inference with multiple permutations of the
    element ordering and aggregates results via majority voting.
    Supports both NL/structured format (task-order permutation in prompt)
    and mvp-markers format (marker ordering permutation in input).
    """
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("medium")

    from itertools import permutations as _perms
    from collections import Counter
    from src.eval.eval import parse_output, evaluate, save_results, save_metrics_table
    from src.eval.mvp_vote import generate_multiview_inputs, vote_predictions

    if "test" not in cfg:
        raise ValueError("No 'test' block found in config.")

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    test_cfg = cfg["test"]
    default_scopes = test_cfg["scopes"]
    fi = cfg.get("data", {}).get("filter_implicit", False)
    se = cfg.get("data", {}).get("syntax_enrichment", None)
    spacy_model = cfg.get("data", {}).get("spacy_model", "en_core_web_sm")
    language = cfg.get("data", {}).get("language", "en")
    test_format = test_cfg.get("output_format", cfg.get("data", {}).get("output_format", "structured"))
    bp = cfg.get("data", {}).get("bare_prompt", False)
    nl_fraction = cfg.get("data", {}).get("natural_language_fraction", 0.0)
    include_categories = cfg.get("data", {}).get("include_categories", False)
    mvp_top_k = cfg.get("data", {}).get("mvp_top_k", 5)
    vote_threshold = test_cfg.get("mvp_vote_threshold", (mvp_top_k + 1) // 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # load model once
    model = T5ABSAModel.load_from_checkpoint(str(ckpt_path))
    model.eval()
    model.cuda()
    tokenizer = model.tokenizer

    gen_kwargs = {
        "max_new_tokens": cfg.get("model", {}).get("max_new_tokens", 64),
        "num_beams": cfg.get("generation", {}).get("num_beams", 1),
    }

    test_metrics_history = []

    for ds_cfg in test_cfg["datasets"]:
        data_path = ds_cfg["data"]
        scopes = ds_cfg.get("scopes", default_scopes)

        canonical_data = _load_data(data_path, filter_implicit=fi, syntax_enrichment=se, spacy_model=spacy_model)

        all_preds = []
        all_golds = []

        for ex in canonical_data:
            # generate gold
            gold_tasks = _resolve_tasks(ds_cfg.get("tasks", ["aspect", "sentiment", "polarity"]))
            gold_gen = to_generative_format(ex, gold_tasks, output_format=test_format, bare_prompt=bp, language=language, include_categories=include_categories, nl_fraction=nl_fraction)
            gold_parsed = parse_output(gold_gen["target"], gold_gen["_keys"], gold_gen["_format"], language=language)

            # generate multiview inputs
            variants = generate_multiview_inputs(
                ex,
                output_format=test_format,
                language=language,
                top_k=mvp_top_k,
                seed=cfg.get("seed", 42),
            )

            # run inference for each variant
            predictions_per_view = []
            for gen in variants:
                inputs = tokenizer(gen["input"], return_tensors="pt", max_length=cfg.get("model", {}).get("max_length", 256), truncation=True).to(model.device)
                output_ids = model.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], **gen_kwargs)
                decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                preds = parse_output(decoded, gen["_keys"], gen["_format"], language=language)
                # normalize to canonical key set for voting
                canonical_preds = []
                for triplet in preds:
                    canonical_triplet = {k: triplet.get(k, "") for k in ["aspect", "sentiment", "polarity"] if k in triplet}
                    if canonical_triplet:
                        canonical_preds.append(canonical_triplet)
                predictions_per_view.append(canonical_preds)

            # vote
            voted = vote_predictions(predictions_per_view, threshold=vote_threshold)
            all_preds.append(voted)
            all_golds.append(gold_parsed)

        # evaluate
        metrics = evaluate(all_preds, all_golds, scopes)
        entry = {"data": data_path, **metrics}
        test_metrics_history.append(entry)

        # print summary
        triplet_key = "aspect+sentiment+polarity"
        if triplet_key in metrics:
            micro = metrics[triplet_key].get("micro", {})
            print(f"  {data_path}: triplet-F1 = {micro.get('f1', 0):.4f} (MvP vote, k={mvp_top_k}, threshold={vote_threshold})")

    # save
    for i, entry in enumerate(test_metrics_history):
        data_label = Path(entry["data"]).stem
        metrics_only = {k: v for k, v in entry.items() if k != "data"}
        save_metrics_table(metrics_only, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")

    save_results([], test_metrics_history, [], [], str(results_dir))
