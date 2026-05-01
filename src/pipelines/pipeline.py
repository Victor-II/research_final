import logging
from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.data import split_by_task, load_aste_file, load_silviolima_domain, load_acos_jsonl, to_generative_format, filter_implicit_aspects, enrich_syntax
from src.model.model import T5ABSAModel
from src.eval.eval import save_results, save_metrics_table
from src.augment.registry import apply_augmentations
from constants import TASK_KEY_MAP, Task


def _resolve_tasks(task_keys: list[str]) -> list[Task]:
    return [Task[k.upper()] for k in task_keys]


def _load_data(file_path: str, filter_implicit: bool = False, syntax_enrichment: str = None) -> list[dict]:
    if file_path.endswith(".jsonl"):
        examples = load_acos_jsonl(file_path)
    elif file_path.endswith(".json"):
        examples = load_silviolima_domain(file_path)
    else:
        examples = load_aste_file(file_path)
    if filter_implicit:
        examples = filter_implicit_aspects(examples)
    if syntax_enrichment:
        examples = enrich_syntax(examples, syntax_enrichment)
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
    for f in train_files:
        canonical_all.extend(_load_data(f, filter_implicit=fi, syntax_enrichment=se))

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

    # curriculum config
    curriculum = data_cfg.get("curriculum", None)

    train_examples = [
        ex
        for part in split_by_task(
            train_files[0],
            tasks_partition,
            shuffle_tasks=shuffle_tasks,
            examples=augmented_train,
            nl_fraction=nl_fraction,
            infer_implicit=data_cfg.get("infer_implicit", False),
        ).values()
        for ex in part
    ]

    # pass config so model can re-split (and re-augment) each epoch
    task_split_cfg = None
    if needs_resplit or has_augmentation or nl_fraction > 0 or curriculum:
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
        }

    val_tasks = _resolve_tasks(cfg["eval"].get("tasks", ["aspect", "sentiment", "polarity"]))
    val_format = cfg["eval"].get("output_format", "structured")
    if canonical_val is not None:
        val_examples = [to_generative_format(ex, val_tasks, output_format=val_format) for ex in canonical_val]
    else:
        val_examples = [to_generative_format(ex, val_tasks, output_format=val_format) for ex in _load_data(cfg["eval"]["data"], filter_implicit=fi, syntax_enrichment=se)]

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
        label_smoothing=m.get("label_smoothing", 0.0),
        num_workers=cfg["trainer"].get("num_workers", 11),
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
    trainer = pl.Trainer(
        max_epochs=t["max_epochs"],
        precision=t["precision"],
        accumulate_grad_batches=t["accumulate_grad_batches"],
        gradient_clip_val=t.get("gradient_clip_val", 1.0),
        log_every_n_steps=t["log_every_n_steps"],
        limit_train_batches=t["limit_train_batches"],
        num_sanity_val_steps=t["num_sanity_val_steps"],
        reload_dataloaders_every_n_epochs=t["reload_dataloaders_every_n_epochs"],
        deterministic=t["deterministic"],
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
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
    test_format = test_cfg.get("output_format", "structured")

    model = T5ABSAModel.load_from_checkpoint(
        str(ckpt_path),
        test_examples=[to_generative_format(ex, first_tasks, output_format=test_format) for ex in _load_data(first["data"], filter_implicit=fi, syntax_enrichment=se)],
        test_scopes=first.get("scopes", default_scopes),
    )
    model._current_test_data = first["data"]
    model._eval_implicit_split = test_cfg.get("eval_implicit_split", False)

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
        model.set_test_data(
            [to_generative_format(ex, ds_tasks, output_format=test_format) for ex in _load_data(ds["data"], filter_implicit=fi, syntax_enrichment=se)],
            scopes,
            ds["data"],
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
    test_format = test_cfg.get("output_format", "structured")

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
        model.set_test_data(
            [to_generative_format(ex, ds_tasks, output_format=test_format) for ex in _load_data(ds["data"], filter_implicit=fi, syntax_enrichment=se)],
            scopes,
            ds["data"],
        )
        test_trainer.test(model, ckpt_path=ckpt)
        ckpt = None  # only load checkpoint on first call

    for i, entry in enumerate(model.test_metrics_history):
        data_label = Path(entry["data"]).stem
        metrics = {k: v for k, v in entry.items() if k != "data"}
        save_metrics_table(metrics, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")
