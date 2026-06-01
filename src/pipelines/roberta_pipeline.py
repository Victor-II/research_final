"""
Pipeline for RoBERTa span extraction model.

Handles data loading, training, and evaluation using the same data loaders
and evaluation infrastructure as the T5 pipeline.
"""

import logging
from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.data import load_files, load_train_val
from src.model.roberta_span import RoBERTaSpanModel, SpanExtractionDataset, _collate_fn
from src.eval.eval import save_results, save_metrics_table


def run_roberta(cfg: dict, output_dir: Path):
    """Train and evaluate RoBERTa span extraction model."""
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg["seed"], workers=True)

    print(f"\n{'='*60}")
    print(f"  RoBERTa Span Extraction: {cfg.get('name', 'unnamed')}")
    print(f"{'='*60}\n")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]

    train_examples, val_examples = load_train_val(cfg)
    syntax_enrichment = data_cfg.get("syntax_enrichment", None)
    fi = data_cfg.get("filter_implicit", False)

    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    # syntax field name for dataset
    syntax_field = "_syntax" if syntax_enrichment else None

    # build model
    model = RoBERTaSpanModel(
        model_name=model_cfg["name"],
        learning_rate=model_cfg["learning_rate"],
        weight_decay=model_cfg.get("weight_decay", 0.01),
        max_length=model_cfg.get("max_length", 128),
        batch_size=model_cfg["batch_size"],
        val_batch_size=model_cfg.get("val_batch_size", 32),
        warmup_ratio=model_cfg.get("warmup_ratio", 0.1),
        span_hidden=model_cfg.get("span_hidden", 256),
        num_workers=trainer_cfg.get("num_workers", 11),
        syntax_field=syntax_field,
        train_examples=train_examples,
        val_examples=val_examples,
        eval_scopes=cfg["eval"]["scopes"],
    )

    # setup output
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)
    model._results_dir = str(results_dir)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir), filename="best",
        monitor="val_f1", mode="max", save_top_k=1,
    )
    callbacks = [checkpoint_cb]
    es_patience = trainer_cfg.get("early_stopping_patience", 0)
    if es_patience > 0:
        callbacks.append(EarlyStopping(monitor="val_f1", mode="max", patience=es_patience))

    trainer = pl.Trainer(
        max_epochs=trainer_cfg["max_epochs"],
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        accumulate_grad_batches=trainer_cfg.get("accumulate_grad_batches", 1),
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 1),
        num_sanity_val_steps=0,
        deterministic=trainer_cfg.get("deterministic", True),
        enable_model_summary=False,
        logger=False,
        callbacks=callbacks,
        devices=1,
    )

    trainer.fit(model)

    # test phase
    if "test" in cfg:
        test_cfg = cfg["test"]
        default_scopes = test_cfg["scopes"]

        test_trainer = pl.Trainer(devices=1, num_nodes=1, enable_model_summary=False)
        ckpt = str(ckpt_dir / "best.ckpt")

        for ds in test_cfg["datasets"]:
            scopes = ds.get("scopes", default_scopes)
            test_data = load_files(ds["data"], filter_implicit=fi, syntax_enrichment=syntax_enrichment, spacy_model=data_cfg.get("spacy_model", "en_core_web_sm"))
            model.set_test_data(test_data, scopes, ds["data"])
            test_trainer.test(model, ckpt_path=ckpt)
            ckpt = None

        for i, entry in enumerate(model.test_metrics_history):
            data_label = Path(entry["data"]).stem
            metrics = {k: v for k, v in entry.items() if k != "data"}
            save_metrics_table(metrics, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")

    save_results(
        model.val_metrics_history,
        model.test_metrics_history,
        model.train_loss_history,
        model.val_loss_history,
        str(results_dir),
    )
    print(f"\nResults saved to: {results_dir}")


def test_roberta(cfg: dict, checkpoint: str, output_dir: Path):
    """Test-only mode for RoBERTa model."""
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("medium")

    test_cfg = cfg["test"]
    default_scopes = test_cfg["scopes"]
    fi = cfg.get("data", {}).get("filter_implicit", False)

    first_ds = test_cfg["datasets"][0]
    first_data = load_files(first_ds["data"], filter_implicit=fi)

    model = RoBERTaSpanModel.load_from_checkpoint(
        checkpoint,
        test_examples=first_data,
        test_scopes=first_ds.get("scopes", default_scopes),
    )
    model._current_test_data = first_ds["data"]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / "results"
    results_dir.mkdir(exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    trainer = pl.Trainer(devices=1, num_nodes=1, enable_model_summary=False)
    trainer.test(model)

    for ds in test_cfg["datasets"][1:]:
        scopes = ds.get("scopes", default_scopes)
        test_data = load_files(ds["data"], filter_implicit=fi)
        model.set_test_data(test_data, scopes, ds["data"])
        trainer.test(model)

    for i, entry in enumerate(model.test_metrics_history):
        data_label = Path(entry["data"]).stem
        metrics = {k: v for k, v in entry.items() if k != "data"}
        save_metrics_table(metrics, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")

    save_results([], model.test_metrics_history, [], [], str(results_dir))
