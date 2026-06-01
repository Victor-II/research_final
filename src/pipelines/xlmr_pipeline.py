"""
Pipeline for XLM-RoBERTa multi-label classifier (category+polarity).
"""

import logging
import random
from pathlib import Path

import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data.data import load_files, load_train_val
from src.model.xlmr_classifier import XLMRClassifier
from src.eval.eval import save_results, save_metrics_table


def run_xlmr_classifier(cfg: dict, output_dir: Path):
    """Train and evaluate XLM-RoBERTa classifier."""
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg["seed"], workers=True)

    print(f"\n{'='*60}")
    print(f"  XLM-R Classifier: {cfg.get('name', 'unnamed')}")
    print(f"{'='*60}\n")

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    trainer_cfg = cfg["trainer"]

    train_examples, val_examples = load_train_val(cfg)
    fi = data_cfg.get("filter_implicit", False)
    eval_cfg = cfg["eval"]

    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    # build model
    categories = model_cfg.get("categories", None)
    polarities = model_cfg.get("polarities", None)

    model = XLMRClassifier(
        model_name=model_cfg["name"],
        learning_rate=model_cfg["learning_rate"],
        weight_decay=model_cfg.get("weight_decay", 0.01),
        max_length=model_cfg.get("max_length", 256),
        batch_size=model_cfg["batch_size"],
        val_batch_size=model_cfg.get("val_batch_size", 32),
        warmup_ratio=model_cfg.get("warmup_ratio", 0.1),
        num_workers=trainer_cfg.get("num_workers", 11),
        categories=categories,
        polarities=polarities,
        train_examples=train_examples,
        val_examples=val_examples,
        eval_scopes=eval_cfg["scopes"],
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
            test_data = load_files(ds["data"], filter_implicit=fi)
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
