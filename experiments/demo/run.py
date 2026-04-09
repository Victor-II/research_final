import logging
import re
import shutil
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from src.data.data import Task, split_by_task, load_aste_file, to_generative_format
from src.model.model import T5ABSAModel
from src.eval.eval import save_results, save_metrics_table
from src.augment.masking import mask_aspects

_TASK_KEY_MAP = {
    "aspect":   (Task.ASPECT,),
    "polarity": (Task.POLARITY,),
    "sentiment":(Task.SENTIMENT,),
    "full":     (Task.ASPECT, Task.POLARITY, Task.SENTIMENT),
}

_CONFIG    = Path(__file__).parent / "config" / "config.yaml"
_CKPT_DIR  = Path(__file__).parent / "checkpoints"
_RESULTS_DIR = Path(__file__).parent / "results"


def _next_run_id() -> int:
    existing = [
        int(m.group(1))
        for p in _CKPT_DIR.glob("run_*/best.ckpt")
        if (m := re.match(r"run_(\d+)", p.parent.name))
    ]
    return max(existing, default=0) + 1


def _latest_checkpoint() -> Path | None:
    candidates = [
        (int(m.group(1)), p)
        for p in _CKPT_DIR.glob("run_*/best.ckpt")
        if (m := re.match(r"run_(\d+)", p.parent.name))
    ]
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def _resolve_tasks(task_keys: list[str]) -> list[Task]:
    return [Task[k.upper()] for k in task_keys]


def run(cfg: dict = None):
    if cfg is None:
        with open(_CONFIG) as f:
            cfg = yaml.safe_load(f)

    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg["seed"], workers=True)

    data_cfg = cfg["data"]
    tasks_partition = {
        _TASK_KEY_MAP[k]: v
        for k, v in data_cfg["tasks_partition"].items()
    }

    canonical_train = load_aste_file(data_cfg["train_file"])
    aug_cfg = data_cfg.get("augmentation", {})
    if aug_cfg.get("mask_aspects"):
        m = aug_cfg["mask_aspects"]
        canonical_train = mask_aspects(
            canonical_train,
            fraction=m.get("fraction", 0.5),
            replace=m.get("replace", False),
            seed=cfg["seed"],
        )

    train_examples = [
        ex
        for part in split_by_task(
            data_cfg["train_file"],
            tasks_partition,
            shuffle_tasks=True,
            examples=canonical_train,
        ).values()
        for ex in part
    ]
    val_tasks = _resolve_tasks(cfg["eval"].get("tasks", ["aspect", "sentiment", "polarity"]))
    val_examples = [to_generative_format(ex, val_tasks) for ex in load_aste_file(cfg["eval"]["data"])]

    print(f"Train: {len(train_examples)} | Val: {len(val_examples)}")

    model_cfg = cfg["model"]
    model = T5ABSAModel(
        model_name=model_cfg["name"],
        learning_rate=model_cfg["learning_rate"],
        max_length=model_cfg["max_length"],
        batch_size=model_cfg["batch_size"],
        val_batch_size=model_cfg["val_batch_size"],
        warmup_ratio=model_cfg["warmup_ratio"],
        max_new_tokens=model_cfg["max_new_tokens"],
        train_examples=train_examples,
        val_examples=val_examples,
        eval_scopes=cfg["eval"]["scopes"],
    )

    run_id = _next_run_id()
    ckpt_dir = _CKPT_DIR / f"run_{run_id:03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir = _RESULTS_DIR / f"run_{run_id:03d}"
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_CONFIG, results_dir / "config.yaml")
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
    )

    t = cfg["trainer"]
    early_stop_cb = EarlyStopping(
        monitor="val_f1",
        mode="max",
        patience=t.get("early_stopping_patience", 3),
    )
    resume_from = t.get("from_checkpoint") or None
    trainer = pl.Trainer(
        max_epochs=t["max_epochs"],
        precision=t["precision"],
        accumulate_grad_batches=t["accumulate_grad_batches"],
        log_every_n_steps=t["log_every_n_steps"],
        limit_train_batches=t["limit_train_batches"],
        num_sanity_val_steps=t["num_sanity_val_steps"],
        reload_dataloaders_every_n_epochs=t["reload_dataloaders_every_n_epochs"],
        deterministic=t["deterministic"],
        enable_model_summary=False,
        logger=False,
        callbacks=[checkpoint_cb, early_stop_cb],
    )

    trainer.fit(model, ckpt_path=resume_from)

    if "test" in cfg:
        test_trainer = pl.Trainer(devices=1, num_nodes=1, precision=t["precision"], enable_model_summary=False, callbacks=[checkpoint_cb])
        ckpt_path = str(ckpt_dir / "best.ckpt")
        test_cfg = cfg["test"]
        default_scopes = test_cfg["scopes"]
        for ds in test_cfg["datasets"]:
            scopes = ds.get("scopes", default_scopes)
            ds_tasks = _resolve_tasks(ds.get("tasks", ["aspect", "sentiment", "polarity"]))
            model.set_test_data(
                [to_generative_format(ex, ds_tasks) for ex in load_aste_file(ds["data"])],
                scopes,
                ds["data"],
            )
            test_trainer.test(model, ckpt_path=ckpt_path)
            ckpt_path = None

        for i, entry in enumerate(model.test_metrics_history):
            data_label = Path(entry["data"]).stem
            metrics = {k: v for k, v in entry.items() if k != "data"}
            save_metrics_table(metrics, epoch=i, out_dir=str(results_dir), prefix=f"test_{data_label}")

    save_results(model.val_metrics_history, model.test_metrics_history, model.train_loss_history, model.val_loss_history, str(results_dir))


def test(cfg: dict = None, ckpt_path: str = None):
    if cfg is None:
        with open(_CONFIG) as f:
            cfg = yaml.safe_load(f)

    if "test" not in cfg:
        raise ValueError("No 'test' block found in config.")

    ckpt = ckpt_path or cfg["test"].get("from_checkpoint") or _latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError("No checkpoint found. Train first or specify 'from_checkpoint' in config.")

    test_cfg = cfg["test"]
    default_scopes = test_cfg["scopes"]
    first = test_cfg["datasets"][0]
    first_tasks = _resolve_tasks(first.get("tasks", ["aspect", "sentiment", "polarity"]))
    model = T5ABSAModel.load_from_checkpoint(
        str(ckpt),
        test_examples=[to_generative_format(ex, first_tasks) for ex in load_aste_file(first["data"])],
        test_scopes=first.get("scopes", default_scopes),
    )
    model._current_test_data = first["data"]

    trainer = pl.Trainer(devices=1, num_nodes=1, enable_model_summary=False)
    trainer.test(model)

    for ds in test_cfg["datasets"][1:]:
        scopes = ds.get("scopes", default_scopes)
        ds_tasks = _resolve_tasks(ds.get("tasks", ["aspect", "sentiment", "polarity"]))
        model.set_test_data(
            [to_generative_format(ex, ds_tasks) for ex in load_aste_file(ds["data"])],
            scopes,
            ds["data"],
        )
        trainer.test(model)

    ckpt_name = Path(str(ckpt)).stem
    results_dir = _RESULTS_DIR / ckpt_name
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(_CONFIG, results_dir / "config.yaml")
    save_results([], model.test_metrics_history, [], [], str(results_dir))


if __name__ == "__main__":
    run()
