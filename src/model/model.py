import json
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AutoTokenizer
from src.eval.eval import parse_output, evaluate
from src.data.data import ABSADataset
from src.model.utils import gather_floats, gather_string_lists


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class T5ABSAModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        max_length: int = 256,
        batch_size: int = 4,
        val_batch_size: int = 16,
        warmup_ratio: float = 0.06,
        lr_scheduler: str = "cosine",
        max_new_tokens: int = 64,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        label_smoothing: float = 0.0,
        num_workers: int = 11,
        do_sample: bool = False,
        temperature: float = 1.0,
        num_return_sequences: int = 1,
        vote_threshold: int = 1,
        train_examples: list[dict] = None,
        val_examples: list[dict] = None,
        test_examples: list[dict] = None,
        eval_scopes: list[dict] = None,
        test_scopes: list[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_examples", "val_examples"])

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._train_examples = train_examples or []
        self._val_examples   = val_examples   or []
        self._test_examples  = test_examples  or []
        self._task_split_cfg = None  # set by pipeline for per-epoch re-splitting

        self._eval_scopes = eval_scopes or [{"keys": ["aspect", "sentiment", "polarity"], "metrics": ["micro_f1"]}]
        self._test_scopes = test_scopes or self._eval_scopes

        # accumulators reset each epoch
        self._val_losses: list[float] = []
        self._train_losses: list[float] = []
        self._val_preds: list[list[dict]] = []
        self._val_golds: list[list[dict]] = []
        self._test_preds: list[list[dict]] = []
        self._test_golds: list[list[dict]] = []
        self.val_metrics_history: list[dict] = []
        self.test_metrics_history: list[dict] = []
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self._results_dir: str | None = None  # set by pipeline for incremental saving

    def set_test_data(self, examples: list[dict], scopes: list[dict], data_path: str = ""):
        self._test_examples = examples
        self._test_scopes = scopes
        self._current_test_data = data_path

    def on_train_start(self):
        self.model.train()

    def train_dataloader(self):
        if self._task_split_cfg is not None:
            from src.data.data import split_by_task, interpolate_curriculum
            from src.augment.registry import apply_augmentations
            cfg = self._task_split_cfg
            seed = cfg["seed"] + self.current_epoch

            canonical = cfg["canonical"]
            if cfg.get("aug_cfg"):
                aug_cfg_copy = {"data": {"augmentation": cfg["aug_cfg"]}, "seed": seed}
                canonical = apply_augmentations(list(canonical), aug_cfg_copy)

            # resolve task partition (curriculum or fixed)
            if cfg.get("curriculum"):
                from constants import TASK_KEY_MAP
                raw_partition = interpolate_curriculum(cfg["curriculum"], self.current_epoch)
                tasks_partition = {TASK_KEY_MAP[k]: v for k, v in raw_partition.items() if v > 0}
            else:
                tasks_partition = cfg["tasks_partition"]

            examples = [
                ex
                for part in split_by_task(
                    cfg["file_path"],
                    tasks_partition,
                    shuffle_tasks=cfg["shuffle_tasks"],
                    examples=canonical,
                    seed=seed,
                    nl_fraction=cfg.get("nl_fraction", 0.0),
                    infer_implicit=cfg.get("infer_implicit", False),
                ).values()
                for ex in part
            ]
        else:
            examples = self._train_examples
        ds = ABSADataset(examples, self.tokenizer, self.hparams.max_length)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, persistent_workers=True)

    def val_dataloader(self):
        ds = ABSADataset(self._val_examples, self.tokenizer, self.hparams.max_length)
        return DataLoader(ds, batch_size=self.hparams.val_batch_size,
                          num_workers=self.hparams.num_workers, persistent_workers=True)

    def test_dataloader(self):
        ds = ABSADataset(self._test_examples, self.tokenizer, self.hparams.max_length)
        return DataLoader(ds, batch_size=self.hparams.val_batch_size,
                          num_workers=self.hparams.num_workers, persistent_workers=True)

    # --- steps ---

    def training_step(self, batch, batch_idx):
        self.model.train()
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        if self.hparams.label_smoothing > 0:
            logits = out.logits
            labels = batch["labels"]
            loss_fn = torch.nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=self.hparams.label_smoothing,
            )
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            loss = out.loss
        self._train_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
        self._val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=self.hparams.batch_size)

        output_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.hparams.max_new_tokens,
            num_beams=self.hparams.num_beams,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
        )
        for i, out in enumerate(output_ids):
            decoded = self.tokenizer.decode(out, skip_special_tokens=True)
            gold    = batch["raw_target"][i]
            keys = batch["keys"][i].split(",")
            fmt = batch["output_format"][i]
            self._val_preds.append(parse_output(decoded, keys, fmt))
            self._val_golds.append(parse_output(gold, keys, fmt))

        return loss

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return

        import torch.distributed as dist

        all_preds = self._gather_string_lists(self._val_preds)
        all_golds = self._gather_string_lists(self._val_golds)
        all_val_losses = self._gather_floats(self._val_losses)

        first_scope = "+".join(self._eval_scopes[0]["keys"])
        val_f1 = torch.tensor(0.0, device=self.device)

        if self.global_rank == 0:
            metrics = evaluate(all_preds, all_golds, self._eval_scopes)
            val_f1.fill_(metrics[first_scope]["micro"]["f1"])
            self.val_metrics_history.append({"epoch": self.current_epoch, **metrics})
            self.train_loss_history.append(sum(self._train_losses) / len(self._train_losses) if self._train_losses else 0.0)
            self.val_loss_history.append(sum(all_val_losses) / len(all_val_losses) if all_val_losses else 0.0)

        # Broadcast val_f1 from rank 0 so all ranks log the same value
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(val_f1, src=0)

        self.log("val_f1", val_f1, prog_bar=True, sync_dist=False)

        # save training history incrementally to disk (survives DDP)
        if self.global_rank == 0 and self._results_dir:
            import json as _json
            history = {
                "train_loss": self.train_loss_history,
                "val_loss": self.val_loss_history,
                "val": self.val_metrics_history,
            }
            with open(f"{self._results_dir}/_history.json", "w") as f:
                _json.dump(history, f, indent=2)

        self._val_preds.clear()
        self._val_golds.clear()
        self._val_losses.clear()
        self._train_losses.clear()

    def test_step(self, batch, batch_idx):
        n_seq = self.hparams.num_return_sequences
        gen_kwargs = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.hparams.max_new_tokens,
            num_beams=self.hparams.num_beams,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
        )
        if self.hparams.do_sample:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.hparams.temperature

        if n_seq <= 1:
            output_ids = self.model.generate(**gen_kwargs)
            for i, out in enumerate(output_ids):
                decoded = self.tokenizer.decode(out, skip_special_tokens=True)
                gold    = batch["raw_target"][i]
                keys    = batch["keys"][i].split(",")
                fmt     = batch["output_format"][i]
                self._test_preds.append(parse_output(decoded, keys, fmt))
                self._test_golds.append(parse_output(gold, keys, fmt))
        else:
            gen_kwargs["num_return_sequences"] = n_seq
            if self.hparams.num_beams < n_seq and not self.hparams.do_sample:
                gen_kwargs["num_beams"] = n_seq
            output_ids = self.model.generate(**gen_kwargs)
            bsz = batch["input_ids"].size(0)
            for i in range(bsz):
                gold = batch["raw_target"][i]
                keys = batch["keys"][i].split(",")
                fmt  = batch["output_format"][i]
                # collect triplets from all N sequences, count occurrences
                from collections import Counter
                counts = Counter()
                for j in range(n_seq):
                    decoded = self.tokenizer.decode(output_ids[i * n_seq + j], skip_special_tokens=True)
                    for triplet in parse_output(decoded, keys, fmt):
                        counts[frozenset(triplet.items())] += 1
                voted = [dict(k) for k, c in counts.items() if c >= self.hparams.vote_threshold]
                self._test_preds.append(voted)
                self._test_golds.append(parse_output(gold, keys, fmt))

    def on_test_epoch_end(self):
        if not self._test_preds:
            return
        if self.global_rank == 0:
            metrics = evaluate(self._test_preds, self._test_golds, self._test_scopes)
            entry = {"data": getattr(self, "_current_test_data", ""), **metrics}

            if getattr(self, "_eval_implicit_split", False):
                explicit_preds, explicit_golds = [], []
                implicit_preds, implicit_golds = [], []
                for pred_triplets, gold_triplets in zip(self._test_preds, self._test_golds):
                    ep = [t for t in pred_triplets if t.get("aspect") != "IMPLICIT"]
                    ip = [t for t in pred_triplets if t.get("aspect") == "IMPLICIT"]
                    eg = [t for t in gold_triplets if t.get("aspect") != "IMPLICIT"]
                    ig = [t for t in gold_triplets if t.get("aspect") == "IMPLICIT"]
                    explicit_preds.append(ep)
                    explicit_golds.append(eg)
                    implicit_preds.append(ip)
                    implicit_golds.append(ig)
                if any(g for g in explicit_golds):
                    explicit_metrics = evaluate(explicit_preds, explicit_golds, self._test_scopes)
                    entry["explicit"] = explicit_metrics
                if any(g for g in implicit_golds):
                    implicit_metrics = evaluate(implicit_preds, implicit_golds, self._test_scopes)
                    entry["implicit"] = implicit_metrics

            self.test_metrics_history.append(entry)
        self._test_preds.clear()
        self._test_golds.clear()

    def _gather_floats(self, local: list[float]) -> list[float]:
        return gather_floats(local, self.device)

    def _gather_string_lists(self, local: list[list[dict]]) -> list[list[dict]]:
        return gather_string_lists(local, self.device)

    # --- optimizer ---

    def configure_optimizers(self):
        from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, get_constant_schedule
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        schedulers = {
            "cosine": lambda: get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps),
            "linear": lambda: get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps),
            "constant": lambda: get_constant_schedule(optimizer),
        }
        scheduler = schedulers[self.hparams.lr_scheduler]()
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    # --- inference ---

    def predict(self, text: str, max_new_tokens: int = 128) -> list[dict]:
        """Run inference on a raw input string and return parsed triplets."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        try:
            return json.loads(decoded)
        except json.JSONDecodeError:
            return [{"raw": decoded}]
