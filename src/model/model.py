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
        max_length: int = 256,
        batch_size: int = 4,
        val_batch_size: int = 16,
        warmup_ratio: float = 0.06,
        max_new_tokens: int = 64,
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

    def set_test_data(self, examples: list[dict], scopes: list[dict], data_path: str = ""):
        self._test_examples = examples
        self._test_scopes = scopes
        self._current_test_data = data_path

    def on_train_start(self):
        self.model.train()

    def train_dataloader(self):
        ds = ABSADataset(self._train_examples, self.tokenizer, self.hparams.max_length)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=11, persistent_workers=True)

    def val_dataloader(self):
        ds = ABSADataset(self._val_examples, self.tokenizer, self.hparams.max_length)
        return DataLoader(ds, batch_size=self.hparams.val_batch_size,
                          num_workers=11, persistent_workers=True)

    def test_dataloader(self):
        ds = ABSADataset(self._test_examples, self.tokenizer, self.hparams.max_length)
        return DataLoader(ds, batch_size=self.hparams.val_batch_size,
                          num_workers=11, persistent_workers=True)

    # --- steps ---

    def training_step(self, batch, batch_idx):
        self.model.train()
        loss = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        ).loss
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
        )
        for i, out in enumerate(output_ids):
            decoded = self.tokenizer.decode(out, skip_special_tokens=True)
            gold    = batch["raw_target"][i]
            keys = batch["keys"][i].split(",")
            self._val_preds.append(parse_output(decoded, keys))
            self._val_golds.append(parse_output(gold, keys))
            if batch_idx == 0 and i == 0 and self.global_rank == 0:
                print(f"\n[DEBUG] pred : {decoded[:200]}")
                print(f"[DEBUG] gold : {gold[:200]}\n")

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

        self._val_preds.clear()
        self._val_golds.clear()
        self._val_losses.clear()
        self._train_losses.clear()

    def test_step(self, batch, batch_idx):
        output_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.hparams.max_new_tokens,
        )
        for i, out in enumerate(output_ids):
            decoded = self.tokenizer.decode(out, skip_special_tokens=True)
            gold    = batch["raw_target"][i]
            keys    = batch["keys"][i].split(",")
            self._test_preds.append(parse_output(decoded, keys))
            self._test_golds.append(parse_output(gold, keys))

    def on_test_epoch_end(self):
        if not self._test_preds:
            return
        if self.global_rank == 0:
            metrics = evaluate(self._test_preds, self._test_golds, self._test_scopes)
            self.test_metrics_history.append({"data": getattr(self, "_current_test_data", ""), **metrics})
        self._test_preds.clear()
        self._test_golds.clear()

    def _gather_floats(self, local: list[float]) -> list[float]:
        return gather_floats(local, self.device)

    def _gather_string_lists(self, local: list[list[dict]]) -> list[list[dict]]:
        return gather_string_lists(local, self.device)

    # --- optimizer ---

    def configure_optimizers(self):
        from transformers import get_linear_schedule_with_warmup
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
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
