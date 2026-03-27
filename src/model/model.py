import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration, AutoTokenizer
from src.eval.eval import parse_output, evaluate, save_metrics_table, run_all_plots


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ABSADataset(Dataset):
    """Wraps the list[dict] produced by split_by_task into a torch Dataset."""

    def __init__(self, examples: list[dict], tokenizer: AutoTokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inputs = self.tokenizer(
            ex["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            ex["target"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels,
            "raw_target": ex["target"],
            "raw_input": ex["input"],
            "keys": ",".join(ex.get("_keys", ["aspect", "sentiment", "polarity"])),
        }

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
        warmup_ratio: float = 0.06,
        train_examples: list[dict] = None,
        val_examples: list[dict] = None,
        eval_keys: list[list[str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_examples", "val_examples"])

        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._train_examples = train_examples or []
        self._val_examples   = val_examples   or []

        # Which key groups to evaluate, e.g. [["aspect"], ["aspect", "polarity"]]
        # Defaults to evaluating all keys found in the data
        self._eval_keys = eval_keys or [["aspect", "sentiment", "polarity"]]

        # accumulators reset each validation epoch
        self._val_losses: list[float] = []
        self._train_losses: list[float] = []
        self._val_preds: list[list[dict]] = []
        self._val_golds: list[list[dict]] = []

    # --- dataloaders ---

    def train_dataloader(self):
        ds = ABSADataset(self._train_examples, self.tokenizer, self.hparams.max_length)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=11, persistent_workers=True)

    def val_dataloader(self):
        ds = ABSADataset(self._val_examples, self.tokenizer, self.hparams.max_length)
        return DataLoader(ds, batch_size=self.hparams.batch_size,
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
            max_new_tokens=128,
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

        first_scope = "+".join(self._eval_keys[0])
        val_f1 = torch.tensor(0.0, device=self.device)

        if self.global_rank == 0:
            metrics = evaluate(all_preds, all_golds, self._eval_keys)
            val_f1.fill_(metrics[first_scope]["f1"])
            # run_all_plots(all_preds, all_golds, all_val_losses, self._eval_keys, epoch=self.current_epoch)
            save_metrics_table(metrics, epoch=self.current_epoch)

        # Broadcast val_f1 from rank 0 so all ranks log the same value
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.broadcast(val_f1, src=0)

        self.log("val_f1", val_f1, prog_bar=True, sync_dist=False)

        self._val_preds.clear()
        self._val_golds.clear()
        self._val_losses.clear()
        self._train_losses.clear()

    def _gather_floats(self, local: list[float]) -> list[float]:
        """Gather a flat list of floats across all DDP ranks."""
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return local
        t = torch.tensor(local, device=self.device)
        gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
        # sizes may differ per rank, so use all_gather via padding
        size = torch.tensor(len(local), device=self.device)
        sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, size)
        max_size = max(s.item() for s in sizes)
        buf = torch.zeros(max_size, device=self.device)
        buf[:len(local)] = t
        bufs = [torch.zeros(max_size, device=self.device) for _ in range(dist.get_world_size())]
        dist.all_gather(bufs, buf)
        merged = []
        for s, b in zip(sizes, bufs):
            merged.extend(b[:s.item()].cpu().tolist())
        return merged

    def _gather_string_lists(self, local: list[list[dict]]) -> list[list[dict]]:
        """Gather list-of-list-of-dicts across all DDP ranks via JSON serialisation."""
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
            return local

        # Serialise to a single JSON string per rank
        payload = json.dumps(local).encode()
        size    = torch.tensor(len(payload), device=self.device)

        # Exchange sizes
        sizes = [torch.zeros_like(size) for _ in range(dist.get_world_size())]
        dist.all_gather(sizes, size)

        # Exchange payloads
        max_size = max(s.item() for s in sizes)
        buf = torch.zeros(max_size, dtype=torch.uint8, device=self.device)
        buf[:len(payload)] = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
        bufs = [torch.zeros(max_size, dtype=torch.uint8, device=self.device)
                for _ in range(dist.get_world_size())]
        dist.all_gather(bufs, buf)

        merged = []
        for s, b in zip(sizes, bufs):
            merged.extend(json.loads(bytes(b[:s.item()].cpu().tolist()).decode()))
        return merged



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
