"""
XLM-RoBERTa multi-label classifier for category+polarity prediction.

Each sentence can have multiple (category, polarity) pairs. We model this as
multi-label classification over all category×polarity combinations.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from src.eval.eval import evaluate, save_results, save_metrics_table


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CategoryPolarityDataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer, max_length: int, label2id: dict):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.n_labels = len(label2id)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex["sentence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # multi-hot label vector
        labels = torch.zeros(self.n_labels, dtype=torch.float)
        for ann in ex["annotations"]:
            cat = ann.get("category", "")
            pol = ann.get("polarity", "")
            key = f"{cat}:{pol}"
            if key in self.label2id:
                labels[self.label2id[key]] = 1.0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class XLMRClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_length: int = 256,
        batch_size: int = 16,
        val_batch_size: int = 32,
        warmup_ratio: float = 0.1,
        num_workers: int = 11,
        categories: list[str] = None,
        polarities: list[str] = None,
        train_examples: list[dict] = None,
        val_examples: list[dict] = None,
        test_examples: list[dict] = None,
        eval_scopes: list[dict] = None,
        test_scopes: list[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_examples", "val_examples", "test_examples"])

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        # build label space: category × polarity
        categories = categories or ["experience", "durability", "performance", "battery",
                                     "price_quality", "design", "camera", "audio",
                                     "software", "brand", "service"]
        polarities = polarities or ["positive", "negative", "neutral"]

        self.id2label = []
        self.label2id = {}
        for cat in categories:
            for pol in polarities:
                key = f"{cat}:{pol}"
                self.label2id[key] = len(self.id2label)
                self.id2label.append(key)

        n_labels = len(self.id2label)
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, n_labels),
        )

        self._train_examples = train_examples or []
        self._val_examples = val_examples or []
        self._test_examples = test_examples or []
        self._eval_scopes = eval_scopes or [{"keys": ["polarity", "category"], "metrics": ["micro_f1"]}]
        self._test_scopes = test_scopes or self._eval_scopes

        self._val_preds: list[list[dict]] = []
        self._val_golds: list[list[dict]] = []
        self._test_preds: list[list[dict]] = []
        self._test_golds: list[list[dict]] = []
        self.val_metrics_history: list[dict] = []
        self.test_metrics_history: list[dict] = []
        self.train_loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._results_dir: str | None = None
        self._current_test_data: str = ""

    def set_test_data(self, examples: list[dict], scopes: list[dict], data_path: str = ""):
        self._test_examples = examples
        self._test_scopes = scopes
        self._current_test_data = data_path

    def _make_dataset(self, examples):
        return CategoryPolarityDataset(examples, self.tokenizer, self.hparams.max_length, self.label2id)

    def train_dataloader(self):
        ds = self._make_dataset(self._train_examples)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        ds = self._make_dataset(self._val_examples)
        return DataLoader(ds, batch_size=self.hparams.val_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        ds = self._make_dataset(self._test_examples)
        return DataLoader(ds, batch_size=self.hparams.val_batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def forward(self, batch):
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        # CLS token representation
        cls_repr = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_repr)
        return logits

    def _compute_loss(self, logits, labels):
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self._compute_loss(logits, batch["labels"])
        self._train_losses.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self._compute_loss(logits, batch["labels"])
        self._val_losses.append(loss.item())
        # decode predictions
        preds_batch, golds_batch = self._decode_batch(logits, batch["labels"])
        self._val_preds.extend(preds_batch)
        self._val_golds.extend(golds_batch)

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        preds_batch, golds_batch = self._decode_batch(logits, batch["labels"])
        self._test_preds.extend(preds_batch)
        self._test_golds.extend(golds_batch)

    def _decode_batch(self, logits: torch.Tensor, labels: torch.Tensor):
        """Convert logits and multi-hot labels to list-of-dicts format."""
        probs = torch.sigmoid(logits)
        preds_batch = []
        golds_batch = []
        for i in range(logits.size(0)):
            # predictions: threshold at 0.5
            pred_indices = (probs[i] > 0.5).nonzero(as_tuple=True)[0].tolist()
            preds = []
            for idx in pred_indices:
                cat, pol = self.id2label[idx].split(":")
                preds.append({"category": cat, "polarity": pol})
            preds_batch.append(preds)

            # golds
            gold_indices = labels[i].nonzero(as_tuple=True)[0].tolist()
            golds = []
            for idx in gold_indices:
                cat, pol = self.id2label[idx].split(":")
                golds.append({"category": cat, "polarity": pol})
            golds_batch.append(golds)

        return preds_batch, golds_batch

    def on_validation_epoch_end(self):
        metrics = evaluate(self._val_preds, self._val_golds, self._eval_scopes)
        # extract primary F1 for checkpointing
        primary_key = "+".join(self._eval_scopes[0]["keys"])
        primary_f1 = metrics.get(primary_key, {}).get("micro", {}).get("f1", 0.0)
        self.log("val_f1", primary_f1, prog_bar=True)

        avg_loss = sum(self._val_losses) / max(len(self._val_losses), 1)
        avg_train_loss = sum(self._train_losses) / max(len(self._train_losses), 1)
        self.train_loss_history.append(avg_train_loss)
        self.val_loss_history.append(avg_loss)
        self.val_metrics_history.append(metrics)

        print(f"  Epoch {self.current_epoch}: val_f1={primary_f1:.4f} loss={avg_loss:.4f}")

        self._val_preds.clear()
        self._val_golds.clear()
        self._val_losses.clear()
        self._train_losses.clear()

    def on_test_epoch_end(self):
        metrics = evaluate(self._test_preds, self._test_golds, self._test_scopes)
        entry = {"data": self._current_test_data, **metrics}
        self.test_metrics_history.append(entry)

        for scope_key, scope_metrics in metrics.items():
            if isinstance(scope_metrics, dict):
                for metric_name, values in scope_metrics.items():
                    if isinstance(values, dict) and "f1" in values:
                        print(f"  {scope_key}/{metric_name}: P={values['precision']:.4f} R={values['recall']:.4f} F1={values['f1']:.4f}")

        self._test_preds.clear()
        self._test_golds.clear()

    def configure_optimizers(self):
        no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
        params = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)

        from transformers import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
