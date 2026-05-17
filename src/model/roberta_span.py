"""
RoBERTa-based span extraction model for ABSA.

Architecture:
- RoBERTa encoder for contextual token representations
- Two BIO tagging heads: one for aspect spans, one for sentiment spans
- Biaffine scorer to pair aspect and sentiment spans
- Polarity classifier on paired (aspect, sentiment) representations

This produces (aspect, sentiment, polarity) triplets comparable to the
generative T5 approach, enabling direct comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer

from src.eval.eval import evaluate, save_results, save_metrics_table


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class SpanExtractionDataset(Dataset):
    """Tokenizes sentences and aligns BIO tags to subword tokens."""

    def __init__(self, examples: list[dict], tokenizer: AutoTokenizer, max_length: int = 128, syntax_field: str = None):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.syntax_field = syntax_field  # e.g. "_syntax" for dep-compact
        self._cache = [None] * len(examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self._cache[idx] is not None:
            return self._cache[idx]

        ex = self.examples[idx]
        tokens = ex["tokens"]
        annotations = ex["annotations"]

        # optional syntax as text_pair
        syntax_text = ex.get(self.syntax_field) if self.syntax_field else None

        # tokenize with word-level alignment
        if syntax_text:
            encoding = self.tokenizer(
                tokens,
                syntax_text.split(),
                is_split_into_words=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=False,
            )
        else:
            encoding = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_offsets_mapping=False,
            )

        word_ids = encoding.word_ids()
        seq_len = self.max_length

        # build BIO labels for aspects and sentiments
        aspect_bio = [0] * seq_len   # 0=O, 1=B-ASP, 2=I-ASP
        sentiment_bio = [0] * seq_len  # 0=O, 1=B-OPN, 2=I-OPN

        # track span positions for pairing
        aspect_spans = []  # (start_subword, end_subword, annotation_idx)
        sentiment_spans = []

        for ann_idx, ann in enumerate(annotations):
            # aspect span
            if ann.get("aspect_idx") and ann["aspect"] != "IMPLICIT":
                asp_word_indices = set(ann["aspect_idx"])
                span_start = None
                for sw_idx, wid in enumerate(word_ids):
                    if wid in asp_word_indices:
                        if span_start is None:
                            aspect_bio[sw_idx] = 1  # B
                            span_start = sw_idx
                        else:
                            aspect_bio[sw_idx] = 2  # I
                    else:
                        if span_start is not None:
                            aspect_spans.append((span_start, sw_idx - 1, ann_idx))
                            span_start = None
                if span_start is not None:
                    aspect_spans.append((span_start, seq_len - 1, ann_idx))

            # sentiment span
            if ann.get("sentiment_idx") and ann["sentiment"] != "IMPLICIT":
                opn_word_indices = set(ann["sentiment_idx"])
                span_start = None
                for sw_idx, wid in enumerate(word_ids):
                    if wid in opn_word_indices:
                        if span_start is None:
                            sentiment_bio[sw_idx] = 1  # B
                            span_start = sw_idx
                        else:
                            sentiment_bio[sw_idx] = 2  # I
                    else:
                        if span_start is not None:
                            sentiment_spans.append((span_start, sw_idx - 1, ann_idx))
                            span_start = None
                if span_start is not None:
                    sentiment_spans.append((span_start, seq_len - 1, ann_idx))

        # build pairing matrix and polarity labels
        # pair_matrix[i, j] = 1 if aspect_span i and sentiment_span j belong to same annotation
        # polarity_labels[i, j] = polarity class (0=pos, 1=neg, 2=neu)
        n_asp = len(aspect_spans)
        n_opn = len(sentiment_spans)
        max_spans = 20  # cap for memory

        pair_matrix = torch.zeros(max_spans, max_spans)
        polarity_labels = torch.full((max_spans, max_spans), -100, dtype=torch.long)

        polarity_map = {"positive": 0, "negative": 1, "neutral": 2}

        for i, (_, _, ann_i) in enumerate(aspect_spans[:max_spans]):
            for j, (_, _, ann_j) in enumerate(sentiment_spans[:max_spans]):
                if ann_i == ann_j:
                    pair_matrix[i, j] = 1.0
                    pol = annotations[ann_i].get("polarity", "neutral")
                    polarity_labels[i, j] = polarity_map.get(pol, 2)

        # store span positions (padded)
        aspect_span_pos = torch.zeros(max_spans, 2, dtype=torch.long)
        sentiment_span_pos = torch.zeros(max_spans, 2, dtype=torch.long)
        for i, (s, e, _) in enumerate(aspect_spans[:max_spans]):
            aspect_span_pos[i] = torch.tensor([s, e])
        for i, (s, e, _) in enumerate(sentiment_spans[:max_spans]):
            sentiment_span_pos[i] = torch.tensor([s, e])

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "aspect_bio": torch.tensor(aspect_bio, dtype=torch.long),
            "sentiment_bio": torch.tensor(sentiment_bio, dtype=torch.long),
            "pair_matrix": pair_matrix,
            "polarity_labels": polarity_labels,
            "aspect_span_pos": aspect_span_pos,
            "sentiment_span_pos": sentiment_span_pos,
            "n_aspect_spans": min(n_asp, max_spans),
            "n_sentiment_spans": min(n_opn, max_spans),
            # for decoding at test time
            "_tokens": tokens,
            "_annotations": annotations,
            "_word_ids": word_ids,
        }
        self._cache[idx] = result
        return result


def _collate_fn(batch):
    """Custom collate that handles variable-length metadata."""
    keys_to_stack = ["input_ids", "attention_mask", "aspect_bio", "sentiment_bio",
                     "pair_matrix", "polarity_labels", "aspect_span_pos", "sentiment_span_pos"]
    result = {}
    for k in keys_to_stack:
        result[k] = torch.stack([b[k] for b in batch])
    result["n_aspect_spans"] = [b["n_aspect_spans"] for b in batch]
    result["n_sentiment_spans"] = [b["n_sentiment_spans"] for b in batch]
    result["_tokens"] = [b["_tokens"] for b in batch]
    result["_annotations"] = [b["_annotations"] for b in batch]
    result["_word_ids"] = [b["_word_ids"] for b in batch]
    return result


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BiaffineScorer(nn.Module):
    """Biaffine attention for span pairing."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Bilinear(hidden_size, hidden_size, 1, bias=True)

    def forward(self, asp_repr: torch.Tensor, opn_repr: torch.Tensor) -> torch.Tensor:
        # asp_repr: (batch, n_asp, hidden), opn_repr: (batch, n_opn, hidden)
        # output: (batch, n_asp, n_opn)
        batch, n_asp, h = asp_repr.shape
        n_opn = opn_repr.shape[1]
        asp_exp = asp_repr.unsqueeze(2).expand(-1, -1, n_opn, -1).reshape(-1, h)
        opn_exp = opn_repr.unsqueeze(1).expand(-1, n_asp, -1, -1).reshape(-1, h)
        scores = self.W(asp_exp, opn_exp).reshape(batch, n_asp, n_opn)
        return scores


class RoBERTaSpanModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "roberta-base",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        max_length: int = 128,
        batch_size: int = 16,
        val_batch_size: int = 32,
        warmup_ratio: float = 0.1,
        span_hidden: int = 256,
        num_workers: int = 11,
        syntax_field: str = None,
        train_examples: list[dict] = None,
        val_examples: list[dict] = None,
        test_examples: list[dict] = None,
        eval_scopes: list[dict] = None,
        test_scopes: list[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_examples", "val_examples", "test_examples"])

        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        hidden = self.encoder.config.hidden_size

        # BIO tagging heads
        self.aspect_head = nn.Linear(hidden, 3)  # O, B-ASP, I-ASP
        self.sentiment_head = nn.Linear(hidden, 3)  # O, B-OPN, I-OPN

        # span representation (start + end pooling)
        self.asp_span_proj = nn.Sequential(
            nn.Linear(hidden * 2, span_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.opn_span_proj = nn.Sequential(
            nn.Linear(hidden * 2, span_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # pairing scorer
        self.pair_scorer = BiaffineScorer(span_hidden)

        # polarity classifier (on concatenated asp+opn representations)
        self.polarity_head = nn.Sequential(
            nn.Linear(span_hidden * 2, span_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(span_hidden, 3),  # pos, neg, neu
        )

        self._train_examples = train_examples or []
        self._val_examples = val_examples or []
        self._test_examples = test_examples or []
        self._eval_scopes = eval_scopes or [{"keys": ["aspect", "sentiment", "polarity"], "metrics": ["micro_f1"]}]
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

    def train_dataloader(self):
        ds = SpanExtractionDataset(self._train_examples, self.tokenizer, self.hparams.max_length,
                                   syntax_field=self.hparams.get("syntax_field"))
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0,
                          collate_fn=_collate_fn)

    def val_dataloader(self):
        ds = SpanExtractionDataset(self._val_examples, self.tokenizer, self.hparams.max_length,
                                   syntax_field=self.hparams.get("syntax_field"))
        return DataLoader(ds, batch_size=self.hparams.val_batch_size,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0,
                          collate_fn=_collate_fn)

    def test_dataloader(self):
        ds = SpanExtractionDataset(self._test_examples, self.tokenizer, self.hparams.max_length,
                                   syntax_field=self.hparams.get("syntax_field"))
        return DataLoader(ds, batch_size=self.hparams.val_batch_size,
                          num_workers=self.hparams.num_workers,
                          persistent_workers=self.hparams.num_workers > 0,
                          collate_fn=_collate_fn)

    def _get_span_repr(self, hidden_states: torch.Tensor, span_pos: torch.Tensor, n_spans: list[int]) -> torch.Tensor:
        """Extract span representations by concatenating start and end token embeddings."""
        batch_size = hidden_states.shape[0]
        max_spans = span_pos.shape[1]
        h = hidden_states.shape[2]
        repr_list = []
        for b in range(batch_size):
            spans = []
            for s in range(max_spans):
                start, end = span_pos[b, s, 0].item(), span_pos[b, s, 1].item()
                if s < n_spans[b]:
                    span_repr = torch.cat([hidden_states[b, start], hidden_states[b, end]])
                else:
                    span_repr = torch.zeros(h * 2, device=hidden_states.device)
                spans.append(span_repr)
            repr_list.append(torch.stack(spans))
        return torch.stack(repr_list)  # (batch, max_spans, hidden*2)

    def forward(self, batch):
        outputs = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        self._last_hidden = hidden  # cache for decode

        # BIO logits
        asp_logits = self.aspect_head(hidden)  # (batch, seq_len, 3)
        opn_logits = self.sentiment_head(hidden)  # (batch, seq_len, 3)

        # span representations from gold positions (during training)
        asp_repr = self.asp_span_proj(
            self._get_span_repr(hidden, batch["aspect_span_pos"], batch["n_aspect_spans"])
        )
        opn_repr = self.opn_span_proj(
            self._get_span_repr(hidden, batch["sentiment_span_pos"], batch["n_sentiment_spans"])
        )

        # pairing scores
        pair_scores = self.pair_scorer(asp_repr, opn_repr)  # (batch, max_asp, max_opn)

        # polarity logits for all pairs
        batch_size, n_asp, h = asp_repr.shape
        n_opn = opn_repr.shape[1]
        asp_exp = asp_repr.unsqueeze(2).expand(-1, -1, n_opn, -1)
        opn_exp = opn_repr.unsqueeze(1).expand(-1, n_asp, -1, -1)
        pair_repr = torch.cat([asp_exp, opn_exp], dim=-1)  # (batch, n_asp, n_opn, h*2)
        pol_logits = self.polarity_head(pair_repr)  # (batch, n_asp, n_opn, 3)

        return asp_logits, opn_logits, pair_scores, pol_logits

    def _compute_loss(self, batch, asp_logits, opn_logits, pair_scores, pol_logits):
        # BIO loss (ignore padding tokens)
        mask = batch["attention_mask"].bool()
        asp_loss = F.cross_entropy(
            asp_logits[mask].view(-1, 3), batch["aspect_bio"][mask].view(-1)
        )
        opn_loss = F.cross_entropy(
            opn_logits[mask].view(-1, 3), batch["sentiment_bio"][mask].view(-1)
        )

        # pairing loss (BCE on valid span pairs)
        pair_loss = F.binary_cross_entropy_with_logits(
            pair_scores, batch["pair_matrix"]
        )

        # polarity loss (only on positive pairs)
        pol_labels = batch["polarity_labels"]
        valid_mask = pol_labels != -100
        if valid_mask.any():
            pol_loss = F.cross_entropy(
                pol_logits[valid_mask].view(-1, 3), pol_labels[valid_mask].view(-1)
            )
        else:
            pol_loss = torch.tensor(0.0, device=asp_logits.device)

        return asp_loss + opn_loss + pair_loss + pol_loss

    def training_step(self, batch, batch_idx):
        asp_logits, opn_logits, pair_scores, pol_logits = self(batch)
        loss = self._compute_loss(batch, asp_logits, opn_logits, pair_scores, pol_logits)
        self._train_losses.append(loss.item())
        self.log("train_loss", loss, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        asp_logits, opn_logits, pair_scores, pol_logits = self(batch)
        loss = self._compute_loss(batch, asp_logits, opn_logits, pair_scores, pol_logits)
        self._val_losses.append(loss.item())
        self.log("val_loss", loss, prog_bar=True, batch_size=self.hparams.batch_size)

        # decode predictions
        preds, golds = self._decode_batch(batch, asp_logits, opn_logits, pair_scores, pol_logits)
        self._val_preds.extend(preds)
        self._val_golds.extend(golds)
        return loss

    def test_step(self, batch, batch_idx):
        asp_logits, opn_logits, pair_scores, pol_logits = self(batch)
        preds, golds = self._decode_batch(batch, asp_logits, opn_logits, pair_scores, pol_logits)
        self._test_preds.extend(preds)
        self._test_golds.extend(golds)

    def _decode_batch(self, batch, asp_logits, opn_logits, pair_scores, pol_logits):
        """Decode BIO tags → spans → pairs → triplets."""
        polarity_map = {0: "positive", 1: "negative", 2: "neutral"}
        batch_preds = []
        batch_golds = []

        hidden = self._last_hidden  # use cached encoder output
        batch_size = asp_logits.shape[0]
        for b in range(batch_size):
            tokens = batch["_tokens"][b]
            word_ids = batch["_word_ids"][b]
            annotations = batch["_annotations"][b]

            # decode BIO tags
            asp_tags = asp_logits[b].argmax(dim=-1).cpu().tolist()
            opn_tags = opn_logits[b].argmax(dim=-1).cpu().tolist()

            # extract spans from BIO
            asp_spans = self._bio_to_spans(asp_tags, word_ids, tokens)
            opn_spans = self._bio_to_spans(opn_tags, word_ids, tokens)

            # if no spans found, empty prediction
            if not asp_spans or not opn_spans:
                batch_preds.append([])
            else:
                h_b = hidden[b]  # (seq_len, hidden_dim)
                h_dim = h_b.shape[-1]

                asp_reprs = []
                for (start_sw, end_sw, _) in asp_spans:
                    asp_reprs.append(torch.cat([h_b[start_sw], h_b[end_sw]]))
                asp_reprs = self.asp_span_proj(torch.stack(asp_reprs).unsqueeze(0))  # (1, n_asp, h)

                opn_reprs = []
                for (start_sw, end_sw, _) in opn_spans:
                    opn_reprs.append(torch.cat([h_b[start_sw], h_b[end_sw]]))
                opn_reprs = self.opn_span_proj(torch.stack(opn_reprs).unsqueeze(0))  # (1, n_opn, h)

                # pair scores
                p_scores = self.pair_scorer(asp_reprs, opn_reprs)[0]  # (n_asp, n_opn)

                # polarity
                n_a = asp_reprs.shape[1]
                n_o = opn_reprs.shape[1]
                a_exp = asp_reprs[0].unsqueeze(1).expand(-1, n_o, -1)
                o_exp = opn_reprs[0].unsqueeze(0).expand(n_a, -1, -1)
                pair_repr = torch.cat([a_exp, o_exp], dim=-1)
                p_logits = self.polarity_head(pair_repr)  # (n_a, n_o, 3)

                # threshold pairing and extract triplets
                triplets = []
                paired_mask = (p_scores > 0).cpu()  # sigmoid > 0.5 ↔ logit > 0
                for i in range(len(asp_spans)):
                    for j in range(len(opn_spans)):
                        if paired_mask[i, j]:
                            pol_idx = p_logits[i, j].argmax().item()
                            triplets.append({
                                "aspect": asp_spans[i][2],
                                "sentiment": opn_spans[j][2],
                                "polarity": polarity_map[pol_idx],
                            })
                batch_preds.append(triplets)

            # gold
            gold_triplets = []
            for ann in annotations:
                if ann.get("aspect") and ann.get("sentiment") and ann.get("polarity"):
                    gold_triplets.append({
                        "aspect": ann["aspect"],
                        "sentiment": ann["sentiment"],
                        "polarity": ann["polarity"],
                    })
            batch_golds.append(gold_triplets)

        return batch_preds, batch_golds

    def _bio_to_spans(self, tags: list[int], word_ids: list, tokens: list[str]) -> list[tuple]:
        """Convert BIO tag sequence to list of (start_subword, end_subword, text) spans."""
        spans = []
        current_start = None
        current_words = set()

        for sw_idx, (tag, wid) in enumerate(zip(tags, word_ids)):
            if tag == 1:  # B
                if current_start is not None:
                    text = " ".join(tokens[w] for w in sorted(current_words) if w < len(tokens))
                    spans.append((current_start, sw_idx - 1, text))
                current_start = sw_idx
                current_words = {wid} if wid is not None else set()
            elif tag == 2 and current_start is not None:  # I
                if wid is not None:
                    current_words.add(wid)
            else:
                if current_start is not None:
                    text = " ".join(tokens[w] for w in sorted(current_words) if w < len(tokens))
                    spans.append((current_start, sw_idx - 1, text))
                    current_start = None
                    current_words = set()

        if current_start is not None:
            text = " ".join(tokens[w] for w in sorted(current_words) if w < len(tokens))
            spans.append((current_start, len(tags) - 1, text))

        return spans

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        metrics = evaluate(self._val_preds, self._val_golds, self._eval_scopes)
        first_scope = "+".join(self._eval_scopes[0]["keys"])
        val_f1 = metrics[first_scope]["micro"]["f1"]
        self.log("val_f1", val_f1, prog_bar=True)
        self.val_metrics_history.append({"epoch": self.current_epoch, **metrics})
        self.train_loss_history.append(sum(self._train_losses) / len(self._train_losses) if self._train_losses else 0.0)
        self.val_loss_history.append(sum(self._val_losses) / len(self._val_losses) if self._val_losses else 0.0)

        if self._results_dir:
            import json
            history = {"train_loss": self.train_loss_history, "val_loss": self.val_loss_history, "val": self.val_metrics_history}
            with open(f"{self._results_dir}/_history.json", "w") as f:
                json.dump(history, f, indent=2)

        self._val_preds.clear()
        self._val_golds.clear()
        self._val_losses.clear()
        self._train_losses.clear()

    def on_test_epoch_end(self):
        if not self._test_preds:
            return
        metrics = evaluate(self._test_preds, self._test_golds, self._test_scopes)
        entry = {"data": self._current_test_data, **metrics}
        self.test_metrics_history.append(entry)
        self._test_preds.clear()
        self._test_golds.clear()

    def configure_optimizers(self):
        from transformers import get_cosine_schedule_with_warmup
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.hparams.learning_rate)
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
