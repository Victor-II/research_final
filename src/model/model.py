import json
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.eval.eval import parse_output, evaluate
from src.data.data import ABSADataset
from src.model.utils import gather_floats, gather_string_lists


def _patch_t5_stack_for_2d_mask(model):
    """Patch T5 model to accept 3D (batch, seq, seq) encoder attention masks.
    Intercepts at the top-level forward to route structured mask to encoder
    while giving the decoder a standard 1D mask."""
    _orig_model_forward = model.forward

    def _patched_model_forward(input_ids=None, attention_mask=None, **kwargs):
        if attention_mask is not None and attention_mask.dim() == 3:
            model.encoder._structured_mask_override = (
                (1.0 - attention_mask[:, None, :, :].to(dtype=model.encoder.embed_tokens.weight.dtype)) * -1e4
            )
            pad_mask = attention_mask[:, 0, :]
            result = _orig_model_forward(input_ids=input_ids, attention_mask=pad_mask, **kwargs)
            model.encoder._structured_mask_override = None
            return result
        return _orig_model_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    model.forward = _patched_model_forward

    # patch each encoder attention layer to use the override mask
    encoder = model.encoder
    for block in encoder.block:
        attn = block.layer[0].SelfAttention
        _orig_attn = attn.forward

        def _make_patched_attn(orig_fn, enc_ref):
            def _patched_attn(*args, **kw):
                override = getattr(enc_ref, '_structured_mask_override', None)
                if override is not None:
                    if 'mask' in kw:
                        kw['mask'] = override
                    elif len(args) >= 2:
                        args = (args[0], override) + args[2:]
                return orig_fn(*args, **kw)
            return _patched_attn

        attn.forward = _make_patched_attn(_orig_attn, encoder)


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
        penalty_alpha: float = 0.0,
        top_k: int = 0,
        num_beam_groups: int = 1,
        diversity_penalty: float = 0.0,
        constrained_decoding: bool = False,
        structured_attention: bool = False,
        focal_gamma: float = 0.0,
        optimizer: str = "adamw",
        language: str = "en",
        train_examples: list[dict] = None,
        val_examples: list[dict] = None,
        test_examples: list[dict] = None,
        eval_scopes: list[dict] = None,
        test_scopes: list[dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_examples", "val_examples"])

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
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
        self._category_set: list[str] | None = None

        if self.hparams.structured_attention:
            _patch_t5_stack_for_2d_mask(self.model)

    def set_test_data(self, examples: list[dict], scopes: list[dict], data_path: str = "", category_set: list[str] = None):
        self._test_examples = examples
        self._test_scopes = scopes
        self._current_test_data = data_path
        self._category_set = category_set

    def on_train_start(self):
        self.model.train()
        self.model.config.use_cache = False

    def on_test_start(self):
        if not self._is_fsdp():
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True

    def on_validation_start(self):
        if not self._is_fsdp():
            self.model.gradient_checkpointing_disable()
        self.model.config.use_cache = True

    def on_validation_end(self):
        if not self._is_fsdp():
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False

    def _is_fsdp(self):
        return hasattr(self, "trainer") and self.trainer is not None and \
               type(self.trainer.strategy).__name__ == "FSDPStrategy"

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
                    category_set=cfg.get("category_set"),
                    bare_prompt=cfg.get("bare_prompt", False),
                    language=cfg.get("language", "en"),
                    include_categories=cfg.get("include_categories", False),
                ).values()
                for ex in part
            ]

            # mix in auxiliary syntax prediction examples
            aux_frac = cfg.get("syntax_auxiliary_fraction", 0.0)
            if aux_frac > 0 and cfg.get("syntax_enrichment"):
                import random as _rnd
                from src.data.data import to_syntax_auxiliary, enrich_syntax_auxiliary
                aux_canonical = cfg.get("_aux_canonical")
                if aux_canonical is None:
                    aux_canonical = enrich_syntax_auxiliary(list(cfg["canonical"]))
                    cfg["_aux_canonical"] = aux_canonical
                aux_tasks = cfg.get("syntax_auxiliary_tasks", ["dep"])
                n_aux = int(len(examples) * aux_frac)
                rng = _rnd.Random(seed)
                for _ in range(n_aux):
                    src = rng.choice(aux_canonical)
                    task = rng.choice(aux_tasks)
                    aux_ex = to_syntax_auxiliary(src, task=task)
                    if aux_ex:
                        examples.append(aux_ex)
        else:
            examples = self._train_examples
        ds = ABSADataset(examples, self.tokenizer, self.hparams.max_length,
                         structured_attention=self.hparams.structured_attention)
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, persistent_workers=True)

    def val_dataloader(self):
        ds = ABSADataset(self._val_examples, self.tokenizer, self.hparams.max_length,
                         structured_attention=self.hparams.structured_attention)
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
        logits = out.logits
        labels = batch["labels"]

        if self.hparams.focal_gamma > 0:
            # focal loss: (1 - p_t)^gamma * CE
            ce = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1),
                ignore_index=-100, reduction="none",
            )
            p_t = torch.exp(-ce)  # probability of correct token
            focal_weight = (1 - p_t) ** self.hparams.focal_gamma
            loss = (focal_weight * ce).mean()
        elif self.hparams.label_smoothing > 0:
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

        # for generate, use 1D padding mask (generate rejects 3D)
        gen_mask = batch["attention_mask"]
        if gen_mask.dim() == 3:
            gen_mask = gen_mask[:, 0, :]

        output_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=gen_mask,
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
            self._val_preds.append(parse_output(decoded, keys, fmt, language=self.hparams.language))
            self._val_golds.append(parse_output(gold, keys, fmt, language=self.hparams.language))

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
            metrics = evaluate(all_preds, all_golds, self._eval_scopes,
                               implicit_mode=getattr(self, "_implicit_mode", None))
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

    def _build_gen_kwargs(self, batch):
        kwargs = dict(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
        )
        # contrastive search: penalty_alpha > 0 and top_k > 0
        # NOTE: incompatible with T5 in transformers >= 4.50 (community impl bug)
        if self.hparams.penalty_alpha > 0 and self.hparams.top_k > 0:
            raise ValueError("Contrastive search is incompatible with T5 in transformers >= 4.50. "
                             "Use beam search or diverse beam search instead.")
        elif self.hparams.do_sample:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.hparams.temperature
            kwargs["num_beams"] = self.hparams.num_beams
        else:
            kwargs["num_beams"] = self.hparams.num_beams
            # diverse beam search
            if self.hparams.num_beam_groups > 1:
                kwargs["num_beam_groups"] = self.hparams.num_beam_groups
                kwargs["diversity_penalty"] = self.hparams.diversity_penalty

        # constrained decoding for NL templates
        if self.hparams.constrained_decoding:
            keys = batch["keys"][0].split(",")
            fmt = batch["output_format"][0]
            if fmt == "natural-language":
                from src.model.constrained import build_logits_processor
                cat_set = getattr(self, "_category_set", None)
                processors = build_logits_processor(self.tokenizer, keys, category_set=cat_set)
                if processors:
                    kwargs["logits_processor"] = processors

        return kwargs

    def test_step(self, batch, batch_idx):
        n_seq = self.hparams.num_return_sequences
        gen_kwargs = self._build_gen_kwargs(batch)

        if n_seq <= 1:
            output_ids = self.model.generate(**gen_kwargs)
            for i, out in enumerate(output_ids):
                decoded = self.tokenizer.decode(out, skip_special_tokens=True)
                gold    = batch["raw_target"][i]
                keys    = batch["keys"][i].split(",")
                fmt     = batch["output_format"][i]
                self._test_preds.append(parse_output(decoded, keys, fmt, language=self.hparams.language))
                self._test_golds.append(parse_output(gold, keys, fmt, language=self.hparams.language))
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
                    for triplet in parse_output(decoded, keys, fmt, language=self.hparams.language):
                        counts[frozenset(triplet.items())] += 1
                voted = [dict(k) for k, c in counts.items() if c >= self.hparams.vote_threshold]
                self._test_preds.append(voted)
                self._test_golds.append(parse_output(gold, keys, fmt, language=self.hparams.language))

    def on_test_epoch_end(self):
        if not self._test_preds:
            return
        if self.global_rank == 0:
            implicit_mode = getattr(self, "_implicit_mode", None)

            if implicit_mode == "full":
                # Run all three modes: exact match, collapse (implicit flag only), resolve (term only)
                metrics = evaluate(self._test_preds, self._test_golds, self._test_scopes, implicit_mode=None)
                metrics["_collapse"] = evaluate(self._test_preds, self._test_golds, self._test_scopes, implicit_mode="collapse")
                metrics["_resolve"] = evaluate(self._test_preds, self._test_golds, self._test_scopes, implicit_mode="resolve")
            else:
                metrics = evaluate(self._test_preds, self._test_golds, self._test_scopes, implicit_mode=implicit_mode)

            entry = {"data": getattr(self, "_current_test_data", ""), **metrics}

            if getattr(self, "_eval_implicit_split", False):
                explicit_preds, explicit_golds = [], []
                implicit_preds, implicit_golds = [], []
                for pred_triplets, gold_triplets in zip(self._test_preds, self._test_golds):
                    ep = [t for t in pred_triplets if not t.get("aspect", "").startswith("IMPLICIT")]
                    ip = [t for t in pred_triplets if t.get("aspect", "").startswith("IMPLICIT")]
                    eg = [t for t in gold_triplets if not t.get("aspect", "").startswith("IMPLICIT")]
                    ig = [t for t in gold_triplets if t.get("aspect", "").startswith("IMPLICIT")]
                    explicit_preds.append(ep)
                    explicit_golds.append(eg)
                    implicit_preds.append(ip)
                    implicit_golds.append(ig)
                if any(g for g in explicit_golds):
                    explicit_metrics = evaluate(explicit_preds, explicit_golds, self._test_scopes,
                                                implicit_mode=implicit_mode)
                    entry["explicit"] = explicit_metrics
                if any(g for g in implicit_golds):
                    implicit_metrics = evaluate(implicit_preds, implicit_golds, self._test_scopes,
                                                implicit_mode=implicit_mode)
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
        if self.hparams.optimizer == "adamw8bit":
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(params, lr=self.hparams.learning_rate)
        else:
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
