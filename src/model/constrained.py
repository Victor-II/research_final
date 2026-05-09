"""
FSM-based constrained decoding for NL template outputs.

Uses decoded text matching to track position in the template, then
constrains the next token for specific slots:
  - polarity: only "positive" / "negative" / "neutral"
  - category: only tokens forming valid category labels (trie-based)
  - aspect/sentiment: unconstrained

Falls back gracefully if the model deviates from the template.
"""
import re
import torch
from transformers import LogitsProcessor
from src.data.data import _NL_TEMPLATES
from constants import CANONICAL_KEY_ORDER

POLARITY_VALUES = ["positive", "negative", "neutral"]


def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _build_trie(token_sequences):
    root = {}
    for seq in token_sequences:
        node = root
        for tok in seq:
            node = node.setdefault(tok, {})
        node[None] = True
    return root


def _trie_walk(trie, prefix):
    """Walk prefix through trie, return (allowed_next_tokens, at_valid_end)."""
    node = trie
    for tok in prefix:
        if tok not in node:
            return set(), False
        node = node[tok]
    return {k for k in node if k is not None}, (None in node)


def _split_template(template: str):
    parts = []
    last = 0
    for m in re.finditer(r"\{(\w+)\}", template):
        if m.start() > last:
            parts.append(("fixed", template[last:m.start()]))
        parts.append(("slot", m.group(1)))
        last = m.end()
    if last < len(template):
        parts.append(("fixed", template[last:]))
    return parts


class NLTemplateFSM(LogitsProcessor):

    def __init__(self, tokenizer, keys: list[str], category_set: list[str] = None):
        canonical_keys = [k for k in CANONICAL_KEY_ORDER if k in keys]
        keys_set = frozenset(canonical_keys)
        template = _NL_TEMPLATES.get(keys_set)
        self.active = template is not None
        if not self.active:
            return

        self.tokenizer = tokenizer
        self.eos_id = tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id
        self.parts = _split_template(template)
        self.delimiters = [p[1] for p in self.parts if p[0] == "fixed"]
        self.slot_keys = [p[1] for p in self.parts if p[0] == "slot"]
        self.sep = " ; "

        self.polarity_trie = _build_trie(
            [_tokenize(tokenizer, v) for v in POLARITY_VALUES]
        )
        self.category_trie = (
            _build_trie([_tokenize(tokenizer, c) for c in category_set])
            if category_set else None
        )

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.active:
            return scores
        for i in range(input_ids.size(0)):
            generated = input_ids[i].tolist()
            if generated and generated[0] == self.pad_id:
                generated = generated[1:]
            slot_key, slot_idx, slot_prefix = self._find_current_slot(generated)
            if slot_key is None:
                continue
            allowed = self._constrain_slot(slot_key, slot_prefix, slot_idx)
            if allowed is not None:
                mask = torch.full_like(scores[i], float("-inf"))
                for tok_id in allowed:
                    mask[tok_id] = 0
                scores[i] = scores[i] + mask
        return scores

    def _find_current_slot(self, generated: list[int]):
        """Returns (slot_key, slot_idx, token_prefix_in_slot) or (None, None, None)."""
        if not generated:
            if self.slot_keys:
                return self.slot_keys[0], 0, []
            return None, None, None

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if self.sep in text:
            text = text.rsplit(self.sep, 1)[1]

        slot_idx = 0
        remaining = text
        for delim_idx, delim in enumerate(self.delimiters):
            pos = remaining.find(delim)
            if pos >= 0:
                remaining = remaining[pos + len(delim):]
                slot_idx = delim_idx + 1
            else:
                for trim in range(1, len(delim)):
                    if remaining.endswith(delim[:trim]):
                        return None, None, None
                break

        if slot_idx >= len(self.slot_keys):
            return None, None, None

        slot_text = remaining
        slot_token_ids = _tokenize(self.tokenizer, slot_text) if slot_text.strip() else []
        return self.slot_keys[slot_idx], slot_idx, slot_token_ids

    def _constrain_slot(self, slot_key: str, prefix: list[int], slot_idx: int) -> set | None:
        if slot_key == "polarity":
            allowed, at_end = _trie_walk(self.polarity_trie, prefix)
            if at_end:
                allowed = allowed | self._transition_tokens(slot_idx)
            return allowed if allowed else None

        if slot_key == "category" and self.category_trie is not None:
            allowed, at_end = _trie_walk(self.category_trie, prefix)
            if at_end:
                allowed = allowed | self._transition_tokens(slot_idx)
            return allowed if allowed else None

        return None

    def _transition_tokens(self, slot_idx) -> set:
        """Tokens that can follow a completed constrained slot."""
        result = {self.eos_id}
        sep_ids = _tokenize(self.tokenizer, self.sep)
        if sep_ids:
            # skip leading SentencePiece space marker (token id 3 for T5)
            for tid in sep_ids:
                if tid != 3:
                    result.add(tid)
                    break
        # next delimiter's first meaningful token
        delim_idx = slot_idx
        if delim_idx < len(self.delimiters):
            delim_ids = _tokenize(self.tokenizer, self.delimiters[delim_idx])
            for tid in delim_ids:
                if tid != 3:
                    result.add(tid)
                    break
        return result


def build_logits_processor(tokenizer, keys: list[str], category_set: list[str] = None):
    processor = NLTemplateFSM(tokenizer, keys, category_set)
    if processor.active:
        return [processor]
    return None
