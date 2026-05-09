"""
Build 2D structured attention masks from dep-compact syntax annotations.

Mask design:
  - Non-syntax tokens (Task/Input/Output lines): full mutual attention
  - Sentence word tokens: attend to all non-syntax tokens + their own syntax entry
  - Syntax entry tokens: attend to (a) their own sentence word, (b) dependency-linked
    syntax entries, (c) themselves
  - This creates an information bottleneck: syntax info flows to the sentence
    representation only through the specific word it annotates.
"""
import re
import torch
from transformers import PreTrainedTokenizer


def build_syntax_attention_mask(
    input_text: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> torch.Tensor | None:
    """Build a (max_length, max_length) binary attention mask.
    Returns None if no Syntax line found. 1.0 = attend, 0.0 = mask."""
    if "\nSyntax:" not in input_text:
        return None

    parts = input_text.split("\nSyntax: ", 1)
    if len(parts) != 2:
        return None

    pre_syntax = parts[0]
    syntax_and_rest = parts[1]
    syntax_text = syntax_and_rest.split("\n", 1)[0] if "\n" in syntax_and_rest else syntax_and_rest

    # tokenize full input to get total length
    full_ids = tokenizer.encode(input_text, add_special_tokens=True, max_length=max_length, truncation=True)
    seq_len = min(len(full_ids), max_length)

    # find token boundaries by tokenizing prefixes
    pre_syntax_with_label = pre_syntax + "\nSyntax: "
    pre_ids = tokenizer.encode(pre_syntax_with_label, add_special_tokens=True)
    syntax_token_start = len(pre_ids) - 1  # exclude trailing EOS

    if syntax_token_start >= seq_len:
        return None  # syntax got truncated

    # parse dep entries: word->head:dep
    dep_entries = syntax_text.strip().split()
    deps = []
    for entry in dep_entries:
        m = re.match(r"^(.+?)->(.+?):(.+)$", entry)
        if m:
            deps.append({"word": m.group(1), "head": m.group(2), "dep": m.group(3), "raw": entry})

    # map each dep entry to its token range in the full sequence
    # tokenize entries sequentially from syntax_token_start
    entry_ranges = {}  # word -> (start, end) token indices
    pos = syntax_token_start
    for i, d in enumerate(deps):
        raw = d["raw"]
        if i > 0:
            # space between entries
            sp_ids = tokenizer.encode(" " + raw, add_special_tokens=False)
            entry_ids = sp_ids
        else:
            entry_ids = tokenizer.encode(raw, add_special_tokens=False)
        end = pos + len(entry_ids)
        if end > seq_len:
            break
        entry_ranges[d["word"]] = (pos, end)
        pos = end

    # map sentence words to token ranges
    input_match = re.search(r"Input: (.+?)(?:\n|$)", pre_syntax)
    if not input_match:
        return None
    sentence = input_match.group(1)
    before_sentence = pre_syntax[:input_match.end()]
    before_ids = tokenizer.encode(before_sentence, add_special_tokens=True)
    sent_start = len(before_ids) - 1

    sentence_words = sentence.split()
    word_ranges = {}  # word -> (start, end) token indices
    wpos = sent_start
    for w in sentence_words:
        w_ids = tokenizer.encode(w, add_special_tokens=False)
        wend = wpos + len(w_ids)
        if wend > seq_len:
            wend = seq_len
        if w not in word_ranges:  # first occurrence
            word_ranges[w] = (wpos, wend)
        wpos = wend

    # build dependency adjacency: which words are linked
    dep_links = set()
    for d in deps:
        w, h = d["word"], d["head"]
        if w in entry_ranges and h in entry_ranges:
            dep_links.add((w, h))
            dep_links.add((h, w))

    # build mask
    mask = torch.zeros(max_length, max_length)

    # 0. every token attends to itself (prevents NaN in softmax)
    for i in range(seq_len):
        mask[i, i] = 1.0

    # 1. non-syntax tokens: full mutual attention
    ns_end = min(syntax_token_start, seq_len)
    mask[:ns_end, :ns_end] = 1.0

    # 1b. tokens after syntax section (e.g. "Output: natural language")
    if entry_ranges:
        after_syntax = max(e for _, (_, e) in entry_ranges.items())
        if after_syntax < seq_len:
            # these tokens get full attention with all non-syntax tokens
            mask[after_syntax:seq_len, after_syntax:seq_len] = 1.0
            mask[:ns_end, after_syntax:seq_len] = 1.0
            mask[after_syntax:seq_len, :ns_end] = 1.0

    # 2. syntax entry self-attention (each entry attends to itself)
    for w, (s, e) in entry_ranges.items():
        mask[s:e, s:e] = 1.0

    # 3. syntax <-> corresponding sentence word (bidirectional)
    for w in entry_ranges:
        if w in word_ranges:
            ss, se = entry_ranges[w]
            ws, we = word_ranges[w]
            mask[ss:se, ws:we] = 1.0  # syntax -> sentence word
            mask[ws:we, ss:se] = 1.0  # sentence word -> syntax

    # 4. dependency-linked syntax entries attend to each other
    for w1, w2 in dep_links:
        if w1 in entry_ranges and w2 in entry_ranges:
            s1, e1 = entry_ranges[w1]
            s2, e2 = entry_ranges[w2]
            mask[s1:e1, s2:e2] = 1.0

    return mask
