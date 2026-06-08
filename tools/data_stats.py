"""
Dataset statistics and analysis tool.

Usage:
    python tools/data_stats.py <file1> [file2 ...] [--spacy-model en_core_web_sm]
    python tools/data_stats.py downloads/emag_ro/*.csv --tokenizer readerbench/RoBERT-base --no-pos
    python tools/data_stats.py downloads/aste/rest14/train.txt downloads/aste/laptop14/test.txt --latex

Outputs:
    - Basic counts (sentences, annotations, avg annotations/sentence)
    - Sentence length stats (min, max, mean, median, percentiles) at word and token level
    - Unique word count (vocabulary size)
    - Aspect/sentiment span length distributions
    - POS tag distributions for aspect and sentiment spans
    - Polarity distribution
    - Category distribution (if present)
    - Overlap analysis (when multiple files given)
    - Implicit vs explicit breakdown
    - JSON output for later use in figures
"""

import argparse
import json
import sys
import statistics
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.data import load_files


def _get_spans(examples: list[dict], key: str) -> list[str]:
    """Extract all non-null spans for a given key."""
    spans = []
    for ex in examples:
        for ann in ex["annotations"]:
            val = ann.get(key)
            if val and val not in ("NULL", "IMPLICIT", "NONE"):
                spans.append(val)
    return spans


def _pos_distribution(spans: list[str], nlp) -> Counter:
    """Get POS tag distribution for a list of text spans."""
    pos_counts = Counter()
    texts = list(set(spans))
    for doc in nlp.pipe(texts, batch_size=256):
        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_counts[token.pos_] += 1
    return pos_counts


def _head_pos(spans: list[str], nlp) -> Counter:
    """Get POS of the syntactic head of each span."""
    head_pos = Counter()
    texts = list(set(spans))
    for doc in nlp.pipe(texts, batch_size=256):
        root = [t for t in doc if t.head == t or t.head not in doc]
        if root:
            head_pos[root[0].pos_] += 1
        elif len(doc) > 0:
            head_pos[doc[0].pos_] += 1
    return head_pos


def _percentile(data: list, p: float) -> float:
    """Compute percentile (0-100)."""
    if not data:
        return 0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_data):
        return sorted_data[-1]
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac


def analyze(examples: list[dict], label: str, nlp=None, tokenizer=None) -> dict:
    """Compute and print statistics for a dataset. Returns stats dict."""
    n_sentences = len(examples)
    n_annotations = sum(len(ex["annotations"]) for ex in examples)
    word_lengths = [len(ex["tokens"]) for ex in examples]
    ann_per_sent = [len(ex["annotations"]) for ex in examples]

    # token-level lengths
    token_lengths = []
    if tokenizer:
        for ex in examples:
            toks = tokenizer(ex["sentence"], add_special_tokens=False)["input_ids"]
            token_lengths.append(len(toks))

    # unique words
    all_words = set()
    for ex in examples:
        for w in ex["tokens"]:
            all_words.add(w.lower())
    vocab_size = len(all_words)

    # polarity distribution
    polarities = Counter()
    for ex in examples:
        for ann in ex["annotations"]:
            pol = ann.get("polarity", "?")
            polarities[pol] += 1

    # category distribution
    categories = Counter()
    for ex in examples:
        for ann in ex["annotations"]:
            cat = ann.get("category")
            if cat:
                categories[cat] += 1

    # implicit counts
    n_implicit_asp = sum(1 for ex in examples for ann in ex["annotations"]
                         if ann.get("aspect") in ("IMPLICIT", "NULL", None))
    n_implicit_opn = sum(1 for ex in examples for ann in ex["annotations"]
                         if ann.get("sentiment") in ("IMPLICIT", "NULL", None))

    # span lengths
    aspects = _get_spans(examples, "aspect")
    sentiments = _get_spans(examples, "sentiment")
    asp_word_lens = [len(a.split()) for a in aspects] if aspects else []
    opn_word_lens = [len(s.split()) for s in sentiments] if sentiments else []

    asp_token_lens = []
    opn_token_lens = []
    if tokenizer and aspects:
        asp_token_lens = [len(tokenizer(a, add_special_tokens=False)["input_ids"]) for a in aspects]
    if tokenizer and sentiments:
        opn_token_lens = [len(tokenizer(s, add_special_tokens=False)["input_ids"]) for s in sentiments]

    # build stats dict
    stats = {
        "label": label,
        "sentences": n_sentences,
        "annotations": n_annotations,
        "avg_annotations_per_sentence": round(n_annotations / max(n_sentences, 1), 2),
        "vocab_size": vocab_size,
        "sentence_length_words": {
            "min": min(word_lengths) if word_lengths else 0,
            "max": max(word_lengths) if word_lengths else 0,
            "mean": round(statistics.mean(word_lengths), 1) if word_lengths else 0,
            "median": round(statistics.median(word_lengths), 0) if word_lengths else 0,
            "p25": round(_percentile(word_lengths, 25), 1),
            "p75": round(_percentile(word_lengths, 75), 1),
            "p95": round(_percentile(word_lengths, 95), 1),
        },
        "polarity_distribution": dict(polarities.most_common()),
        "category_distribution": dict(categories.most_common()) if categories else None,
        "implicit_aspects": n_implicit_asp,
        "implicit_opinions": n_implicit_opn,
    }

    if token_lengths:
        stats["sentence_length_tokens"] = {
            "min": min(token_lengths),
            "max": max(token_lengths),
            "mean": round(statistics.mean(token_lengths), 1),
            "median": round(statistics.median(token_lengths), 0),
            "p25": round(_percentile(token_lengths, 25), 1),
            "p75": round(_percentile(token_lengths, 75), 1),
            "p95": round(_percentile(token_lengths, 95), 1),
        }

    if asp_word_lens:
        stats["aspect_span_words"] = {
            "min": min(asp_word_lens), "max": max(asp_word_lens),
            "mean": round(statistics.mean(asp_word_lens), 2),
            "median": round(statistics.median(asp_word_lens), 0),
        }
    if asp_token_lens:
        stats["aspect_span_tokens"] = {
            "min": min(asp_token_lens), "max": max(asp_token_lens),
            "mean": round(statistics.mean(asp_token_lens), 2),
            "median": round(statistics.median(asp_token_lens), 0),
        }
    if opn_word_lens:
        stats["opinion_span_words"] = {
            "min": min(opn_word_lens), "max": max(opn_word_lens),
            "mean": round(statistics.mean(opn_word_lens), 2),
            "median": round(statistics.median(opn_word_lens), 0),
        }
    if opn_token_lens:
        stats["opinion_span_tokens"] = {
            "min": min(opn_token_lens), "max": max(opn_token_lens),
            "mean": round(statistics.mean(opn_token_lens), 2),
            "median": round(statistics.median(opn_token_lens), 0),
        }

    # annotations per sentence
    stats["annotations_per_sentence"] = {
        "single": sum(1 for a in ann_per_sent if a == 1),
        "multi": sum(1 for a in ann_per_sent if a > 1),
        "max": max(ann_per_sent) if ann_per_sent else 0,
    }

    # print
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Sentences:          {n_sentences}")
    print(f"  Total annotations:  {n_annotations}")
    print(f"  Avg ann/sentence:   {stats['avg_annotations_per_sentence']}")
    print(f"  Vocabulary size:    {vocab_size} unique words")
    print()

    print(f"  Sentence length (words):")
    s = stats["sentence_length_words"]
    print(f"    Min: {s['min']}, Max: {s['max']}, Mean: {s['mean']}, Median: {s['median']}")
    print(f"    P25: {s['p25']}, P75: {s['p75']}, P95: {s['p95']}")

    if "sentence_length_tokens" in stats:
        print(f"  Sentence length (subword tokens):")
        s = stats["sentence_length_tokens"]
        print(f"    Min: {s['min']}, Max: {s['max']}, Mean: {s['mean']}, Median: {s['median']}")
        print(f"    P25: {s['p25']}, P75: {s['p75']}, P95: {s['p95']}")
    print()

    print(f"  Annotations per sentence:")
    a = stats["annotations_per_sentence"]
    print(f"    Single: {a['single']} ({100*a['single']/n_sentences:.0f}%)")
    print(f"    Multi:  {a['multi']} ({100*a['multi']/n_sentences:.0f}%)")
    print(f"    Max:    {a['max']}")
    print()

    print(f"  Polarity distribution:")
    for pol, count in polarities.most_common():
        print(f"    {pol}: {count} ({100*count/n_annotations:.1f}%)")
    print()

    if n_implicit_asp > 0 or n_implicit_opn > 0:
        print(f"  Implicit/NULL:")
        print(f"    Aspects:    {n_implicit_asp} ({100*n_implicit_asp/n_annotations:.1f}%)")
        print(f"    Opinions:   {n_implicit_opn} ({100*n_implicit_opn/n_annotations:.1f}%)")
        print()

    if asp_word_lens:
        print(f"  Aspect span length (words): min={min(asp_word_lens)}, max={max(asp_word_lens)}, mean={statistics.mean(asp_word_lens):.2f}")
        if asp_token_lens:
            print(f"  Aspect span length (tokens): min={min(asp_token_lens)}, max={max(asp_token_lens)}, mean={statistics.mean(asp_token_lens):.2f}")

    if opn_word_lens:
        print(f"  Opinion span length (words): min={min(opn_word_lens)}, max={max(opn_word_lens)}, mean={statistics.mean(opn_word_lens):.2f}")
        if opn_token_lens:
            print(f"  Opinion span length (tokens): min={min(opn_token_lens)}, max={max(opn_token_lens)}, mean={statistics.mean(opn_token_lens):.2f}")
    print()

    if categories:
        print(f"  Category distribution:")
        for cat, count in categories.most_common():
            print(f"    {cat}: {count} ({100*count/n_annotations:.1f}%)")
        print()

    # POS analysis
    if nlp and aspects:
        print(f"  Aspect POS distribution (top 5):")
        pos = _pos_distribution(aspects, nlp)
        total = sum(pos.values())
        for tag, count in pos.most_common(5):
            print(f"    {tag}: {count} ({100*count/total:.1f}%)")

    if nlp and sentiments:
        print(f"  Opinion POS distribution (top 5):")
        pos = _pos_distribution(sentiments, nlp)
        total = sum(pos.values())
        for tag, count in pos.most_common(5):
            print(f"    {tag}: {count} ({100*count/total:.1f}%)")
    print()

    return stats


def overlap_analysis(file_data: dict[str, list[dict]]):
    """Analyze sentence overlap between datasets."""
    names = list(file_data.keys())
    if len(names) < 2:
        return

    print(f"\n{'='*60}")
    print(f"  Overlap Analysis")
    print(f"{'='*60}")

    sentence_sets = {name: set(ex["sentence"] for ex in exs) for name, exs in file_data.items()}

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = sentence_sets[a] & sentence_sets[b]
            total = min(len(sentence_sets[a]), len(sentence_sets[b]))
            print(f"  {a} ∩ {b}: {len(overlap)} sentences ({100*len(overlap)/total:.1f}% of smaller set)")
    print()


def _get_token_vocab(examples: list[dict], key: str) -> set[str]:
    """Get token-level vocabulary (individual words from spans)."""
    tokens = set()
    for ex in examples:
        for ann in ex["annotations"]:
            val = ann.get(key)
            if val and val not in ("NULL", "IMPLICIT", "NONE"):
                for w in val.lower().split():
                    tokens.add(w)
    return tokens


def vocabulary_overlap(file_data: dict[str, list[dict]]):
    """Analyze aspect and opinion vocabulary overlap between datasets."""
    names = list(file_data.keys())
    if len(names) < 2:
        return

    print(f"\n{'='*60}")
    print(f"  Vocabulary Overlap (token-level)")
    print(f"{'='*60}")

    aspect_vocabs = {name: _get_token_vocab(exs, "aspect") for name, exs in file_data.items()}
    opinion_vocabs = {name: _get_token_vocab(exs, "sentiment") for name, exs in file_data.items()}

    short = [n[:12] for n in names]
    w = 8

    # only print if there are actual aspect spans
    if any(len(v) > 0 for v in aspect_vocabs.values()):
        print(f"\n  Aspect token overlap (% of row vocabulary found in column):")
        print(f"  {'':>14}", end="")
        for s in short:
            print(f" {s:>{w}}", end="")
        print()

        for i, name in enumerate(names):
            v = aspect_vocabs[name]
            print(f"  {short[i]:>14}", end="")
            for j, other in enumerate(names):
                if i == j:
                    print(f" {'—':>{w}}", end="")
                else:
                    ov = aspect_vocabs[other]
                    pct = 100 * len(v & ov) / len(v) if v else 0
                    print(f" {pct:>{w}.1f}", end="")
            print(f"  (n={len(v)})")

    if any(len(v) > 0 for v in opinion_vocabs.values()):
        print(f"\n  Opinion token overlap (% of row vocabulary found in column):")
        print(f"  {'':>14}", end="")
        for s in short:
            print(f" {s:>{w}}", end="")
        print()

        for i, name in enumerate(names):
            v = opinion_vocabs[name]
            print(f"  {short[i]:>14}", end="")
            for j, other in enumerate(names):
                if i == j:
                    print(f" {'—':>{w}}", end="")
                else:
                    ov = opinion_vocabs[other]
                    pct = 100 * len(v & ov) / len(v) if v else 0
                    print(f" {pct:>{w}.1f}", end="")
            print(f"  (n={len(v)})")
    print()


def format_latex_table(all_stats: list[dict]) -> str:
    """Generate a LaTeX summary table from stats dicts."""
    lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Dataset statistics summary.}",
        "\\small",
        "\\begin{tabular}{@{}l" + "r" * len(all_stats) + "@{}}",
        "\\toprule",
    ]
    headers = " & ".join(s["label"] for s in all_stats)
    lines.append(f"& {headers} \\\\")
    lines.append("\\midrule")

    def row(name, key_path):
        vals = []
        for s in all_stats:
            v = s
            for k in key_path:
                v = v.get(k, {}) if isinstance(v, dict) else None
                if v is None:
                    break
            vals.append(str(v) if v is not None else "---")
        return f"{name} & {' & '.join(vals)} \\\\"

    lines.append(row("Sentences", ["sentences"]))
    lines.append(row("Annotations", ["annotations"]))
    lines.append(row("Avg ann/sent", ["avg_annotations_per_sentence"]))
    lines.append(row("Vocabulary", ["vocab_size"]))
    lines.append("\\midrule")
    lines.append(row("Sent. len (words) mean", ["sentence_length_words", "mean"]))
    lines.append(row("Sent. len (words) max", ["sentence_length_words", "max"]))
    if any("sentence_length_tokens" in s for s in all_stats):
        lines.append(row("Sent. len (tokens) mean", ["sentence_length_tokens", "mean"]))
        lines.append(row("Sent. len (tokens) max", ["sentence_length_tokens", "max"]))

    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("files", nargs="+", help="Data files to analyze")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model for POS analysis")
    parser.add_argument("--no-pos", action="store_true", help="Skip POS analysis (faster)")
    parser.add_argument("--tokenizer", default=None, help="HuggingFace tokenizer for subword token stats (e.g. google/flan-t5-base)")
    parser.add_argument("--latex", action="store_true", help="Output LaTeX summary table")
    parser.add_argument("--output", default=None, help="Save stats JSON to this file")
    args = parser.parse_args()

    nlp = None
    if not args.no_pos:
        try:
            import spacy
            nlp = spacy.load(args.spacy_model, disable=["ner", "lemmatizer"])
            print(f"Loaded spaCy model: {args.spacy_model}")
        except Exception as e:
            print(f"Warning: could not load spaCy ({e}), skipping POS analysis")

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"Loaded tokenizer: {args.tokenizer}")

    file_data = {}
    all_stats = []
    for f in args.files:
        label = Path(f).stem
        parent = Path(f).parent.name
        if label in ("train", "test", "dev"):
            label = f"{parent}/{label}"
        examples = load_files(f)
        file_data[label] = examples
        stats = analyze(examples, label, nlp, tokenizer)
        all_stats.append(stats)

    if len(args.files) > 1:
        overlap_analysis(file_data)
        vocabulary_overlap(file_data)

    if args.latex:
        print("\n" + "=" * 60)
        print("  LaTeX Table")
        print("=" * 60)
        print(format_latex_table(all_stats))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        print(f"\nStats saved to: {out_path}")


if __name__ == "__main__":
    main()
