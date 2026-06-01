"""
Dataset statistics and analysis tool.

Usage:
    python tools/data_stats.py <file1> [file2 ...] [--spacy-model en_core_web_sm]

Outputs:
    - Basic counts (sentences, annotations, avg annotations/sentence)
    - Sentence length stats (min, max, mean, median)
    - Aspect/sentiment span length distributions
    - POS tag distributions for aspect and sentiment spans
    - Polarity distribution
    - Overlap analysis (when multiple files given)
    - Implicit vs explicit breakdown
"""

import argparse
import sys
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
    # batch process for speed
    texts = list(set(spans))
    for doc in nlp.pipe(texts, batch_size=256):
        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_counts[token.pos_] += 1
    return pos_counts


def _head_pos(spans: list[str], nlp) -> Counter:
    """Get POS of the syntactic head of each span (approximation: root of span)."""
    head_pos = Counter()
    texts = list(set(spans))
    for doc in nlp.pipe(texts, batch_size=256):
        # use the root token of the span
        root = [t for t in doc if t.head == t or t.head not in doc]
        if root:
            head_pos[root[0].pos_] += 1
        elif len(doc) > 0:
            head_pos[doc[0].pos_] += 1
    return head_pos


def analyze(examples: list[dict], label: str, nlp=None):
    """Print statistics for a dataset."""
    n_sentences = len(examples)
    n_annotations = sum(len(ex["annotations"]) for ex in examples)
    sent_lengths = [len(ex["tokens"]) for ex in examples]
    ann_per_sent = [len(ex["annotations"]) for ex in examples]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Sentences:          {n_sentences}")
    print(f"  Total annotations:  {n_annotations}")
    print(f"  Avg ann/sentence:   {n_annotations/max(n_sentences,1):.2f}")
    print()

    # sentence lengths
    if sent_lengths:
        import statistics
        print(f"  Sentence length (tokens):")
        print(f"    Min: {min(sent_lengths)}, Max: {max(sent_lengths)}, "
              f"Mean: {statistics.mean(sent_lengths):.1f}, Median: {statistics.median(sent_lengths):.0f}")
    print()

    # annotations per sentence
    if ann_per_sent:
        import statistics
        multi = sum(1 for a in ann_per_sent if a > 1)
        print(f"  Annotations per sentence:")
        print(f"    Single: {sum(1 for a in ann_per_sent if a == 1)} ({100*sum(1 for a in ann_per_sent if a == 1)/n_sentences:.0f}%)")
        print(f"    Multi:  {multi} ({100*multi/n_sentences:.0f}%)")
        print(f"    Max:    {max(ann_per_sent)}")
    print()

    # polarity distribution
    polarities = Counter()
    for ex in examples:
        for ann in ex["annotations"]:
            pol = ann.get("polarity", "?")
            polarities[pol] += 1
    if polarities:
        print(f"  Polarity distribution:")
        for pol, count in polarities.most_common():
            print(f"    {pol}: {count} ({100*count/n_annotations:.1f}%)")
    print()

    # implicit vs explicit
    n_implicit_asp = sum(1 for ex in examples for ann in ex["annotations"]
                         if ann.get("aspect") in ("IMPLICIT", "NULL", None))
    n_implicit_opn = sum(1 for ex in examples for ann in ex["annotations"]
                         if ann.get("sentiment") in ("IMPLICIT", "NULL", None))
    if n_implicit_asp > 0 or n_implicit_opn > 0:
        print(f"  Implicit/NULL:")
        print(f"    Aspects:    {n_implicit_asp} ({100*n_implicit_asp/n_annotations:.1f}%)")
        print(f"    Sentiments: {n_implicit_opn} ({100*n_implicit_opn/n_annotations:.1f}%)")
        print()

    # span lengths
    aspects = _get_spans(examples, "aspect")
    sentiments = _get_spans(examples, "sentiment")

    if aspects:
        asp_lens = [len(a.split()) for a in aspects]
        import statistics
        print(f"  Aspect span length (words):")
        print(f"    Min: {min(asp_lens)}, Max: {max(asp_lens)}, "
              f"Mean: {statistics.mean(asp_lens):.2f}, Median: {statistics.median(asp_lens):.0f}")
        len_dist = Counter(asp_lens)
        top5 = len_dist.most_common(5)
        print(f"    Distribution: {', '.join(f'{l}w={c}' for l,c in sorted(top5))}")

    if sentiments:
        opn_lens = [len(s.split()) for s in sentiments]
        import statistics
        print(f"  Sentiment span length (words):")
        print(f"    Min: {min(opn_lens)}, Max: {max(opn_lens)}, "
              f"Mean: {statistics.mean(opn_lens):.2f}, Median: {statistics.median(opn_lens):.0f}")
        len_dist = Counter(opn_lens)
        top5 = len_dist.most_common(5)
        print(f"    Distribution: {', '.join(f'{l}w={c}' for l,c in sorted(top5))}")
    print()

    # POS analysis
    if nlp and aspects:
        print(f"  Aspect POS distribution (top 5):")
        pos = _pos_distribution(aspects, nlp)
        total = sum(pos.values())
        for tag, count in pos.most_common(5):
            print(f"    {tag}: {count} ({100*count/total:.1f}%)")

    if nlp and sentiments:
        print(f"  Sentiment POS distribution (top 5):")
        pos = _pos_distribution(sentiments, nlp)
        total = sum(pos.values())
        for tag, count in pos.most_common(5):
            print(f"    {tag}: {count} ({100*count/total:.1f}%)")
    print()

    # category distribution (if present)
    categories = Counter()
    for ex in examples:
        for ann in ex["annotations"]:
            cat = ann.get("category")
            if cat:
                categories[cat] += 1
    if categories:
        print(f"  Category distribution (top 10):")
        for cat, count in categories.most_common(10):
            print(f"    {cat}: {count} ({100*count/n_annotations:.1f}%)")
        print()


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


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics")
    parser.add_argument("files", nargs="+", help="Data files to analyze")
    parser.add_argument("--spacy-model", default="en_core_web_sm", help="spaCy model for POS analysis")
    parser.add_argument("--no-pos", action="store_true", help="Skip POS analysis (faster)")
    args = parser.parse_args()

    nlp = None
    if not args.no_pos:
        try:
            import spacy
            nlp = spacy.load(args.spacy_model, disable=["ner", "lemmatizer"])
            print(f"Loaded spaCy model: {args.spacy_model}")
        except Exception as e:
            print(f"Warning: could not load spaCy ({e}), skipping POS analysis")

    file_data = {}
    for f in args.files:
        label = Path(f).stem
        # make label more readable
        parent = Path(f).parent.name
        if label in ("train", "test", "dev"):
            label = f"{parent}/{label}"
        examples = load_files(f)
        file_data[label] = examples
        analyze(examples, label, nlp)

    if len(args.files) > 1:
        overlap_analysis(file_data)


if __name__ == "__main__":
    main()
