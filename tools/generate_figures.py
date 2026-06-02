"""Generate dissertation figures from data analysis."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.data.data import load_aste_file

OUT_DIR = Path("dissertation/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})


def load_all_datasets():
    datasets = {
        "Rest14 train": "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/train.txt",
        "Rest14 test": "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/test.txt",
        "Rest15 test": "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/403.Restaurant15/test.txt",
        "Rest16 test": "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/404.Restaurant16/test.txt",
        "Laptop14 test": "downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/401.Laptop14/test.txt",
        "Electronics": "downloads/DMASTE/dataset/electronics/train.txt",
        "Beauty": "downloads/DMASTE/dataset/beauty/train.txt",
        "Fashion": "downloads/DMASTE/dataset/fashion/train.txt",
        "Home": "downloads/DMASTE/dataset/home/train.txt",
        "Book": "downloads/DMASTE/dataset/book/test.txt",
        "Grocery": "downloads/DMASTE/dataset/grocery/test.txt",
        "Pet": "downloads/DMASTE/dataset/pet/test.txt",
        "Toy": "downloads/DMASTE/dataset/toy/test.txt",
    }
    return {name: load_aste_file(path) for name, path in datasets.items()}


def get_token_vocab(examples, key):
    tokens = set()
    for ex in examples:
        for ann in ex["annotations"]:
            val = ann.get(key)
            if val and val not in ("IMPLICIT", "NULL", "NONE", None):
                for w in val.lower().split():
                    tokens.add(w)
    return tokens


def get_pos_distribution(examples, key):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    from collections import Counter

    spans = set()
    for ex in examples:
        for ann in ex["annotations"]:
            val = ann.get(key)
            if val and val not in ("IMPLICIT", "NULL", "NONE", None):
                spans.add(val)

    pos_counts = Counter()
    for doc in nlp.pipe(list(spans), batch_size=256):
        for token in doc:
            if not token.is_punct and not token.is_space:
                pos_counts[token.pos_] += 1
    return pos_counts


# ============================================================
# Figure 1: Opinion POS distribution comparison
# ============================================================
def fig_opinion_pos(all_data):
    print("Generating: pos_comparison.pdf")

    # Per-dataset POS distributions
    dataset_names = [
        "Rest14 train", "Rest14 test", "Rest15 test", "Rest16 test", "Laptop14 test",
        "Electronics", "Beauty", "Fashion", "Home",
        "Book", "Grocery", "Pet", "Toy",
    ]

    short_labels = [
        "Rest14\ntrain", "Rest14\ntest", "Rest15\ntest", "Rest16\ntest", "Laptop14\ntest",
        "Electronics\ntrain", "Beauty\ntrain", "Fashion\ntrain", "Home\ntrain",
        "Book\ntest", "Grocery\ntest", "Pet\ntest", "Toy\ntest",
    ]

    opinion_tags = ["ADJ", "VERB", "NOUN", "ADV", "OTHER"]
    aspect_tags = ["NOUN", "PROPN", "VERB", "ADJ", "OTHER"]

    # Compute opinion POS
    opinion_dists = {}
    for name in dataset_names:
        dist = get_pos_distribution(all_data[name], "sentiment")
        grand_total = sum(dist.values())
        if grand_total == 0:
            opinion_dists[name] = {t: 0 for t in opinion_tags}
            continue
        result = {}
        for tag in ["ADJ", "VERB", "NOUN", "ADV"]:
            result[tag] = dist.get(tag, 0) / grand_total * 100
        result["OTHER"] = (grand_total - sum(dist.get(t, 0) for t in ["ADJ", "VERB", "NOUN", "ADV"])) / grand_total * 100
        opinion_dists[name] = result

    # Compute aspect POS
    aspect_dists = {}
    for name in dataset_names:
        dist = get_pos_distribution(all_data[name], "aspect")
        grand_total = sum(dist.values())
        if grand_total == 0:
            aspect_dists[name] = {t: 0 for t in aspect_tags}
            continue
        result = {}
        for tag in ["NOUN", "PROPN", "VERB", "ADJ"]:
            result[tag] = dist.get(tag, 0) / grand_total * 100
        result["OTHER"] = (grand_total - sum(dist.get(t, 0) for t in ["NOUN", "PROPN", "VERB", "ADJ"])) / grand_total * 100
        aspect_dists[name] = result

    x = np.arange(len(dataset_names))
    width = 0.16

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    # Top: Aspect POS
    ax = axes[0]
    pos_colors = {"NOUN": "#4C72B0", "PROPN": "#DD8452", "VERB": "#55A868", "ADJ": "#C44E52", "ADV": "#8172B2", "OTHER": "#8C8C8C"}
    for i, tag in enumerate(aspect_tags):
        values = [aspect_dists[name][tag] for name in dataset_names]
        ax.bar(x + (i - 2) * width, values, width, label=tag, color=pos_colors[tag])

    ax.set_ylabel("Percentage (%)")
    ax.set_title("POS Distribution of Aspect Spans")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.legend(loc="upper right", ncol=5, framealpha=0.9)
    ax.set_ylim(0, 70)
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(x=4.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=8.5, color="gray", linestyle="--", alpha=0.5)

    # Bottom: Opinion POS
    ax = axes[1]
    for i, tag in enumerate(opinion_tags):
        values = [opinion_dists[name][tag] for name in dataset_names]
        ax.bar(x + (i - 2) * width, values, width, label=tag, color=pos_colors[tag])

    ax.set_ylabel("Percentage (%)")
    ax.set_title("POS Distribution of Opinion Spans")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8)
    ax.legend(loc="upper right", ncol=5, framealpha=0.9)
    ax.set_ylim(0, 45)
    ax.grid(axis="y", alpha=0.3)
    ax.axvline(x=4.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=8.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "pos_comparison.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 2: Vocabulary overlap (% test seen in train)
# ============================================================
def fig_vocab_overlap(all_data):
    print("Generating: vocab_overlap.pdf")

    # SemEval: Rest14 train -> various test sets
    rest14_train_asp = get_token_vocab(all_data["Rest14 train"], "aspect")
    rest14_train_opn = get_token_vocab(all_data["Rest14 train"], "sentiment")

    # DMASTE: all source -> target
    dmaste_source = all_data["Electronics"] + all_data["Beauty"] + all_data["Fashion"] + all_data["Home"]
    dmaste_train_asp = get_token_vocab(dmaste_source, "aspect")
    dmaste_train_opn = get_token_vocab(dmaste_source, "sentiment")

    test_sets = [
        ("Rest14\n(ID)", all_data["Rest14 test"], rest14_train_asp, rest14_train_opn),
        ("Rest15\n(near)", all_data["Rest15 test"], rest14_train_asp, rest14_train_opn),
        ("Rest16\n(near)", all_data["Rest16 test"], rest14_train_asp, rest14_train_opn),
        ("Laptop14\n(OOD)", all_data["Laptop14 test"], rest14_train_asp, rest14_train_opn),
        ("Book\n(OOD)", all_data["Book"], dmaste_train_asp, dmaste_train_opn),
        ("Grocery\n(OOD)", all_data["Grocery"], dmaste_train_asp, dmaste_train_opn),
        ("Pet\n(OOD)", all_data["Pet"], dmaste_train_asp, dmaste_train_opn),
        ("Toy\n(OOD)", all_data["Toy"], dmaste_train_asp, dmaste_train_opn),
    ]

    labels = []
    asp_overlaps = []
    opn_overlaps = []

    for label, exs, train_asp, train_opn in test_sets:
        test_asp = get_token_vocab(exs, "aspect")
        test_opn = get_token_vocab(exs, "sentiment")
        asp_pct = 100 * len(test_asp & train_asp) / len(test_asp) if test_asp else 0
        opn_pct = 100 * len(test_opn & train_opn) / len(test_opn) if test_opn else 0
        labels.append(label)
        asp_overlaps.append(asp_pct)
        opn_overlaps.append(opn_pct)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, asp_overlaps, width, label="Aspect tokens", color="#4C72B0")
    ax.bar(x + width/2, opn_overlaps, width, label="Opinion tokens", color="#DD8452")

    ax.set_ylabel("% of test vocabulary seen in training")
    ax.set_title("Vocabulary Overlap Between Training and Test Sets")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 85)
    ax.grid(axis="y", alpha=0.3)

    # add vertical separator between SemEval and DMASTE
    ax.axvline(x=3.5, color="gray", linestyle="--", alpha=0.5)
    ax.text(1.5, 80, "SemEval", ha="center", fontsize=9, color="gray")
    ax.text(5.5, 80, "DMASTE", ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "vocab_overlap.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 3: Vocabulary overlap vs F1 scatter
# ============================================================
def fig_overlap_vs_f1(all_data):
    print("Generating: overlap_vs_f1.pdf")

    # F1 scores from experiments (nl-baseline s42 for SemEval, allsource-nl for DMASTE)
    f1_scores = {
        "Rest14 test": 0.7265,
        "Rest15 test": 0.6073,
        "Rest16 test": 0.6723,
        "Laptop14 test": 0.5430,
        "Book": 0.4127,
        "Grocery": 0.4662,
        "Pet": 0.4322,
        "Toy": 0.5016,
    }

    rest14_train_asp = get_token_vocab(all_data["Rest14 train"], "aspect")
    rest14_train_opn = get_token_vocab(all_data["Rest14 train"], "sentiment")
    dmaste_source = all_data["Electronics"] + all_data["Beauty"] + all_data["Fashion"] + all_data["Home"]
    dmaste_train_asp = get_token_vocab(dmaste_source, "aspect")
    dmaste_train_opn = get_token_vocab(dmaste_source, "sentiment")

    # Compute stats for each test set
    import statistics
    test_stats = {}
    for name, f1 in f1_scores.items():
        exs = all_data[name]
        if name in ("Rest14 test", "Rest15 test", "Rest16 test", "Laptop14 test"):
            train_asp, train_opn = rest14_train_asp, rest14_train_opn
            group = "semeval"
        else:
            train_asp, train_opn = dmaste_train_asp, dmaste_train_opn
            group = "dmaste"

        test_asp = get_token_vocab(exs, "aspect")
        test_opn = get_token_vocab(exs, "sentiment")
        asp_pct = len(test_asp & train_asp) / len(test_asp) * 100 if test_asp else 0
        opn_pct = len(test_opn & train_opn) / len(test_opn) * 100 if test_opn else 0
        avg_overlap = (asp_pct + opn_pct) / 2

        n_ann = sum(len(ex["annotations"]) for ex in exs)
        n_implicit = sum(1 for ex in exs for ann in ex["annotations"] if ann.get("aspect") in ("IMPLICIT", None))
        implicit_pct = 100 * n_implicit / n_ann if n_ann else 0
        avg_sent_len = statistics.mean(len(ex["tokens"]) for ex in exs)
        avg_ann = n_ann / len(exs)

        test_stats[name] = {
            "f1": f1,
            "avg_overlap": avg_overlap,
            "implicit_pct": implicit_pct,
            "avg_sent_len": avg_sent_len,
            "avg_ann": avg_ann,
            "group": group,
        }

    # Create 2x2 scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    dimensions = [
        ("avg_overlap", "Avg. Vocabulary Overlap (%)", axes[0, 0]),
        ("implicit_pct", "Implicit Aspects (%)", axes[0, 1]),
        ("avg_sent_len", "Avg. Sentence Length (tokens)", axes[1, 0]),
        ("avg_ann", "Avg. Triplets per Sentence", axes[1, 1]),
    ]

    for dim_key, dim_label, ax in dimensions:
        for name, s in test_stats.items():
            color = "#4C72B0" if s["group"] == "semeval" else "#DD8452"
            marker = "o" if s["group"] == "semeval" else "^"
            ax.scatter(s[dim_key], s["f1"], color=color, s=60, marker=marker, zorder=5)
            short_name = name.replace(" test", "").replace("Rest", "R").replace("Laptop", "Lap")
            ax.annotate(short_name, (s[dim_key], s["f1"]), textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_xlabel(dim_label)
        ax.set_ylabel("Triplet F1")
        ax.grid(alpha=0.3)

    # Add legend to first subplot
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4C72B0', markersize=8, label='SemEval'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#DD8452', markersize=8, label='DMASTE'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="lower right")

    plt.suptitle("Dataset Characteristics vs. Model Performance", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "overlap_vs_f1.pdf", bbox_inches="tight")
    plt.close()


# ============================================================
# Figure 4: Dataset characteristics comparison
# ============================================================
def fig_dataset_characteristics(all_data):
    print("Generating: dataset_characteristics.pdf")

    dataset_order = [
        "Rest14 train", "Rest14 test", "Rest15 test", "Rest16 test", "Laptop14 test",
        "Electronics", "Beauty", "Fashion", "Home",
        "Book", "Grocery", "Pet", "Toy",
    ]
    short_labels = [
        "Rest14\ntrain", "Rest14\ntest", "Rest15\ntest", "Rest16\ntest", "Laptop14\ntest",
        "Electronics\ntrain", "Beauty\ntrain", "Fashion\ntrain", "Home\ntrain",
        "Book\ntest", "Grocery\ntest", "Pet\ntest", "Toy\ntest",
    ]

    import statistics
    stats = []
    for name in dataset_order:
        exs = all_data[name]
        n_ann = sum(len(ex["annotations"]) for ex in exs)
        n_implicit = sum(1 for ex in exs for ann in ex["annotations"] if ann.get("aspect") in ("IMPLICIT", None))

        asp_lens = []
        opn_lens = []
        for ex in exs:
            for ann in ex["annotations"]:
                a = ann.get("aspect")
                if a and a not in ("IMPLICIT", "NULL", "NONE"):
                    asp_lens.append(len(a.split()))
                s = ann.get("sentiment")
                if s and s not in ("IMPLICIT", "NULL", "NONE"):
                    opn_lens.append(len(s.split()))

        stats.append({
            "name": name,
            "avg_len": statistics.mean(len(ex["tokens"]) for ex in exs),
            "avg_ann": n_ann / len(exs),
            "implicit_pct": 100 * n_implicit / n_ann if n_ann else 0,
            "avg_asp_len": statistics.mean(asp_lens) if asp_lens else 0,
            "avg_opn_len": statistics.mean(opn_lens) if opn_lens else 0,
        })

    x = np.arange(len(dataset_order))
    colors = ["#4C72B0"] * 5 + ["#DD8452"] * 4 + ["#55A868"] * 4

    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

    # Avg sentence length
    axes[0].bar(x, [s["avg_len"] for s in stats], color=colors)
    axes[0].set_ylabel("Tokens")
    axes[0].set_title("Average Sentence Length")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axvline(x=4.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].axvline(x=8.5, color="gray", linestyle="--", alpha=0.5)

    # Avg annotations per sentence
    axes[1].bar(x, [s["avg_ann"] for s in stats], color=colors)
    axes[1].set_ylabel("Triplets")
    axes[1].set_title("Average Triplets per Sentence")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].axvline(x=4.5, color="gray", linestyle="--", alpha=0.5)
    axes[1].axvline(x=8.5, color="gray", linestyle="--", alpha=0.5)

    # Implicit aspect %
    axes[2].bar(x, [s["implicit_pct"] for s in stats], color=colors)
    axes[2].set_ylabel("%")
    axes[2].set_title("Implicit Aspects (%)")
    axes[2].grid(axis="y", alpha=0.3)
    axes[2].axvline(x=4.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].axvline(x=8.5, color="gray", linestyle="--", alpha=0.5)

    # Avg aspect span length
    axes[3].bar(x, [s["avg_asp_len"] for s in stats], color=colors)
    axes[3].set_ylabel("Words")
    axes[3].set_title("Average Aspect Span Length")
    axes[3].grid(axis="y", alpha=0.3)
    axes[3].axvline(x=4.5, color="gray", linestyle="--", alpha=0.5)
    axes[3].axvline(x=8.5, color="gray", linestyle="--", alpha=0.5)

    # Avg opinion span length
    axes[4].bar(x, [s["avg_opn_len"] for s in stats], color=colors)
    axes[4].set_ylabel("Words")
    axes[4].set_title("Average Opinion Span Length")
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(short_labels, fontsize=8)
    axes[4].grid(axis="y", alpha=0.3)
    axes[4].axvline(x=4.5, color="gray", linestyle="--", alpha=0.5)
    axes[4].axvline(x=8.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "dataset_characteristics.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("Loading datasets...")
    all_data = load_all_datasets()

    fig_opinion_pos(all_data)
    fig_vocab_overlap(all_data)
    fig_overlap_vs_f1(all_data)
    fig_dataset_characteristics(all_data)

    print(f"\nAll figures saved to {OUT_DIR}/")
