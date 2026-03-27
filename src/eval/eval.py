"""
General-purpose evaluation utilities.

All metric functions operate on lists of dicts (one per example), making them
agnostic to the specific task or dataset format. The keys to evaluate on are
passed explicitly, so the same functions work for triplet extraction (ASTE),
aspect-polarity classification (APC), or any other structured output.

Prediction / gold format
------------------------
Each example is represented as a list of dicts, e.g.:
  - ASTE: [{"aspect": "food", "sentiment": "great", "polarity": "positive"}, ...]
  - APC:  [{"aspect": "battery life", "polarity": "positive"}]
  - ATE:  [{"aspect": "screen"}]

The functions accept:
  preds: list[list[dict]]  — one list of dicts per example
  golds: list[list[dict]]  — matching gold list of dicts per example
  keys:  list[str]         — which dict keys to include in comparison
"""

import json
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_output(raw: str, keys: list[str]) -> list[dict]:
    """Parse bracket notation '[v1, v2, v3] [v1, v2]' into a list of dicts given key order."""
    results = []
    for match in re.finditer(r"\[([^\[\]]+)\]", raw):
        values = [v.strip() for v in match.group(1).split(",")]
        if len(values) == len(keys):
            results.append(dict(zip(keys, values)))
    return results


def project(items: list[dict], keys: list[str]) -> list[frozenset]:
    """Project each dict to only the specified keys, returned as frozensets for set comparison."""
    projected = []
    for d in items:
        subset = {k: d[k] for k in keys if k in d}
        if subset:
            projected.append(frozenset(subset.items()))
    return projected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def prf(
    preds: list[list[dict]],
    golds: list[list[dict]],
    keys: list[str],
) -> dict:
    """
    Micro-averaged precision, recall, F1 over structured predictions.

    Args:
        preds: predicted outputs, one list of dicts per example.
        golds: gold outputs, one list of dicts per example.
        keys:  which fields to include in the comparison.
    """
    tp = fp = fn = 0
    for pred, gold in zip(preds, golds):
        pred_set = set(project(pred, keys))
        gold_set = set(project(gold, keys))
        tp += len(pred_set & gold_set)
        fp += len(pred_set - gold_set)
        fn += len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(
    preds: list[list[dict]],
    golds: list[list[dict]],
    eval_keys: list[list[str]],
) -> dict[str, dict]:
    """
    Run PRF for each key combination in eval_keys.

    Returns:
        dict keyed by "+".join(keys), e.g. "aspect+polarity+sentiment"
    """
    return {
        "+".join(keys): prf(preds, golds, keys)
        for keys in eval_keys
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_metrics_table(
    metrics: dict[str, dict],
    epoch: int,
    out_dir: str = ".",
):
    """Save a formatted metrics table to a .txt file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"metrics_epoch{epoch}.txt")

    col_w = max(len(k) for k in metrics) + 2
    header = f"{'scope':<{col_w}} {'precision':>10} {'recall':>10} {'f1':>10}"
    sep    = "-" * len(header)

    lines = [f"Epoch {epoch}", sep, header, sep]
    for scope, m in metrics.items():
        lines.append(
            f"{scope:<{col_w}} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}"
        )
    lines.append(sep)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_loss_curve(
    val_losses: list[float],
    epoch: int,
    out_dir: str = ".",
):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(val_losses, label="val loss", alpha=0.7)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"Val Loss — epoch {epoch}")
    ax.legend()
    fig.savefig(os.path.join(out_dir, f"loss_epoch{epoch}.png"), dpi=120)
    plt.close(fig)


def plot_label_confusion(
    preds: list[list[dict]],
    golds: list[list[dict]],
    match_keys: list[str],
    label_key: str,
    epoch: int,
    out_dir: str = ".",
):
    """
    Confusion matrix for a label field, evaluated on examples where match_keys align.

    Args:
        match_keys: keys used to match pred to gold (e.g. ["aspect", "sentiment"])
        label_key:  the field whose predicted vs gold value is plotted (e.g. "polarity")
    """
    os.makedirs(out_dir, exist_ok=True)
    true_labels, pred_labels = [], []

    for pred_list, gold_list in zip(preds, golds):
        gold_map = {
            frozenset((k, d[k]) for k in match_keys if k in d): d
            for d in gold_list
        }
        for pd in pred_list:
            key = frozenset((k, pd[k]) for k in match_keys if k in pd)
            if key in gold_map and label_key in pd and label_key in gold_map[key]:
                true_labels.append(gold_map[key][label_key])
                pred_labels.append(pd[label_key])

    if not true_labels:
        return

    labels = sorted(set(true_labels) | set(pred_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, colorbar=False)
    ax.set_title(f"{label_key} confusion — epoch {epoch}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label_key}_confusion_epoch{epoch}.png"), dpi=120)
    plt.close(fig)


def run_all_plots(
    preds: list[list[dict]],
    golds: list[list[dict]],
    val_losses: list[float],
    eval_keys: list[list[str]],
    epoch: int,
    out_dir: str = ".",
):
    """
    Convenience function: runs val loss curve + a confusion matrix for every
    single-key group, using all other available keys as match keys.
    """
    plot_loss_curve(val_losses, epoch, out_dir)

    all_keys = sorted({k for gold_list in golds for d in gold_list for k in d})

    for keys in eval_keys:
        if len(keys) == 1:
            label_key  = keys[0]
            # Only plot confusion matrix for classification labels, not free-text fields
            if label_key == "polarity":
                match_keys = [k for k in all_keys if k != label_key]
                if match_keys:
                    plot_label_confusion(preds, golds, match_keys, label_key, epoch, out_dir)

