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
    """Micro-averaged precision, recall, F1 over structured predictions."""
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


def macro_prf(
    preds: list[list[dict]],
    golds: list[list[dict]],
    key: str,
) -> dict:
    """Macro-averaged precision, recall, F1 over a single label key."""
    classes = {d[key] for gold in golds for d in gold if key in d}
    per_class = {}
    for cls in classes:
        tp = fp = fn = 0
        for pred_list, gold_list in zip(preds, golds):
            pred_vals = {frozenset(d.items()) for d in pred_list if d.get(key) == cls}
            gold_vals = {frozenset(d.items()) for d in gold_list if d.get(key) == cls}
            tp += len(pred_vals & gold_vals)
            fp += len(pred_vals - gold_vals)
            fn += len(gold_vals - pred_vals)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        per_class[cls] = {"precision": p, "recall": r, "f1": f}
    macro = {
        m: sum(v[m] for v in per_class.values()) / len(per_class)
        for m in ("precision", "recall", "f1")
    } if per_class else {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {"macro": macro, "per_class": per_class}


def soft_prf(
    preds: list[list[dict]],
    golds: list[list[dict]],
    key: str,
    threshold: float = 0.8,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict:
    """
    Soft precision, recall, F1 using embedding similarity for a single key.
    A predicted value is a soft match if cosine similarity to a gold value >= threshold.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    if key != "aspect":
        raise ValueError("soft_prf is only meaningful for the 'aspect' key")

    model = SentenceTransformer(model_name)

    tp = fp = fn = 0
    for pred_list, gold_list in zip(preds, golds):
        pred_vals = [d[key] for d in pred_list if key in d]
        gold_vals = [d[key] for d in gold_list if key in d]

        if not pred_vals and not gold_vals:
            continue
        if not pred_vals:
            fn += len(gold_vals)
            continue
        if not gold_vals:
            fp += len(pred_vals)
            continue

        all_spans = pred_vals + gold_vals
        embeddings = model.encode(all_spans, convert_to_numpy=True)
        pred_embs = embeddings[:len(pred_vals)]
        gold_embs = embeddings[len(pred_vals):]

        # greedy matching: each gold can only be matched once
        matched_gold = set()
        for i, pe in enumerate(pred_embs):
            best_sim = -1
            best_j = -1
            for j, ge in enumerate(gold_embs):
                if j in matched_gold:
                    continue
                sim = float(np.dot(pe, ge) / (np.linalg.norm(pe) * np.linalg.norm(ge) + 1e-9))
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_sim >= threshold:
                tp += 1
                matched_gold.add(best_j)
            else:
                fp += 1
        fn += len(gold_vals) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate(
    preds: list[list[dict]],
    golds: list[list[dict]],
    eval_scopes: list[dict],
) -> dict[str, dict]:
    """
    Run requested metrics for each scope in eval_scopes.

    Each scope is a dict with 'keys' (list[str]) and 'metrics' (list[str]).
    Supported metrics: 'micro_f1', 'macro_f1'.
    Returns dict keyed by "+".join(keys).
    """
    results = {}
    for scope in eval_scopes:
        keys    = scope["keys"]
        metrics = scope.get("metrics", ["micro_f1"])
        label   = "+".join(keys)
        results[label] = {}
        if "micro_f1" in metrics:
            results[label]["micro"] = prf(preds, golds, keys)
        if "macro_f1" in metrics:
            if len(keys) != 1:
                raise ValueError(f"macro_f1 requires exactly one key, got {keys}")
            results[label]["macro"] = macro_prf(preds, golds, keys[0])
        if "soft_f1" in metrics:
            if keys != ["aspect"]:
                raise ValueError("soft_f1 is only valid for keys=['aspect']")
            threshold = scope.get("soft_threshold", 0.8)
            model_name = scope.get("soft_model", "all-MiniLM-L6-v2")
            results[label]["soft"] = soft_prf(preds, golds, "aspect", threshold, model_name)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_results(
    val_history: list[dict],
    test_history: list[dict],
    train_loss_history: list[float],
    val_loss_history: list[float],
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    results = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
        "val": val_history,
        "test": test_history,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def save_metrics_table(
    metrics: dict[str, dict],
    epoch: int,
    out_dir: str = ".",
    prefix: str = "metrics",
):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_epoch{epoch}.txt")

    col_w = max(len(k) for k in metrics) + 2
    header = f"{'scope':<{col_w}} {'type':<10} {'precision':>10} {'recall':>10} {'f1':>10}"
    sep    = "-" * len(header)

    lines = [f"Epoch {epoch}", sep, header, sep]
    for scope, result in metrics.items():
        if "micro" in result:
            m = result["micro"]
            lines.append(f"{scope:<{col_w}} {'micro':<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
        if "macro" in result:
            m = result["macro"]["macro"]
            lines.append(f"{scope:<{col_w}} {'macro':<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
            for cls, cm in result["macro"]["per_class"].items():
                lines.append(f"{scope:<{col_w}} {cls:<10} {cm['precision']:>10.4f} {cm['recall']:>10.4f} {cm['f1']:>10.4f}")
        if "soft" in result:
            m = result["soft"]
            lines.append(f"{scope:<{col_w}} {'soft':<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
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

