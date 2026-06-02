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

def parse_output(raw: str, keys: list[str], output_format: str = "structured", language: str = "en") -> list[dict]:
    """Parse model output into a list of dicts given key order."""
    if output_format == "natural-language":
        return _parse_nl_output(raw, keys, language=language)
    if output_format == "mvp-markers":
        return _parse_mvp_markers_output(raw, keys)
    results = []
    for match in re.finditer(r"\[([^\[\]]+)\]", raw):
        values = [v.strip() for v in match.group(1).split(",")]
        if len(values) == len(keys):
            results.append(dict(zip(keys, values)))
    return results


def _parse_mvp_markers_output(raw: str, keys: list[str]) -> list[dict]:
    """Parse MvP marker format output into structured dicts.

    Format: [A] pizza [O] delicious [S] positive [SSEP] [A] service [O] slow [S] negative
    """
    KEY_TO_MARKER = {"aspect": "[A]", "sentiment": "[O]", "polarity": "[S]", "category": "[C]"}
    MARKER_TO_KEY = {v: k for k, v in KEY_TO_MARKER.items()}

    # split on [SSEP]
    tuples_raw = re.split(r'\[SSEP\]', raw)
    results = []

    # build marker regex for the expected keys
    markers = [KEY_TO_MARKER.get(k, f"[{k[0].upper()}]") for k in keys]
    # escape markers for regex
    marker_patterns = [re.escape(m) for m in markers]

    for tuple_str in tuples_raw:
        tuple_str = tuple_str.strip()
        if not tuple_str:
            continue

        # try to parse: extract value after each marker, up to the next marker or end
        d = {}
        for i, (key, marker_pat) in enumerate(zip(keys, marker_patterns)):
            # look for this marker followed by content until the next marker or end
            if i < len(keys) - 1:
                next_markers = "|".join(marker_patterns[i+1:] + [r'\[SSEP\]'])
                pattern = marker_pat + r'\s*(.+?)(?=\s*(?:' + next_markers + r')\s*|$)'
            else:
                pattern = marker_pat + r'\s*(.+?)$'
            m = re.search(pattern, tuple_str)
            if m:
                d[key] = m.group(1).strip()

        if len(d) == len(keys) and all(d.values()):
            results.append(d)

    return results


def _parse_nl_output(raw: str, keys: list[str], language: str = "en") -> list[dict]:
    """Parse natural-language template output back into structured dicts."""
    from src.data.templates import get_templates, get_plural_templates, get_all_templates_for_keys, translate_from_output
    from constants import CANONICAL_KEY_ORDER
    canonical_keys = [k for k in CANONICAL_KEY_ORDER if k in keys]
    keys_set = frozenset(canonical_keys)
    templates, implicit_templates = get_templates(language)
    template = templates.get(keys_set)
    if not template:
        return []

    # try plural form first for single-key tasks
    if len(keys_set) == 1:
        plural_templates = get_plural_templates(language)
        plural_tmpl = plural_templates.get(keys_set)
        if plural_tmpl:
            key = next(iter(keys_set))
            # build regex: replace {list} with a capture group
            plural_pattern = re.escape(plural_tmpl).replace(re.escape("{list}"), "(.+)")
            m = re.match(plural_pattern, raw)
            if m:
                values = [v.strip() for v in m.group(1).split(",")]
                return [{key: translate_from_output(v, key, language)} for v in values if v]

    # get all template variants (frozenset + ordered tuples)
    all_templates = get_all_templates_for_keys(keys_set, language)

    def _build_regex(tmpl, ckeys):
        p = tmpl
        for key in ckeys:
            p = p.replace("{" + key + "}", f"(?P<{key}>.+?)")
        p = p[::-1].replace("?+.", "+.", 1)[::-1]
        return re.compile(p)

    # build regexes for all template variants
    regexes = []

    # 1. annotated implicit aspect + implicit sentiment
    impl_sent_tmpl = implicit_templates.get(keys_set)
    if impl_sent_tmpl and "aspect" in canonical_keys:
        annotated_impl_sent = impl_sent_tmpl.replace("{aspect}", "the implied aspect (?P<aspect>.+?)")
        # need to rebuild without the standard {aspect} replacement
        p = impl_sent_tmpl
        for key in canonical_keys:
            if key == "aspect":
                p = p.replace("{aspect}", "the implied aspect (?P<aspect>.+?)")
            else:
                p = p.replace("{" + key + "}", f"(?P<{key}>.+?)")
        p = p[::-1].replace("?+.", "+.", 1)[::-1]
        regexes.append(("annotated_impl_both", re.compile(p)))

    # 2. unknown aspect + implicit sentiment
    if impl_sent_tmpl and "aspect" in canonical_keys:
        p = impl_sent_tmpl.replace("{aspect}", "an unspecified aspect")
        for key in canonical_keys:
            if key != "aspect":
                p = p.replace("{" + key + "}", f"(?P<{key}>.+?)")
        p = p[::-1].replace("?+.", "+.", 1)[::-1]
        regexes.append(("unknown_aspect_impl_sent", re.compile(p)))

    # 3. explicit aspect + implicit sentiment
    if impl_sent_tmpl:
        regexes.append(("impl_sent", _build_regex(impl_sent_tmpl, canonical_keys)))

    # 4. annotated implicit aspect + explicit sentiment
    if "aspect" in canonical_keys:
        annotated_tmpl = template.replace("{aspect}", "the implied aspect {aspect}")
        regexes.append(("annotated_impl", _build_regex(annotated_tmpl, canonical_keys)))

    # 5. unknown aspect + explicit sentiment
    if "aspect" in canonical_keys:
        unknown_tmpl = template.replace("{aspect}", "an unspecified aspect")
        p = unknown_tmpl
        for key in canonical_keys:
            if key != "aspect":
                p = p.replace("{" + key + "}", f"(?P<{key}>.+?)")
        p = p[::-1].replace("?+.", "+.", 1)[::-1]
        regexes.append(("unknown_aspect", re.compile(p)))

    # 6. standard explicit template (last, least specific)
    regexes.append(("explicit", _build_regex(template, canonical_keys)))

    # 7. additional ordered template variants
    for alt_tmpl in all_templates:
        if alt_tmpl != template:
            regexes.append(("explicit", _build_regex(alt_tmpl, canonical_keys)))

    segments = [s.strip() for s in raw.split(" ; ")]
    results = []
    for seg in segments:
        matched = False
        for variant, regex in regexes:
            m = regex.match(seg)
            if not m:
                continue
            d = {}
            for k in canonical_keys:
                if k in m.groupdict():
                    d[k] = m.group(k).strip()
                elif k == "aspect" and "unknown_aspect" in variant:
                    d[k] = "IMPLICIT"
                elif k == "sentiment" and "impl_sent" in variant or "impl_both" in variant:
                    d[k] = "IMPLICIT"
            # tag implicit aspects
            if "annotated" in variant and "aspect" in d:
                d["aspect"] = f"IMPLICIT:{d['aspect']}"
            elif "unknown_aspect" in variant:
                d["aspect"] = "IMPLICIT"
            # tag implicit sentiments
            if "impl_sent" in variant or "impl_both" in variant:
                d["sentiment"] = "IMPLICIT"
            if len(d) == len(canonical_keys):
                # translate parsed values back to canonical English
                if language != "en":
                    for k in list(d.keys()):
                        if d[k] not in ("IMPLICIT",) and not d[k].startswith("IMPLICIT:"):
                            d[k] = translate_from_output(d[k], k, language)
                results.append(d)
                matched = True
                break
        # no match = malformed, scores zero (strict parsing)
    return results


def project(items: list[dict], keys: list[str]) -> list[frozenset]:
    """Project each dict to only the specified keys, returned as frozensets for set comparison."""
    projected = []
    for d in items:
        subset = {k: d[k] for k in keys if k in d}
        if subset:
            projected.append(frozenset(subset.items()))
    return projected


def _normalize_implicit(items: list[dict], mode: str | None) -> list[dict]:
    """Normalize IMPLICIT aspect/opinion values based on evaluation mode.

    mode=None/"full": no change, IMPLICIT:term must match exactly
    mode="collapse":  IMPLICIT:term → IMPLICIT (standard benchmark, NULL = NULL)
    mode="resolve":   IMPLICIT:term → term (compare only the inferred term)
    """
    if not mode or mode == "full":
        return items
    out = []
    for d in items:
        d = dict(d)
        for key in ("aspect", "sentiment"):
            val = d.get(key, "")
            if isinstance(val, str) and val.startswith("IMPLICIT:"):
                if mode == "collapse":
                    d[key] = "IMPLICIT"
                elif mode == "resolve":
                    d[key] = val[len("IMPLICIT:"):]
        out.append(d)
    return out


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


def _token_overlap_f1(a: str, b: str) -> float:
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    if not a_tokens or not b_tokens:
        return float(a_tokens == b_tokens)
    common = sum(min(a_tokens.count(t), b_tokens.count(t)) for t in set(a_tokens))
    p = common / len(a_tokens)
    r = common / len(b_tokens)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _lcs_length(a: list[str], b: list[str]) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def _lcs_f1(a: str, b: str) -> float:
    a_tokens = a.lower().split()
    b_tokens = b.lower().split()
    if not a_tokens or not b_tokens:
        return float(a_tokens == b_tokens)
    lcs = _lcs_length(a_tokens, b_tokens)
    p = lcs / len(a_tokens)
    r = lcs / len(b_tokens)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _tuple_similarity(pred: frozenset, gold: frozenset, sim_fn) -> float:
    pred_d = dict(pred)
    gold_d = dict(gold)
    if pred_d.keys() != gold_d.keys():
        return 0.0
    scores = [sim_fn(pred_d[k], gold_d[k]) for k in pred_d]
    return min(scores)


def lenient_prf(
    preds: list[list[dict]],
    golds: list[list[dict]],
    keys: list[str],
    sim_fn,
    threshold: float = 0.8,
) -> dict:
    tp = fp = fn = 0
    for pred_list, gold_list in zip(preds, golds):
        pred_tuples = project(pred_list, keys)
        gold_tuples = project(gold_list, keys)

        matched_gold = set()
        for pt in pred_tuples:
            best_sim = -1.0
            best_j = -1
            for j, gt in enumerate(gold_tuples):
                if j in matched_gold:
                    continue
                sim = _tuple_similarity(pt, gt, sim_fn)
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_sim >= threshold:
                tp += 1
                matched_gold.add(best_j)
            else:
                fp += 1
        fn += len(gold_tuples) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


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


def avg_overlap(
    preds: list[list[dict]],
    golds: list[list[dict]],
    keys: list[str],
) -> dict:
    """Compute average token overlap and LCS overlap between matched pred-gold pairs.

    Uses greedy matching (best overlap first). Reports mean overlap scores
    across all matched pairs, plus the fraction of predictions that have any match.
    """
    token_scores = []
    lcs_scores = []
    n_preds = 0
    n_matched = 0

    for pred_list, gold_list in zip(preds, golds):
        pred_tuples = project(pred_list, keys)
        gold_tuples = project(gold_list, keys)
        n_preds += len(pred_tuples)

        matched_gold = set()
        for pt in pred_tuples:
            best_token = 0.0
            best_lcs = 0.0
            best_j = -1
            for j, gt in enumerate(gold_tuples):
                if j in matched_gold:
                    continue
                tok_sim = _tuple_similarity(pt, gt, _token_overlap_f1)
                if tok_sim > best_token:
                    best_token = tok_sim
                    best_lcs = _tuple_similarity(pt, gt, _lcs_f1)
                    best_j = j
            if best_j >= 0 and best_token > 0:
                token_scores.append(best_token)
                lcs_scores.append(best_lcs)
                matched_gold.add(best_j)
                n_matched += 1

    return {
        "avg_token_overlap": sum(token_scores) / len(token_scores) if token_scores else 0.0,
        "avg_lcs_overlap": sum(lcs_scores) / len(lcs_scores) if lcs_scores else 0.0,
        "match_rate": n_matched / n_preds if n_preds else 0.0,
        "n_matched": n_matched,
        "n_preds": n_preds,
    }


def evaluate(
    preds: list[list[dict]],
    golds: list[list[dict]],
    eval_scopes: list[dict],
    implicit_mode: str | None = None,
) -> dict[str, dict]:
    """
    Run requested metrics for each scope in eval_scopes.

    Each scope is a dict with 'keys' (list[str]) and 'metrics' (list[str]).
    Supported metrics: 'micro_f1', 'macro_f1'.

    implicit_mode: None (no change) / "full" (same as None, IMPLICIT:x must match exactly),
                   "collapse" (IMPLICIT:x → IMPLICIT), "resolve" (IMPLICIT:x → x).
    Returns dict keyed by "+".join(keys).
    """
    if implicit_mode:
        preds = [_normalize_implicit(p, implicit_mode) for p in preds]
        golds = [_normalize_implicit(g, implicit_mode) for g in golds]

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
        if "token_f1" in metrics:
            threshold = scope.get("token_threshold", 0.8)
            results[label]["token"] = lenient_prf(preds, golds, keys, _token_overlap_f1, threshold)
        if "rouge_l" in metrics:
            threshold = scope.get("token_threshold", 0.8)
            results[label]["rouge_l"] = lenient_prf(preds, golds, keys, _lcs_f1, threshold)
        if "avg_overlap" in metrics:
            results[label]["avg_overlap"] = avg_overlap(preds, golds, keys)
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
        if "token" in result:
            m = result["token"]
            lines.append(f"{scope:<{col_w}} {'token':<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
        if "rouge_l" in result:
            m = result["rouge_l"]
            lines.append(f"{scope:<{col_w}} {'rouge_l':<10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
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

