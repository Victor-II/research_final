"""Detailed error analysis: run a checkpoint on test data and categorise every error."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from sentence_transformers import SentenceTransformer
import numpy as np

from src.data.data import load_aste_file, to_generative_format, enrich_syntax, Task
from src.eval.eval import parse_output, prf
from src.model.t5_model import T5ABSAModel


def load_model(ckpt_path, device="cuda"):
    model = T5ABSAModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def run_inference(model, examples, output_format="natural-language", device="cuda", batch_size=32):
    """Run batched inference and return per-example results."""
    tasks = [Task.ASPECT, Task.SENTIMENT, Task.POLARITY]
    keys = ["aspect", "sentiment", "polarity"]

    # prepare all inputs
    gen_data = []
    for ex in examples:
        gen = to_generative_format(ex, tasks, output_format=output_format)
        gen_data.append((ex, gen))

    results = []
    for start in range(0, len(gen_data), batch_size):
        batch = gen_data[start:start + batch_size]
        input_texts = [g["input"] for _, g in batch]

        encoded = model.tokenizer(
            input_texts, return_tensors="pt", max_length=512,
            truncation=True, padding=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.model.generate(
                **encoded, max_new_tokens=128, num_beams=1,
            )

        for i, (ex, gen) in enumerate(batch):
            raw_output = model.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            pred = parse_output(raw_output, keys, output_format)
            gold = parse_output(gen["target"], keys, output_format)
            results.append({
                "sentence": ex["sentence"],
                "input": gen["input"],
                "raw_output": raw_output,
                "gold_target": gen["target"],
                "pred": pred,
                "gold": gold,
                "canonical": ex,
            })

        print(f"    {min(start + batch_size, len(gen_data))}/{len(gen_data)}")

    return results


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def classify_errors(results, train_aspects, embed_model, sim_threshold=0.7):
    """Classify each prediction into error categories."""

    train_aspect_set = set(a.lower() for a in train_aspects)

    # Pre-compute embeddings for all unique aspects
    all_aspects = set()
    for r in results:
        for t in r["pred"]:
            if t.get("aspect"):
                all_aspects.add(t["aspect"].lower())
        for t in r["gold"]:
            if t.get("aspect"):
                all_aspects.add(t["aspect"].lower())
    all_aspects = list(all_aspects)
    if all_aspects:
        aspect_embs = embed_model.encode(all_aspects, convert_to_numpy=True, show_progress_bar=False)
        aspect_to_emb = dict(zip(all_aspects, aspect_embs))
    else:
        aspect_to_emb = {}

    analysis = []

    for r in results:
        pred_set = [frozenset(t.items()) for t in r["pred"]]
        gold_set = [frozenset(t.items()) for t in r["gold"]]

        pred_matched = [False] * len(r["pred"])
        gold_matched = [False] * len(r["gold"])

        # exact matches
        for i, p in enumerate(pred_set):
            for j, g in enumerate(gold_set):
                if p == g and not gold_matched[j]:
                    pred_matched[i] = True
                    gold_matched[j] = True
                    break

        # classify unmatched predictions (false positives)
        fp_errors = []
        for i, p in enumerate(r["pred"]):
            if pred_matched[i]:
                continue
            error = classify_single_fp(p, r["gold"], gold_matched, train_aspect_set, aspect_to_emb, sim_threshold, r["sentence"])
            fp_errors.append(error)

        # classify unmatched golds (false negatives)
        fn_errors = []
        for j, g in enumerate(r["gold"]):
            if gold_matched[j]:
                continue
            error = classify_single_fn(g, r["pred"], pred_matched, train_aspect_set, aspect_to_emb, sim_threshold)
            fn_errors.append(error)

        # malformed output check
        malformed = len(r["gold"]) > 0 and len(r["pred"]) == 0 and r["raw_output"].strip() != ""

        analysis.append({
            "sentence": r["sentence"],
            "raw_output": r["raw_output"],
            "gold_target": r["gold_target"],
            "pred": r["pred"],
            "gold": r["gold"],
            "n_correct": sum(pred_matched),
            "fp_errors": fp_errors,
            "fn_errors": fn_errors,
            "malformed": malformed,
        })

    return analysis


_PRONOUNS = {"it", "this", "that", "they", "them", "its", "these", "those", "thing", "things", "everything", "something", "nothing", "one"}


def _is_proper_noun_or_brand(text, sentence):
    """Check if text looks like a product/brand name (capitalized, not sentence-start)."""
    words = text.split()
    if not words:
        return False
    # if all words are capitalized and it's not the start of the sentence
    sent_start = sentence.strip().lower().startswith(text.lower())
    if all(w[0].isupper() for w in words if w) and not sent_start:
        return True
    # common product patterns
    lower = text.lower()
    if any(p in lower for p in ["mac", "imac", "mbp", "ipad", "iphone", "thinkpad", "lenovo", "dell", "hp", "asus", "acer"]):
        return True
    return False


def _aspect_in_sentence(aspect, sentence):
    """Check if the aspect text appears as a substring in the sentence."""
    return aspect.lower() in sentence.lower()


def classify_single_fp(pred, golds, gold_matched, train_aspects, aspect_embs, threshold, sentence=""):
    """Classify a single false positive prediction."""
    pa = pred.get("aspect", "").lower()
    pp = pred.get("polarity", "").lower()
    ps = pred.get("sentiment", "").lower()
    sent_lower = sentence.lower()

    # check if aspect matches a gold but polarity/sentiment differs
    for j, g in enumerate(golds):
        if gold_matched[j]:
            continue
        ga = g.get("aspect", "").lower()
        gp = g.get("polarity", "").lower()
        gs = g.get("sentiment", "").lower()

        if pa == ga:
            if pp != gp:
                return {"type": "polarity_error", "pred": pred, "gold": g}
            if ps != gs:
                return {"type": "sentiment_error", "pred": pred, "gold": g}

    # check if aspect is semantically close to a gold aspect (boundary/paraphrase error)
    if pa in aspect_embs:
        for j, g in enumerate(golds):
            if gold_matched[j]:
                continue
            ga = g.get("aspect", "").lower()
            if ga in aspect_embs:
                sim = cosine_sim(aspect_embs[pa], aspect_embs[ga])
                if sim >= threshold:
                    gp = g.get("polarity", "").lower()
                    if pp != gp:
                        return {"type": "aspect_boundary+polarity", "pred": pred, "gold": g, "sim": sim}
                    return {"type": "aspect_boundary", "pred": pred, "gold": g, "sim": sim}

    # check if sentiment word is not in the input sentence
    if ps and ps not in sent_lower:
        return {"type": "hallucinated_sentiment", "pred": pred}

    # check if aspect is from training domain
    if pa in train_aspects:
        # distinguish: is the aspect actually in the input sentence?
        if _aspect_in_sentence(pa, sentence):
            return {"type": "unannotated_aspect", "pred": pred}
        else:
            return {"type": "hallucinated_train_aspect", "pred": pred}

    # spurious — split into subcategories
    raw_aspect = pred.get("aspect", "")
    if raw_aspect.lower() in _PRONOUNS:
        return {"type": "spurious_pronoun", "pred": pred}
    if _is_proper_noun_or_brand(raw_aspect, sentence):
        return {"type": "spurious_product_name", "pred": pred}
    return {"type": "spurious_other", "pred": pred}


def classify_single_fn(gold, preds, pred_matched, train_aspects, aspect_embs, threshold):
    """Classify a single false negative (missed gold)."""
    ga = gold.get("aspect", "").lower()
    gp = gold.get("polarity", "").lower()
    gs = gold.get("sentiment", "").lower()

    # check if a pred has a close aspect (was partially matched)
    if ga in aspect_embs:
        for i, p in enumerate(preds):
            if pred_matched[i]:
                continue
            pa = p.get("aspect", "").lower()
            if pa in aspect_embs:
                sim = cosine_sim(aspect_embs[pa], aspect_embs[ga])
                if sim >= threshold:
                    pp = p.get("polarity", "").lower()
                    ps = p.get("sentiment", "").lower()
                    # determine what's wrong: aspect boundary, sentiment, polarity, or combo
                    aspect_exact = (pa == ga)
                    sentiment_exact = (ps == gs)
                    polarity_exact = (pp == gp)
                    if aspect_exact and not sentiment_exact:
                        # shouldn't happen (would have been caught as sentiment_error in FP),
                        # but just in case
                        return {"type": "near_miss_sentiment", "gold": gold, "pred": p, "sim": sim}
                    if not aspect_exact and polarity_exact and sentiment_exact:
                        return {"type": "near_miss_aspect", "gold": gold, "pred": p, "sim": sim}
                    if not aspect_exact and not polarity_exact:
                        return {"type": "near_miss_aspect+polarity", "gold": gold, "pred": p, "sim": sim}
                    if not aspect_exact and not sentiment_exact:
                        return {"type": "near_miss_aspect+sentiment", "gold": gold, "pred": p, "sim": sim}
                    # aspect close but not exact, polarity matches, sentiment matches
                    return {"type": "near_miss_aspect", "gold": gold, "pred": p, "sim": sim}

    # is this an unseen aspect?
    if ga not in train_aspects:
        return {"type": "unseen_aspect", "gold": gold}

    # seen aspect but missed
    return {"type": "missed_seen", "gold": gold}


def aggregate_errors(analysis):
    """Aggregate error statistics."""
    fp_types = Counter()
    fn_types = Counter()
    total_correct = 0
    total_gold = 0
    total_pred = 0
    total_malformed = 0
    n_perfect = 0
    n_empty_pred = 0

    # detailed tracking
    polarity_confusion = Counter()  # (pred_pol, gold_pol)
    hallucinated_aspects = Counter()
    hallucinated_sentiments = Counter()
    unannotated_aspects = Counter()
    missed_unseen_aspects = Counter()
    boundary_examples = []
    spurious_pronoun_examples = []
    spurious_product_examples = []
    spurious_other_examples = []

    for a in analysis:
        n_gold = len(a["gold"])
        n_pred = len(a["pred"])
        total_gold += n_gold
        total_pred += n_pred
        total_correct += a["n_correct"]
        if a["malformed"]:
            total_malformed += 1
        if n_gold > 0 and n_pred == 0:
            n_empty_pred += 1
        if not a["fp_errors"] and not a["fn_errors"] and n_gold > 0:
            n_perfect += 1

        for e in a["fp_errors"]:
            fp_types[e["type"]] += 1
            if e["type"] == "polarity_error":
                polarity_confusion[(e["pred"]["polarity"], e["gold"]["polarity"])] += 1
            elif e["type"] == "hallucinated_train_aspect":
                hallucinated_aspects[e["pred"]["aspect"].lower()] += 1
            elif e["type"] == "unannotated_aspect":
                unannotated_aspects[e["pred"]["aspect"].lower()] += 1
            elif e["type"] == "hallucinated_sentiment":
                hallucinated_sentiments[e["pred"]["sentiment"].lower()] += 1
            elif e["type"] in ("aspect_boundary", "aspect_boundary+polarity"):
                boundary_examples.append(e)
            elif e["type"] == "spurious_pronoun":
                spurious_pronoun_examples.append(e)
            elif e["type"] == "spurious_product_name":
                spurious_product_examples.append(e)
            elif e["type"] == "spurious_other":
                spurious_other_examples.append(e)

        for e in a["fn_errors"]:
            fn_types[e["type"]] += 1
            if e["type"] == "unseen_aspect":
                missed_unseen_aspects[e["gold"]["aspect"].lower()] += 1

    return {
        "total_sentences": len(analysis),
        "total_gold_triplets": total_gold,
        "total_pred_triplets": total_pred,
        "total_correct": total_correct,
        "n_perfect_sentences": n_perfect,
        "n_empty_predictions": n_empty_pred,
        "n_malformed": total_malformed,
        "fp_types": dict(fp_types.most_common()),
        "fn_types": dict(fn_types.most_common()),
        "polarity_confusion": {f"{p}->{g}": c for (p, g), c in polarity_confusion.most_common()},
        "hallucinated_train_aspects": dict(Counter(hallucinated_aspects).most_common(20)),
        "unannotated_aspects": dict(Counter(unannotated_aspects).most_common(20)),
        "hallucinated_sentiments": dict(Counter(hallucinated_sentiments).most_common(20)),
        "missed_unseen_aspects": dict(Counter(missed_unseen_aspects).most_common(20)),
        "n_boundary_errors": len(boundary_examples),
        "boundary_examples": boundary_examples[:10],
        "n_spurious_pronoun": len(spurious_pronoun_examples),
        "spurious_pronoun_examples": spurious_pronoun_examples[:10],
        "n_spurious_product": len(spurious_product_examples),
        "spurious_product_examples": spurious_product_examples[:10],
        "n_spurious_other": len(spurious_other_examples),
        "spurious_other_examples": spurious_other_examples[:10],
    }


def print_report(stats, label):
    import sys
    _print_report_to(sys.stdout, stats, label)


def _print_report_to(out, stats, label):
    total_g = stats["total_gold_triplets"]
    total_p = stats["total_pred_triplets"]
    total_c = stats["total_correct"]
    p = total_c / total_p if total_p else 0
    r = total_c / total_g if total_g else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0

    out.write(f"\n{'='*80}\n")
    out.write(f"ERROR ANALYSIS: {label}\n")
    out.write(f"{'='*80}\n")

    out.write(f"\nOverall: P={p:.4f} R={r:.4f} F1={f1:.4f}\n")
    out.write(f"Sentences: {stats['total_sentences']}\n")
    out.write(f"Gold triplets: {total_g}, Pred triplets: {total_p}, Correct: {total_c}\n")
    out.write(f"Perfect sentences: {stats['n_perfect_sentences']}\n")
    out.write(f"Empty predictions: {stats['n_empty_predictions']}\n")
    out.write(f"Malformed outputs: {stats['n_malformed']}\n")

    total_fp = total_p - total_c
    total_fn = total_g - total_c

    out.write(f"\n--- FALSE POSITIVES ({total_fp} total) ---\n")
    for t, c in sorted(stats["fp_types"].items(), key=lambda x: -x[1]):
        pct = c / total_fp * 100 if total_fp else 0
        out.write(f"  {t}: {c} ({pct:.1f}%)\n")

    out.write(f"\n--- FALSE NEGATIVES ({total_fn} total) ---\n")
    for t, c in sorted(stats["fn_types"].items(), key=lambda x: -x[1]):
        pct = c / total_fn * 100 if total_fn else 0
        out.write(f"  {t}: {c} ({pct:.1f}%)\n")

    if stats["polarity_confusion"]:
        out.write(f"\n--- POLARITY CONFUSION ---\n")
        for k, c in sorted(stats["polarity_confusion"].items(), key=lambda x: -x[1]):
            out.write(f"  {k}: {c}\n")

    if stats.get("unannotated_aspects"):
        out.write(f"\n--- UNANNOTATED ASPECTS (in sentence but not in gold, top 20) ---\n")
        for a, c in list(stats["unannotated_aspects"].items())[:20]:
            out.write(f"  {a}: {c}\n")

    if stats["hallucinated_train_aspects"]:
        out.write(f"\n--- HALLUCINATED TRAIN ASPECTS (not in sentence, top 20) ---\n")
        for a, c in list(stats["hallucinated_train_aspects"].items())[:20]:
            out.write(f"  {a}: {c}\n")

    if stats.get("hallucinated_sentiments"):
        out.write(f"\n--- HALLUCINATED SENTIMENTS (top 20) ---\n")
        for a, c in list(stats["hallucinated_sentiments"].items())[:20]:
            out.write(f"  {a}: {c}\n")

    if stats["missed_unseen_aspects"]:
        out.write(f"\n--- MISSED UNSEEN ASPECTS (top 20) ---\n")
        for a, c in list(stats["missed_unseen_aspects"].items())[:20]:
            out.write(f"  {a}: {c}\n")

    if stats["boundary_examples"]:
        out.write(f"\n--- ASPECT BOUNDARY ERRORS (showing {len(stats['boundary_examples'])}/{stats['n_boundary_errors']}) ---\n")
        for e in stats["boundary_examples"]:
            out.write(f"  pred: {e['pred']['aspect']} -> gold: {e['gold']['aspect']} (sim={e.get('sim',0):.2f})\n")

    if stats["spurious_pronoun_examples"]:
        out.write(f"\n--- SPURIOUS PRONOUN ASPECTS (showing {len(stats['spurious_pronoun_examples'])}/{stats['n_spurious_pronoun']}) ---\n")
        for e in stats["spurious_pronoun_examples"]:
            out.write(f"  {e['pred']}\n")

    if stats["spurious_product_examples"]:
        out.write(f"\n--- SPURIOUS PRODUCT NAME ASPECTS (showing {len(stats['spurious_product_examples'])}/{stats['n_spurious_product']}) ---\n")
        for e in stats["spurious_product_examples"]:
            out.write(f"  {e['pred']}\n")

    if stats["spurious_other_examples"]:
        out.write(f"\n--- SPURIOUS OTHER (showing {len(stats['spurious_other_examples'])}/{stats['n_spurious_other']}) ---\n")
        for e in stats["spurious_other_examples"]:
            out.write(f"  {e['pred']}\n")



def worst_sentences(analysis, n=20):
    """Return the n sentences with the most errors for manual inspection."""
    scored = []
    for a in analysis:
        n_errors = len(a["fp_errors"]) + len(a["fn_errors"])
        if n_errors > 0:
            scored.append((n_errors, a))
    scored.sort(key=lambda x: -x[0])
    return scored[:n]


def print_comparison_table(all_detailed):
    """Print a side-by-side comparison table across all test sets."""
    labels = list(all_detailed.keys())
    short_labels = [l.split("/")[0] for l in labels]
    stats_list = [all_detailed[l]["stats"] for l in labels]

    # collect all FP and FN types across all datasets
    all_fp_types = sorted(set(t for s in stats_list for t in s["fp_types"]))
    all_fn_types = sorted(set(t for s in stats_list for t in s["fn_types"]))

    # column width
    w = max(len(l) for l in short_labels) + 2
    w = max(w, 22)
    lw = 30  # label column width

    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    # header
    print(f"{'':>{lw}}", end="")
    for sl in short_labels:
        print(f"  {sl:>{w}}", end="")
    print()
    print("-" * (lw + (w + 2) * len(labels)))

    # overall metrics
    for metric_name, metric_fn in [
        ("F1", lambda s: _f1(s)),
        ("Precision", lambda s: _prec(s)),
        ("Recall", lambda s: _rec(s)),
    ]:
        print(f"{metric_name:>{lw}}", end="")
        for s in stats_list:
            print(f"  {metric_fn(s):>{w}}", end="")
        print()

    print()

    # counts
    for row_name, key in [
        ("Sentences", "total_sentences"),
        ("Gold triplets", "total_gold_triplets"),
        ("Pred triplets", "total_pred_triplets"),
        ("Correct", "total_correct"),
        ("Perfect sentences", "n_perfect_sentences"),
        ("Empty predictions", "n_empty_predictions"),
        ("Malformed", "n_malformed"),
    ]:
        print(f"{row_name:>{lw}}", end="")
        for s in stats_list:
            print(f"  {s[key]:>{w}}", end="")
        print()

    # total errors per dataset
    total_errors = []
    for s in stats_list:
        fp = s["total_pred_triplets"] - s["total_correct"]
        fn = s["total_gold_triplets"] - s["total_correct"]
        total_errors.append(fp + fn)

    print()
    print(f"{'Total FP':>{lw}}", end="")
    for s in stats_list:
        fp = s["total_pred_triplets"] - s["total_correct"]
        print(f"  {fp:>{w}}", end="")
    print()
    print(f"{'Total FN':>{lw}}", end="")
    for s in stats_list:
        fn = s["total_gold_triplets"] - s["total_correct"]
        print(f"  {fn:>{w}}", end="")
    print()
    print(f"{'Total errors':>{lw}}", end="")
    for te in total_errors:
        print(f"  {te:>{w}}", end="")
    print()

    # FP breakdown
    print(f"\n{'--- FP breakdown ---':>{lw}}")
    for t in all_fp_types:
        print(f"{'  ' + t:>{lw}}", end="")
        for i, s in enumerate(stats_list):
            c = s["fp_types"].get(t, 0)
            fp = s["total_pred_triplets"] - s["total_correct"]
            te = total_errors[i]
            fp_pct = c / fp * 100 if fp else 0
            te_pct = c / te * 100 if te else 0
            print(f"  {f'{c} ({fp_pct:.0f}%FP, {te_pct:.0f}%tot)':>{w}}", end="")
        print()

    # FN breakdown
    print(f"\n{'--- FN breakdown ---':>{lw}}")
    for t in all_fn_types:
        print(f"{'  ' + t:>{lw}}", end="")
        for i, s in enumerate(stats_list):
            c = s["fn_types"].get(t, 0)
            fn = s["total_gold_triplets"] - s["total_correct"]
            te = total_errors[i]
            fn_pct = c / fn * 100 if fn else 0
            te_pct = c / te * 100 if te else 0
            print(f"  {f'{c} ({fn_pct:.0f}%FN, {te_pct:.0f}%tot)':>{w}}", end="")
        print()

    # polarity confusion
    all_pol = sorted(set(k for s in stats_list for k in s["polarity_confusion"]))
    if all_pol:
        print(f"\n{'--- Polarity confusion ---':>{lw}}")
        for k in all_pol:
            print(f"{'  ' + k:>{lw}}", end="")
            for s in stats_list:
                c = s["polarity_confusion"].get(k, 0)
                print(f"  {c:>{w}}", end="")
            print()


def _f1(s):
    tp = s["total_correct"]
    p = tp / s["total_pred_triplets"] if s["total_pred_triplets"] else 0
    r = tp / s["total_gold_triplets"] if s["total_gold_triplets"] else 0
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    return f"{f1:.4f}"

def _prec(s):
    tp = s["total_correct"]
    p = tp / s["total_pred_triplets"] if s["total_pred_triplets"] else 0
    return f"{p:.4f}"

def _rec(s):
    tp = s["total_correct"]
    r = tp / s["total_gold_triplets"] if s["total_gold_triplets"] else 0
    return f"{r:.4f}"


def main():
    import argparse
    import io
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-files", nargs="+", required=True)
    parser.add_argument("--train-files", nargs="+", required=True)
    parser.add_argument("--output-format", default="natural-language")
    parser.add_argument("--syntax", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sim-threshold", type=float, default=0.7)
    parser.add_argument("--save", default="tools/error_analysis_results.json", help="Save detailed JSON to this path")
    args = parser.parse_args()

    # Collect train aspects
    train_aspects = set()
    for f in args.train_files:
        for ex in load_aste_file(f):
            for ann in ex["annotations"]:
                if ann.get("aspect"):
                    train_aspects.add(ann["aspect"].lower())

    print(f"Train aspects: {len(train_aspects)} unique")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint, args.device)

    # Load embedding model
    print("Loading sentence-transformers...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=args.device)

    all_detailed = {}

    for test_file in args.test_files:
        label = Path(test_file).parts[-2] + "/" + Path(test_file).stem
        print(f"Running inference on: {test_file}", end="", flush=True)

        examples = load_aste_file(test_file)
        if args.syntax:
            examples = enrich_syntax(examples, args.syntax)

        results = run_inference(model, examples, args.output_format, args.device)
        analysis = classify_errors(results, train_aspects, embed_model, args.sim_threshold)
        stats = aggregate_errors(analysis)

        # capture per-dataset report to string for JSON
        buf = io.StringIO()
        _print_report_to(buf, stats, label)

        worst = worst_sentences(analysis, 15)
        buf.write(f"\n--- WORST SENTENCES (top {len(worst)}) ---\n")
        for n_err, a in worst:
            buf.write(f"\n  [{n_err} errors] {a['sentence']}\n")
            buf.write(f"  Gold:   {a['gold_target']}\n")
            buf.write(f"  Pred:   {a['raw_output']}\n")
            for e in a["fp_errors"]:
                buf.write(f"    FP [{e['type']}]: {e['pred']}\n")
            for e in a["fn_errors"]:
                buf.write(f"    FN [{e['type']}]: {e['gold']}\n")

        all_detailed[label] = {
            "stats": stats,
            "report": buf.getvalue(),
            "per_sentence": [
                {
                    "sentence": a["sentence"],
                    "raw_output": a["raw_output"],
                    "gold_target": a["gold_target"],
                    "pred": a["pred"],
                    "gold": a["gold"],
                    "n_correct": a["n_correct"],
                    "fp_errors": [{"type": e["type"], "pred": e.get("pred"), "gold": e.get("gold"), "sim": e.get("sim")} for e in a["fp_errors"]],
                    "fn_errors": [{"type": e["type"], "pred": e.get("pred"), "gold": e.get("gold"), "sim": e.get("sim")} for e in a["fn_errors"]],
                }
                for a in analysis
            ],
        }
        print(f" ... {len(examples)} examples, F1={_f1(stats)}")

    # Print comparison table to console
    if len(all_detailed) > 1:
        print_comparison_table(all_detailed)
    else:
        # single dataset — print the report directly
        for label, data in all_detailed.items():
            print(data["report"])

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    with open(args.save, "w") as f:
        json.dump(all_detailed, f, indent=2, default=str)
    print(f"\nDetailed results saved to {args.save}")


if __name__ == "__main__":
    main()
