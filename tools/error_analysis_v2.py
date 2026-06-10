"""Error analysis v2: element-level classification, no FP/FN split."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.data.data import load_aste_file, to_generative_format, Task
from src.eval.eval import parse_output
from src.model.t5_model import T5ABSAModel


def load_model(ckpt_path, device="cuda"):
    model = T5ABSAModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()
    model.to(device)
    return model


def run_inference(model, examples, output_format="natural-language", device="cuda", batch_size=32):
    tasks = [Task.ASPECT, Task.SENTIMENT, Task.POLARITY]
    keys = ["aspect", "sentiment", "polarity"]

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
            output_ids = model.model.generate(**encoded, max_new_tokens=128, num_beams=1)

        for i, (ex, gen) in enumerate(batch):
            raw_output = model.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            pred = parse_output(raw_output, keys, output_format)
            gold = parse_output(gen["target"], keys, output_format)
            results.append({
                "sentence": ex["sentence"],
                "raw_output": raw_output,
                "pred": pred,
                "gold": gold,
            })
        print(f"    {min(start + batch_size, len(gen_data))}/{len(gen_data)}")

    return results


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def span_in_sentence(span, sentence):
    """Check if span appears verbatim in the sentence (case-insensitive)."""
    return span.lower() in sentence.lower()


def spans_overlap(span_a, span_b):
    """Check if two spans share at least one word."""
    words_a = set(span_a.lower().split())
    words_b = set(span_b.lower().split())
    return len(words_a & words_b) > 0


def classify_span(pred_span, gold_spans, sentence, embeddings, sim_threshold=0.7):
    """
    Classify a single predicted span against gold spans.

    Returns one of:
        "correct"       - exact match with a gold span
        "boundary"      - appears in sentence, overlaps with a gold span but not exact
        "spurious"      - appears in sentence, no overlap with any gold span
        "paraphrase"    - not in sentence, but cosine-similar to a gold span
        "hallucination" - not in sentence, not similar to any gold span
    """
    pred_lower = pred_span.lower().strip()
    if not pred_lower:
        return "hallucination", None

    # exact match
    for g in gold_spans:
        if pred_lower == g.lower().strip():
            return "correct", g

    # is the span in the sentence?
    in_sentence = span_in_sentence(pred_lower, sentence)

    if in_sentence:
        # check overlap with any gold span
        for g in gold_spans:
            if spans_overlap(pred_lower, g):
                return "boundary", g
        return "spurious", None
    else:
        # not in sentence: check cosine similarity to golds
        if pred_lower in embeddings:
            for g in gold_spans:
                g_lower = g.lower().strip()
                if g_lower in embeddings:
                    sim = cosine_sim(embeddings[pred_lower], embeddings[g_lower])
                    if sim >= sim_threshold:
                        return "paraphrase", g
        return "hallucination", None


def classify_missed_span(gold_span, train_vocab):
    """
    Classify why a gold span was missed.

    Returns one of:
        "unseen" - span never appeared in training
        "seen"   - span appeared in training but was still missed
    """
    if gold_span.lower().strip() in train_vocab:
        return "seen"
    return "unseen"


def analyze_sentence(pred_triplets, gold_triplets, sentence, embeddings,
                     train_aspects, train_opinions, sim_threshold=0.7):
    """
    Analyze errors for a single sentence. Returns per-element error counts.
    """
    gold_aspects = [g.get("aspect", "") for g in gold_triplets]
    gold_opinions = [g.get("sentiment", "") for g in gold_triplets]
    gold_polarities = [g.get("polarity", "") for g in gold_triplets]

    pred_matched = [False] * len(pred_triplets)
    gold_matched = [False] * len(gold_triplets)

    # exact triplet matching
    for i, p in enumerate(pred_triplets):
        for j, g in enumerate(gold_triplets):
            if gold_matched[j]:
                continue
            if (p.get("aspect", "").lower() == g.get("aspect", "").lower() and
                p.get("sentiment", "").lower() == g.get("sentiment", "").lower() and
                p.get("polarity", "").lower() == g.get("polarity", "").lower()):
                pred_matched[i] = True
                gold_matched[j] = True
                break

    errors = {
        "n_correct": sum(pred_matched),
        "aspect_errors": [],
        "opinion_errors": [],
        "polarity_errors": [],
        "triplet_patterns": [],
        "missed_aspects": [],
        "missed_opinions": [],
    }

    # classify unmatched predictions (per-element)
    for i, p in enumerate(pred_triplets):
        if pred_matched[i]:
            continue

        pa = p.get("aspect", "")
        ps = p.get("sentiment", "")
        pp = p.get("polarity", "")

        aspect_class, aspect_gold = classify_span(pa, gold_aspects, sentence, embeddings, sim_threshold)
        opinion_class, opinion_gold = classify_span(ps, gold_opinions, sentence, embeddings, sim_threshold)

        if aspect_class != "correct":
            errors["aspect_errors"].append({
                "type": aspect_class,
                "pred": pa,
                "gold": aspect_gold,
            })

        if opinion_class != "correct":
            errors["opinion_errors"].append({
                "type": opinion_class,
                "pred": ps,
                "gold": opinion_gold,
            })

        # polarity: check if this polarity is consistent with any gold
        # that has the same (or overlapping) aspect
        polarity_correct = False
        gold_pol = None
        for g in gold_triplets:
            ga = g.get("aspect", "").lower()
            gp = g.get("polarity", "").lower()
            # exact aspect match
            if ga == pa.lower():
                gold_pol = g.get("polarity", "")
                if gp == pp.lower():
                    polarity_correct = True
                break
        # if no exact aspect match, try the nearest gold by opinion
        if gold_pol is None:
            for g in gold_triplets:
                gs = g.get("sentiment", "").lower()
                if gs == ps.lower():
                    gold_pol = g.get("polarity", "")
                    if g.get("polarity", "").lower() == pp.lower():
                        polarity_correct = True
                    break
        if not polarity_correct:
            errors["polarity_errors"].append({
                "pred": pp,
                "gold": gold_pol,
                "aspect": pa,
                "opinion": ps,
            })

        # triplet-level error pattern
        wrong = []
        if aspect_class != "correct":
            wrong.append("aspect")
        if opinion_class != "correct":
            wrong.append("opinion")
        if not polarity_correct:
            wrong.append("polarity")
        if not wrong:
            pattern = "duplicate"
        else:
            pattern = "+".join(wrong)
        errors["triplet_patterns"].append(pattern)

    # classify unmatched golds (missed extractions)
    for j, g in enumerate(gold_triplets):
        if gold_matched[j]:
            continue
        ga = g.get("aspect", "")
        gs = g.get("sentiment", "")
        errors["missed_aspects"].append({
            "type": classify_missed_span(ga, train_aspects),
            "span": ga,
        })
        errors["missed_opinions"].append({
            "type": classify_missed_span(gs, train_opinions),
            "span": gs,
        })

    return errors


def run_analysis(ckpt_path, test_file, train_file, output_format="natural-language",
                 sim_threshold=0.7, device="cuda"):
    print(f"Loading model from {ckpt_path}")
    model = load_model(ckpt_path, device)

    print(f"Loading test data: {test_file}")
    test_examples = load_aste_file(test_file)

    print(f"Loading train data: {train_file}")
    train_examples = load_aste_file(train_file)

    # collect training vocabulary
    train_aspects = set()
    train_opinions = set()
    for ex in train_examples:
        for ann in ex.get("annotations", []):
            if ann.get("aspect"):
                train_aspects.add(ann["aspect"].lower().strip())
            if ann.get("sentiment"):
                train_opinions.add(ann["sentiment"].lower().strip())

    print("Running inference...")
    results = run_inference(model, test_examples, output_format, device)

    print("Computing embeddings...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    all_spans = set()
    for r in results:
        for t in r["pred"]:
            if t.get("aspect"): all_spans.add(t["aspect"].lower().strip())
            if t.get("sentiment"): all_spans.add(t["sentiment"].lower().strip())
        for t in r["gold"]:
            if t.get("aspect"): all_spans.add(t["aspect"].lower().strip())
            if t.get("sentiment"): all_spans.add(t["sentiment"].lower().strip())

    span_list = list(all_spans)
    if span_list:
        emb_matrix = embed_model.encode(span_list, convert_to_numpy=True, show_progress_bar=False)
        embeddings = dict(zip(span_list, emb_matrix))
    else:
        embeddings = {}

    print("Classifying errors...")
    all_errors = []
    totals = {
        "n_sentences": len(results),
        "n_correct": 0,
        "n_pred": 0,
        "n_gold": 0,
        "aspect_errors": Counter(),
        "opinion_errors": Counter(),
        "polarity_errors": 0,
        "triplet_patterns": Counter(),
        "missed_aspects": Counter(),
        "missed_opinions": Counter(),
    }

    examples_by_type = {
        "aspect": {"boundary": [], "spurious": [], "paraphrase": [], "hallucination": []},
        "opinion": {"boundary": [], "spurious": [], "paraphrase": [], "hallucination": []},
    }

    for r in results:
        totals["n_pred"] += len(r["pred"])
        totals["n_gold"] += len(r["gold"])

        errs = analyze_sentence(
            r["pred"], r["gold"], r["sentence"], embeddings,
            train_aspects, train_opinions, sim_threshold,
        )
        totals["n_correct"] += errs["n_correct"]
        totals["polarity_errors"] += len(errs["polarity_errors"])

        for e in errs["aspect_errors"]:
            totals["aspect_errors"][e["type"]] += 1
            if len(examples_by_type["aspect"][e["type"]]) < 5:
                examples_by_type["aspect"][e["type"]].append({
                    "sentence": r["sentence"], **e
                })

        for e in errs["opinion_errors"]:
            totals["opinion_errors"][e["type"]] += 1
            if len(examples_by_type["opinion"][e["type"]]) < 5:
                examples_by_type["opinion"][e["type"]].append({
                    "sentence": r["sentence"], **e
                })

        for pattern in errs["triplet_patterns"]:
            totals["triplet_patterns"][pattern] += 1

        for e in errs["missed_aspects"]:
            totals["missed_aspects"][e["type"]] += 1
        for e in errs["missed_opinions"]:
            totals["missed_opinions"][e["type"]] += 1

    # summary
    total_aspect_errors = sum(totals["aspect_errors"].values())
    total_opinion_errors = sum(totals["opinion_errors"].values())
    total_triplet_errors = sum(totals["triplet_patterns"].values())
    total_missed_aspects = sum(totals["missed_aspects"].values())
    total_missed_opinions = sum(totals["missed_opinions"].values())

    def pct_dict(counter, total):
        return {k: {"count": v, "pct": round(100 * v / total, 1) if total > 0 else 0}
                for k, v in counter.most_common()}

    summary = {
        "n_sentences": totals["n_sentences"],
        "n_gold_triplets": totals["n_gold"],
        "n_pred_triplets": totals["n_pred"],
        "n_correct": totals["n_correct"],
        "triplet_f1": 2 * totals["n_correct"] / (totals["n_pred"] + totals["n_gold"] + 1e-9),
        "aspect_errors": pct_dict(totals["aspect_errors"], total_aspect_errors),
        "opinion_errors": pct_dict(totals["opinion_errors"], total_opinion_errors),
        "polarity_errors": totals["polarity_errors"],
        "triplet_patterns": pct_dict(totals["triplet_patterns"], total_triplet_errors),
        "missed_aspects": pct_dict(totals["missed_aspects"], total_missed_aspects),
        "missed_opinions": pct_dict(totals["missed_opinions"], total_missed_opinions),
        "examples": examples_by_type,
    }

    return summary


def print_summary(summary, out=sys.stdout):
    out.write(f"Sentences: {summary['n_sentences']}\n")
    out.write(f"Gold triplets: {summary['n_gold_triplets']}\n")
    out.write(f"Pred triplets: {summary['n_pred_triplets']}\n")
    out.write(f"Correct: {summary['n_correct']}\n")
    out.write(f"Triplet F1: {summary['triplet_f1']:.4f}\n\n")

    out.write("--- ASPECT ERRORS (unmatched predictions) ---\n")
    for typ, info in summary["aspect_errors"].items():
        out.write(f"  {typ}: {info['count']} ({info['pct']}%)\n")

    out.write("\n--- OPINION ERRORS (unmatched predictions) ---\n")
    for typ, info in summary["opinion_errors"].items():
        out.write(f"  {typ}: {info['count']} ({info['pct']}%)\n")

    out.write(f"\n--- POLARITY ERRORS ---\n")
    out.write(f"  {summary['polarity_errors']}\n")

    out.write(f"\n--- TRIPLET-LEVEL ERROR PATTERNS ---\n")
    for pattern, info in summary["triplet_patterns"].items():
        out.write(f"  {pattern}: {info['count']} ({info['pct']}%)\n")

    out.write("\n--- MISSED ASPECTS (unmatched golds) ---\n")
    for typ, info in summary["missed_aspects"].items():
        out.write(f"  {typ}: {info['count']} ({info['pct']}%)\n")

    out.write("\n--- MISSED OPINIONS (unmatched golds) ---\n")
    for typ, info in summary["missed_opinions"].items():
        out.write(f"  {typ}: {info['count']} ({info['pct']}%)\n")

    out.write("\n--- EXAMPLES ---\n")
    for element in ["aspect", "opinion"]:
        for typ, examples in summary["examples"][element].items():
            if not examples:
                continue
            out.write(f"\n  {element} {typ} (showing {len(examples)}):\n")
            for e in examples:
                out.write(f"    sentence: {e['sentence'][:80]}...\n")
                out.write(f"    pred: {e['pred']} | gold: {e['gold']}\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python error_analysis_v2.py <checkpoint> <test_file> <train_file> [output_json]")
        sys.exit(1)

    ckpt = sys.argv[1]
    test = sys.argv[2]
    train = sys.argv[3]
    out_file = sys.argv[4] if len(sys.argv) > 4 else None

    summary = run_analysis(ckpt, test, train)
    print_summary(summary)

    if out_file:
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_file}")
