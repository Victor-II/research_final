"""
Generate opinion synonym augmentations via LLM.

For each training example, prompts an LLM to rewrite the sentence with a
different opinion expression while maintaining the same polarity. The result
is saved as a JSON file that can be loaded during training as augmentation.

Usage:
    python tools/generate_opinion_synonyms.py \
        --train-file downloads/ABSADatasets/datasets/aste_datasets/400.SemEval/402.Restaurant14/train.txt \
        --model gemma2:27b \
        --output downloads/opinion_synonyms_rest14.json \
        --n-variants 2
"""

import argparse
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.data import load_aste_file


OLLAMA_URL = "http://localhost:11434"


def call_ollama(messages: list[dict], model: str, timeout: int = 120) -> str | None:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": model, "messages": messages, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        text = resp.json()["message"]["content"].strip()
        return text if text else None
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


SYSTEM_PROMPT = """You are an expert in aspect-based sentiment analysis. Given a sentence and its annotations, rewrite the sentence by replacing the opinion terms with different words or phrases that express the SAME polarity.

Rules:
- Keep the aspect terms EXACTLY as they are (do not change, move, or remove them)
- Replace ONLY the opinion terms with synonyms or alternative expressions
- The new opinion must express the same polarity (positive/negative/neutral) as the original
- Keep the sentence natural and grammatically correct
- Do NOT add new aspects or new opinions
- Do NOT change the sentence structure significantly
- Do NOT use markdown formatting (no bold, no asterisks)"""

FEW_SHOT_EXAMPLES = [
    {
        "user": """Sentence: "The food is uniformly exceptional , with a very capable kitchen ."
Annotations:
  - Aspect: "food" | Opinion: "exceptional" | Polarity: positive
  - Aspect: "kitchen" | Opinion: "capable" | Polarity: positive

Generate exactly 2 rewritten versions.""",
        "assistant": """1. Sentence: The food is uniformly outstanding , with a very skilled kitchen .
   Opinions: exceptional -> outstanding, capable -> skilled
2. Sentence: The food is consistently superb , with a very talented kitchen .
   Opinions: exceptional -> superb, capable -> talented"""
    },
    {
        "user": """Sentence: "The price is reasonable although the service is poor ."
Annotations:
  - Aspect: "price" | Opinion: "reasonable" | Polarity: positive
  - Aspect: "service" | Opinion: "poor" | Polarity: negative

Generate exactly 2 rewritten versions.""",
        "assistant": """1. Sentence: The price is affordable although the service is terrible .
   Opinions: reasonable -> affordable, poor -> terrible
2. Sentence: The price is fair although the service is awful .
   Opinions: reasonable -> fair, poor -> awful"""
    },
    {
        "user": """Sentence: "The environment is romantic , but the food is horrible , the service is pathetic ."
Annotations:
  - Aspect: "environment" | Opinion: "romantic" | Polarity: positive
  - Aspect: "food" | Opinion: "horrible" | Polarity: negative
  - Aspect: "service" | Opinion: "pathetic" | Polarity: negative

Generate exactly 2 rewritten versions.""",
        "assistant": """1. Sentence: The environment is intimate , but the food is disgusting , the service is dreadful .
   Opinions: romantic -> intimate, horrible -> disgusting, pathetic -> dreadful
2. Sentence: The environment is cozy , but the food is inedible , the service is appalling .
   Opinions: romantic -> cozy, horrible -> inedible, pathetic -> appalling"""
    },
    {
        "user": """Sentence: "While the ambiance and atmosphere were great , the food and service could have been a lot better ."
Annotations:
  - Aspect: "ambiance" | Opinion: "great" | Polarity: positive
  - Aspect: "atmosphere" | Opinion: "great" | Polarity: positive
  - Aspect: "food" | Opinion: "could have been a lot better" | Polarity: negative
  - Aspect: "service" | Opinion: "could have been a lot better" | Polarity: negative

Generate exactly 2 rewritten versions.""",
        "assistant": """1. Sentence: While the ambiance and atmosphere were wonderful , the food and service left much to be desired .
   Opinions: great -> wonderful, great -> wonderful, could have been a lot better -> left much to be desired, could have been a lot better -> left much to be desired
2. Sentence: While the ambiance and atmosphere were fantastic , the food and service were quite disappointing .
   Opinions: great -> fantastic, great -> fantastic, could have been a lot better -> were quite disappointing, could have been a lot better -> were quite disappointing"""
    },
    {
        "user": """Sentence: "Food was average and creme brulee was awful - the sugar was charred , not caramelized and smelled of kerosene ."
Annotations:
  - Aspect: "Food" | Opinion: "average" | Polarity: neutral
  - Aspect: "creme brulee" | Opinion: "awful" | Polarity: negative
  - Aspect: "sugar" | Opinion: "charred" | Polarity: negative

Generate exactly 2 rewritten versions.""",
        "assistant": """1. Sentence: Food was mediocre and creme brulee was terrible - the sugar was burnt , not caramelized and smelled of kerosene .
   Opinions: average -> mediocre, awful -> terrible, charred -> burnt
2. Sentence: Food was unremarkable and creme brulee was dreadful - the sugar was scorched , not caramelized and smelled of kerosene .
   Opinions: average -> unremarkable, awful -> dreadful, charred -> scorched"""
    },
]


def build_messages(sentence: str, annotations: list[dict], n_variants: int = 2) -> list[dict]:
    """Build chat messages with system prompt, few-shot examples, and the actual query."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["user"]})
        messages.append({"role": "assistant", "content": ex["assistant"]})

    # Format annotations for the actual query
    ann_lines = []
    for ann in annotations:
        aspect = ann.get("aspect", "IMPLICIT")
        opinion = ann.get("sentiment", "?")
        polarity = ann.get("polarity", "?")
        ann_lines.append(f"  - Aspect: \"{aspect}\" | Opinion: \"{opinion}\" | Polarity: {polarity}")

    annotations_str = "\n".join(ann_lines)

    user_msg = f"""Sentence: "{sentence}"

Annotations:
{annotations_str}

Generate exactly {n_variants} rewritten versions."""

    messages.append({"role": "user", "content": user_msg})
    return messages


def parse_variants(response: str, n_expected: int, annotations: list[dict]) -> list[dict]:
    """Parse numbered variants with opinion mappings from LLM response.
    
    Returns list of dicts with 'sentence' and 'opinion_map' (old -> new).
    """
    import re
    results = []
    
    # split by numbered entries
    blocks = re.split(r'\n?\d+\.', response)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        # extract sentence
        sent_match = re.search(r'[Ss]entence:\s*(.+?)(?:\n|$)', block)
        if not sent_match:
            continue
        sentence = sent_match.group(1).strip().strip('"').strip("'")
        
        # extract opinion mappings
        opinion_map = {}
        opinions_match = re.search(r'[Oo]pinions?:\s*(.+?)(?:\n|$)', block)
        if opinions_match:
            mappings_str = opinions_match.group(1).strip()
            # parse "old -> new, old -> new"
            for mapping in mappings_str.split(","):
                mapping = mapping.strip()
                if "->" in mapping:
                    parts = mapping.split("->")
                    if len(parts) == 2:
                        old = parts[0].strip().strip('"').strip("'")
                        new = parts[1].strip().strip('"').strip("'")
                        if old and new:
                            opinion_map[old.lower()] = new
        
        if sentence and opinion_map:
            results.append({"sentence": sentence, "opinion_map": opinion_map})
    
    return results[:n_expected]


def find_opinion_in_sentence(sentence: str, opinion: str) -> tuple[int, int] | None:
    """Find the start and end character positions of the opinion in the sentence."""
    lower_sent = sentence.lower()
    lower_op = opinion.lower()
    idx = lower_sent.find(lower_op)
    if idx >= 0:
        return (idx, idx + len(opinion))
    return None


def extract_new_opinion(original_sentence: str, new_sentence: str, original_opinion: str) -> str | None:
    """
    Given original and rewritten sentence, figure out what replaced the original opinion.
    Uses a simple diff-based approach: find the longest common prefix/suffix and
    extract what's in between in the new sentence.
    """
    # Normalize whitespace for comparison
    orig_lower = original_sentence.lower().split()
    new_lower = new_sentence.lower().split()
    opinion_tokens = original_opinion.lower().split()

    # find opinion token positions in original
    op_start = None
    for i in range(len(orig_lower) - len(opinion_tokens) + 1):
        if orig_lower[i:i+len(opinion_tokens)] == opinion_tokens:
            op_start = i
            break

    if op_start is None:
        return None

    op_end = op_start + len(opinion_tokens)

    # find how many tokens before the opinion match in the new sentence
    # (prefix matching)
    prefix_len = 0
    for i in range(min(op_start, len(new_lower))):
        if i < len(orig_lower) and new_lower[i] == orig_lower[i]:
            prefix_len = i + 1
        else:
            break

    # find how many tokens after the opinion match from the end
    suffix_len = 0
    orig_suffix = orig_lower[op_end:]
    new_suffix_start = len(new_lower)
    for i in range(1, min(len(orig_suffix), len(new_lower)) + 1):
        if orig_lower[-i] == new_lower[-i]:
            suffix_len = i
        else:
            break

    # extract the middle part from the new sentence
    new_end = len(new_lower) - suffix_len if suffix_len > 0 else len(new_lower)
    new_start = prefix_len

    if new_start >= new_end:
        return None

    # reconstruct from original (non-lowered) new sentence tokens
    new_tokens = new_sentence.split()
    if new_start >= len(new_tokens) or new_end > len(new_tokens):
        return None

    new_opinion = " ".join(new_tokens[new_start:new_end])
    # strip punctuation from edges
    new_opinion = new_opinion.strip(".,!?;:'\"")
    return new_opinion if new_opinion else None


def process_example(example: dict, model: str, n_variants: int) -> list[dict]:
    """Process a single example: prompt LLM, parse results, validate."""
    sentence = example["sentence"]
    annotations = example["annotations"]

    # skip examples with no valid opinions
    valid_anns = [a for a in annotations if a.get("sentiment") and a["sentiment"] not in ("IMPLICIT", "NULL")]
    if not valid_anns:
        return []

    prompt = build_messages(sentence, annotations, n_variants)
    response = call_ollama(prompt, model)

    if not response:
        return []

    variants = parse_variants(response, n_variants, annotations)
    results = []

    for variant in variants:
        new_sentence = variant["sentence"]
        opinion_map = variant["opinion_map"]

        # validate: check that aspects are preserved
        aspects_preserved = True
        for ann in valid_anns:
            aspect = ann.get("aspect", "")
            if aspect and aspect != "IMPLICIT" and aspect.lower() not in new_sentence.lower():
                aspects_preserved = False
                break

        if not aspects_preserved:
            continue

        # build new annotations using the opinion map
        new_annotations = []
        any_changed = False
        for ann in annotations:
            new_ann = dict(ann)
            opinion = ann.get("sentiment")
            if opinion and opinion not in ("IMPLICIT", "NULL"):
                mapped = opinion_map.get(opinion.lower())
                if mapped and mapped.lower() != opinion.lower():
                    new_ann["sentiment"] = mapped
                    new_ann["sentiment_idx"] = None
                    any_changed = True
            new_annotations.append(new_ann)

        if not any_changed:
            continue

        # verify new opinions appear in the new sentence
        opinions_in_sentence = True
        for ann in new_annotations:
            op = ann.get("sentiment")
            if op and op not in ("IMPLICIT", "NULL"):
                if op.lower() not in new_sentence.lower():
                    opinions_in_sentence = False
                    break

        if not opinions_in_sentence:
            continue

        results.append({
            "sentence": new_sentence,
            "tokens": new_sentence.split(),
            "annotations": new_annotations,
            "original_sentence": sentence,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate opinion synonym augmentations via LLM")
    parser.add_argument("--train-file", required=True, nargs="+", help="Training data files")
    parser.add_argument("--model", default="gemma2:27b", help="Ollama model name")
    parser.add_argument("--output", default="downloads/opinion_synonyms.json", help="Output JSON file")
    parser.add_argument("--n-variants", type=int, default=2, help="Number of variants per example")
    parser.add_argument("--max-examples", type=int, default=None, help="Limit number of examples to process")
    args = parser.parse_args()

    # Load data
    examples = []
    for f in args.train_file:
        examples.extend(load_aste_file(f))
    print(f"Loaded {len(examples)} training examples")

    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"Limited to {len(examples)} examples")

    # Check ollama connection
    print(f"Testing ollama connection (model={args.model})...")
    test = call_ollama([{"role": "user", "content": "Say OK"}], args.model)
    if not test:
        print("ERROR: ollama not responding. Start it and try again.")
        return
    print(f"  OK: \"{test[:50]}\"")

    # Resume from existing output
    output_path = Path(args.output)
    existing_results = []
    done_sentences = set()
    if output_path.exists():
        existing_results = json.loads(output_path.read_text())
        done_sentences = set(r["original_sentence"] for r in existing_results)
        print(f"Resuming: {len(existing_results)} already generated, {len(done_sentences)} sentences done")

    results = list(existing_results)
    total = len(examples)
    processed = 0
    generated = 0

    for i, ex in enumerate(examples):
        if ex["sentence"] in done_sentences:
            continue

        processed += 1
        new_variants = process_example(ex, args.model, args.n_variants)
        results.extend(new_variants)
        generated += len(new_variants)

        if processed % 10 == 0:
            print(f"  [{i+1}/{total}] Processed: {processed}, Generated: {generated} variants")
            # save incrementally
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone! Generated {generated} new variants from {processed} examples.")
    print(f"Total in output: {len(results)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
