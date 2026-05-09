"""Auto-suggest implicit aspect terms using Gemma via ollama.

Reads ASTE-format data, finds implicit aspects, asks LLM to infer them,
saves suggestions for review in the annotator tool.

Usage:
    python tools/suggest_aspects.py <data_file> [--model gemma2:27b] [--output suggestions.json]
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPT_TEMPLATE = """You are annotating product reviews for aspect-based sentiment analysis.

A review sentence contains opinions about specific aspects (features, parts, or attributes) of a product. Some aspects are explicitly mentioned, others are implied but not directly stated.

Your task: for each opinion marked with "???" as its aspect, determine what is actually being discussed.

Rules:
- Use only words or phrases from the sentence itself
- If the opinion refers to the product as a whole, answer "product"
- If you truly cannot determine the aspect, answer "UNKNOWN"
- Answer with ONLY the aspect terms, numbered to match

Example:
Sentence: "Have not had any issues thus far with this Cat6 cable. Just plugged it into my router and was up and running fine."
Opinions:
1. [Cat6 cable] is described as "no issues" (positive)
2. [???] is described as "up and running fine" (positive)
Aspects:
1. Cat6 cable
2. Cat6 cable

Now resolve:
Sentence: "{sentence}"
Opinions:
{opinions}
Aspects:
"""


def call_ollama(prompt: str, model: str, url: str, timeout: int = 120) -> str | None:
    import requests
    try:
        resp = requests.post(
            f"{url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def parse_aste_line(line: str) -> dict | None:
    parts = line.strip().split("####")
    if len(parts) < 2:
        return None
    text = parts[0]
    tokens = text.split()
    triplets = ast.literal_eval(parts[1])
    domain = parts[2].strip() if len(parts) > 2 else None
    annotations = []
    for a_idx, o_idx, pol in triplets:
        implicit_a = (a_idx == [-1] or a_idx == -1)
        implicit_o = (o_idx == [-1] or o_idx == -1)
        aspect = "IMPLICIT" if implicit_a else " ".join(tokens[j] for j in a_idx)
        opinion = "IMPLICIT" if implicit_o else " ".join(tokens[j] for j in o_idx)
        polarity = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}.get(pol, pol.lower())
        annotations.append({
            "aspect": aspect,
            "aspect_idx": None if implicit_a else list(a_idx),
            "opinion": opinion,
            "opinion_idx": None if implicit_o else list(o_idx),
            "polarity": polarity,
        })
    return {"sentence": text, "tokens": tokens, "annotations": annotations, "domain": domain}


def build_prompt(example: dict) -> tuple[str, list[int]]:
    """Build the LLM prompt and return (prompt, list of implicit annotation indices)."""
    opinions_lines = []
    implicit_indices = []
    for i, ann in enumerate(example["annotations"]):
        aspect_str = f"[{ann['aspect']}]" if ann["aspect"] != "IMPLICIT" else "[???]"
        opinion_str = ann["opinion"] if ann["opinion"] != "IMPLICIT" else "an unstated opinion"
        opinions_lines.append(f"{i+1}. {aspect_str} is described as \"{opinion_str}\" ({ann['polarity']})")
        if ann["aspect"] == "IMPLICIT":
            implicit_indices.append(i)

    prompt = PROMPT_TEMPLATE.format(
        sentence=example["sentence"],
        opinions="\n".join(opinions_lines),
    )
    return prompt, implicit_indices


def parse_response(response: str, n_opinions: int) -> list[str]:
    """Parse numbered aspect suggestions from LLM response."""
    suggestions = [None] * n_opinions
    for line in response.strip().split("\n"):
        line = line.strip()
        m = re.match(r"(\d+)\.\s*(.+)", line)
        if m:
            idx = int(m.group(1)) - 1
            aspect = m.group(2).strip().strip('"').strip("'")
            if 0 <= idx < n_opinions:
                suggestions[idx] = aspect
    return suggestions


def run_suggestions(
    data_file: str,
    output_file: str,
    model: str = "gemma2:27b",
    ollama_url: str = "http://localhost:11434",
    resume: bool = True,
):
    # Load data
    examples = []
    with open(data_file) as f:
        for line in f:
            if line.strip():
                ex = parse_aste_line(line)
                if ex:
                    examples.append(ex)

    # Filter to examples with implicit aspects
    candidates = [ex for ex in examples if any(a["aspect"] == "IMPLICIT" for a in ex["annotations"])]
    print(f"Total: {len(examples)} sentences, {len(candidates)} with implicit aspects")

    # Resume
    results = []
    done_sentences = set()
    output_path = Path(output_file)
    if resume and output_path.exists():
        with open(output_path) as f:
            results = json.load(f)
        done_sentences = {r["sentence"] for r in results}
        print(f"Resuming: {len(results)} done, {len(candidates) - len(done_sentences)} remaining")

    # Test ollama
    print(f"Testing ollama ({model})...")
    test = call_ollama("Say OK", model, ollama_url, timeout=300)
    if not test:
        print("ERROR: ollama not responding")
        return
    print(f"  OK: {test[:30]}")

    accepted = skipped = errors = 0
    for i, ex in enumerate(candidates):
        if ex["sentence"] in done_sentences:
            continue

        prompt, implicit_indices = build_prompt(ex)
        if not implicit_indices:
            continue

        response = call_ollama(prompt, model, ollama_url)
        if not response:
            errors += 1
            continue

        suggestions = parse_response(response, len(ex["annotations"]))

        # Build result with suggestions for implicit aspects
        ann_results = []
        for j, ann in enumerate(ex["annotations"]):
            entry = dict(ann)
            if j in implicit_indices:
                suggestion = suggestions[j] if suggestions[j] else "UNKNOWN"
                entry["suggested_aspect"] = suggestion
            ann_results.append(entry)

        results.append({
            "sentence": ex["sentence"],
            "domain": ex.get("domain"),
            "annotations": ann_results,
        })
        accepted += 1

        # Progress
        if accepted % 50 == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  [{accepted + len(done_sentences)}/{len(candidates)}] processed={accepted} errors={errors}")

    # Final save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone. {len(results)} suggestions saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Auto-suggest implicit aspects using LLM")
    parser.add_argument("data_file", help="ASTE-format data file")
    parser.add_argument("--model", default="gemma2:27b", help="ollama model name")
    parser.add_argument("--output", default=None, help="output JSON file")
    parser.add_argument("--url", default="http://localhost:11434", help="ollama URL")
    args = parser.parse_args()

    output = args.output or f"downloads/suggestions/{Path(args.data_file).stem}_suggestions.json"
    run_suggestions(args.data_file, output, model=args.model, ollama_url=args.url)


if __name__ == "__main__":
    main()
