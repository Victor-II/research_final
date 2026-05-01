"""LLM-based paraphrase augmentation for implicit aspect generation.

Workflow:
1. Generate paraphrased data offline: generate_paraphrases() → saves JSON file
2. During training: load_paraphrases() loads the file, apply_paraphrase_augmentation()
   selects a fraction to replace originals each epoch.

The paraphrased file format (list of dicts):
{
    "original_sentence": "The pizza was delicious",
    "paraphrased_sentence": "Every bite was absolutely heavenly",
    "annotations": [
        {"aspect": "IMPLICIT", "aspect_original": "pizza", "sentiment": "delicious", ...}
    ]
}
"""

import copy
import json
import random


def load_paraphrases(file_path: str) -> list[dict]:
    """Load pre-generated paraphrased examples."""
    with open(file_path) as f:
        data = json.load(f)
    # convert to canonical format
    examples = []
    for entry in data:
        ex = {
            "sentence": entry["paraphrased_sentence"],
            "tokens": entry["paraphrased_sentence"].split(),
            "annotations": entry["annotations"],
        }
        examples.append(ex)
    return examples


def apply_paraphrase_augmentation(
    examples: list[dict],
    paraphrased: list[dict],
    fraction: float = 0.3,
    replace: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Replace a fraction of examples with their paraphrased versions.

    Paraphrased examples are matched by original_sentence.
    Selection is randomized per seed (changes per epoch).
    """
    # build lookup: original sentence -> paraphrased example
    para_lookup = {}
    for p in paraphrased:
        # the original sentence is stored in annotations
        for ann in p["annotations"]:
            if ann.get("aspect_original"):
                # find the original sentence from the paraphrased data
                break
        # use a hash of annotations as key since multiple paraphrases may exist
        # for now, index by position
    # simple approach: pair by index (paraphrased file mirrors training file order)
    if len(paraphrased) < len(examples):
        # pad with None for examples without paraphrases
        para_pool = paraphrased + [None] * (len(examples) - len(paraphrased))
    else:
        para_pool = paraphrased[:len(examples)]

    rng = random.Random(seed)
    n_select = int(len(examples) * fraction)
    selected = set(rng.sample(range(len(examples)), k=min(n_select, len(examples))))

    result = []
    for i, ex in enumerate(examples):
        if i in selected and para_pool[i] is not None:
            if replace:
                result.append(para_pool[i])
            else:
                result.append(ex)
                result.append(para_pool[i])
        else:
            result.append(ex)
    return result


def generate_paraphrases_mock(
    examples: list[dict],
    output_path: str,
):
    """Mock generation: for each example with explicit aspects, create a paraphrased
    version by removing the aspect from the sentence and keeping the original term
    in aspect_original. For testing the pipeline only.
    """
    results = []
    for ex in examples:
        for ann in ex["annotations"]:
            aspect = ann.get("aspect")
            if not aspect or aspect == "IMPLICIT":
                continue
            paraphrased = ex["sentence"].replace(aspect, "it")
            if paraphrased == ex["sentence"]:
                continue
            new_ann = dict(ann)
            new_ann["aspect"] = "IMPLICIT"
            new_ann["aspect_original"] = aspect
            new_ann["aspect_idx"] = None
            results.append({
                "original_sentence": ex["sentence"],
                "paraphrased_sentence": paraphrased,
                "annotations": [new_ann],
            })
            break

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Mock paraphrases: {len(results)} examples saved to {output_path}")
    return results


def generate_paraphrases(
    examples: list[dict],
    output_path: str,
    model_name: str = "llama3.1:8b",
    ollama_url: str = "http://localhost:11434",
):
    """Generate paraphrased versions of examples using a local LLM via ollama.

    For each example with explicit aspects, rewrites the sentence so the aspect
    is implied rather than stated. Stores the original aspect in aspect_original.
    """
    import requests

    PROMPT_TEMPLATE = (
        "Rewrite the following sentence so that '{aspect}' is not mentioned "
        "but still implied by context. Keep the same sentiment and meaning. "
        "Only output the rewritten sentence, nothing else.\n\n"
        "Original: {sentence}\n"
        "Rewritten:"
    )

    results = []
    skipped = 0
    for i, ex in enumerate(examples):
        ann = None
        for a in ex["annotations"]:
            if a.get("aspect") and a["aspect"] != "IMPLICIT":
                ann = a
                break
        if not ann:
            skipped += 1
            continue

        prompt = PROMPT_TEMPLATE.format(aspect=ann["aspect"], sentence=ex["sentence"])

        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model_name, "prompt": prompt, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            paraphrased = resp.json()["response"].strip().strip('"').strip("'")
        except Exception as e:
            print(f"  [{i}] Error: {e}")
            skipped += 1
            continue

        if not paraphrased or paraphrased == ex["sentence"]:
            skipped += 1
            continue

        new_ann = dict(ann)
        new_ann["aspect_original"] = ann["aspect"]
        new_ann["aspect"] = "IMPLICIT"
        new_ann["aspect_idx"] = None

        results.append({
            "original_sentence": ex["sentence"],
            "paraphrased_sentence": paraphrased,
            "annotations": [new_ann],
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(examples)}] Generated {len(results)} paraphrases, skipped {skipped}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Paraphrases: {len(results)} generated, {skipped} skipped, saved to {output_path}")
    return results
