"""
STAR-style data multiplication: for each training example, generate sub-task
variants (singles, pairs) as additional training examples.

Inspired by: Lai et al. (2025) "STAR: Stepwise Task Augmentation with Relation
Learning for Aspect Sentiment Quad Prediction"

Two modes:
1. star_multiply() — original sub-task decomposition (aspect-only, pair combos, etc.)
2. star_pairwise() — MvP/STAR pairwise relation examples with marker format.
   Input: sentence [AO][SP] etc.
   Output: [AO] pizza is delicious [SP] great
   Tagged with _level="pairwise" for balanced contribution loss.
"""

import random
from itertools import combinations, permutations

from constants import Task, TASK_TO_KEY
from src.data.data import to_generative_format


# All meaningful sub-task combinations for ASTE (aspect, sentiment, polarity)
ASTE_SUBTASKS = [
    # singles
    (Task.ASPECT,),
    (Task.SENTIMENT,),
    (Task.POLARITY,),
    # pairs
    (Task.ASPECT, Task.SENTIMENT),
    (Task.ASPECT, Task.POLARITY),
    (Task.SENTIMENT, Task.POLARITY),
]

# Pairwise relation markers (element pair → marker token)
PAIRWISE_MARKERS = {
    ("aspect", "sentiment"): "[AO]",
    ("aspect", "polarity"): "[AS]",
    ("sentiment", "polarity"): "[SP]",
    ("aspect", "category"): "[AC]",
}


def star_multiply(
    canonical_examples: list[dict],
    fraction: float = 1.0,
    subtasks: list[tuple] = None,
    output_format: str = "natural-language",
    language: str = "en",
    seed: int = 42,
) -> list[dict]:
    """
    Generate STAR-style sub-task training examples from canonical data.

    For each selected canonical example, generates one generative-format example
    per sub-task combination. These are returned as ready-to-use training examples
    (input/target/_keys/_format dicts).

    All returned examples are tagged with _level="pairwise" for balanced loss.
    """
    if subtasks is None:
        subtasks = ASTE_SUBTASKS

    rng = random.Random(seed)

    if fraction < 1.0:
        n_select = max(1, int(len(canonical_examples) * fraction))
        selected = rng.sample(range(len(canonical_examples)), k=n_select)
    else:
        selected = range(len(canonical_examples))

    aux_examples = []
    for idx in selected:
        ex = canonical_examples[idx]
        for task_combo in subtasks:
            # check that the example has the required fields
            keys_needed = [TASK_TO_KEY[t] for t in task_combo]
            has_all = all(
                any(ann.get(k) is not None for ann in ex["annotations"])
                for k in keys_needed
            )
            if not has_all:
                continue

            gen = to_generative_format(
                ex, list(task_combo),
                output_format=output_format,
                language=language,
                nl_fraction=1.0 if output_format == "natural-language" else 0.0,
            )
            # skip if target is empty (no valid annotations for this sub-task)
            if gen["target"].strip():
                gen["_level"] = "pairwise"
                aux_examples.append(gen)

    return aux_examples


def star_pairwise(
    canonical_examples: list[dict],
    tasks: list[Task] = None,
    output_format: str = "mvp-markers",
    language: str = "en",
    seed: int = 42,
) -> list[dict]:
    """
    Generate STAR pairwise relation examples in MvP marker format.

    For each canonical example, generates pairwise relation examples where:
    - Input: sentence [AO][SP] (pair markers appended to sentence)
    - Output: [AO] pizza is delicious [SP] great

    These teach the model to recognize element-pair relationships.
    All returned examples are tagged with _level="pairwise".
    """
    if tasks is None:
        tasks = [Task.ASPECT, Task.SENTIMENT, Task.POLARITY]

    keys = [TASK_TO_KEY[t] for t in tasks]
    # generate all pair combinations from the task keys
    pair_combos = list(combinations(keys, 2))

    KEY_TO_MARKER = {"aspect": "[A]", "sentiment": "[O]", "polarity": "[S]", "category": "[C]"}

    examples = []
    for ex in canonical_examples:
        sentence = ex["sentence"]
        annotations = ex["annotations"]

        for pair in pair_combos:
            k1, k2 = pair
            pair_marker = PAIRWISE_MARKERS.get(pair)
            if pair_marker is None:
                pair_marker = PAIRWISE_MARKERS.get((k2, k1))
            if pair_marker is None:
                pair_marker = f"[{k1[0].upper()}{k2[0].upper()}]"

            # collect valid annotations for this pair
            valid_anns = []
            for ann in annotations:
                v1, v2 = ann.get(k1), ann.get(k2)
                if v1 is not None and v2 is not None:
                    valid_anns.append({k1: v1, k2: v2})

            if not valid_anns:
                continue

            # build input: sentence + pair markers
            input_text = f"{sentence} {pair_marker}"

            # build output: [AO] aspect is sentiment [SP] polarity ...
            # Use the element markers to wrap values
            m1 = KEY_TO_MARKER.get(k1, f"[{k1[0].upper()}]")
            m2 = KEY_TO_MARKER.get(k2, f"[{k2[0].upper()}]")
            output_parts = []
            for ann in valid_anns:
                output_parts.append(f"{m1} {ann[k1]} {m2} {ann[k2]}")
            target = " [SSEP] ".join(output_parts)

            examples.append({
                "input": input_text,
                "target": target,
                "_keys": list(pair),
                "_format": output_format,
                "_level": "pairwise",
            })

    return examples
