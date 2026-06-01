"""
STAR-style data multiplication: for each training example, generate sub-task
variants (singles, pairs) as additional training examples.

Inspired by: Lai et al. (2025) "STAR: Stepwise Task Augmentation with Relation
Learning for Aspect Sentiment Quad Prediction"

The key idea: instead of only training on the full triplet task, also train on
decomposed sub-tasks (aspect-only, sentiment-only, polarity-only, aspect+sentiment,
aspect+polarity, sentiment+polarity). This teaches the model to capture
dependencies between elements at different granularities.
"""

import random
from itertools import combinations

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

    Args:
        canonical_examples: list of canonical dicts
        fraction: proportion of examples to generate sub-tasks from (1.0 = all)
        subtasks: list of task tuples to generate (default: all ASTE sub-tasks)
        output_format: "natural-language" or "structured"
        language: output language
        seed: random seed for selection

    Returns:
        list of generative-format dicts (to be appended to training examples)
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
                aux_examples.append(gen)

    return aux_examples
