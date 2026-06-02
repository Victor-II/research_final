"""
MvP-style multi-prompt voting at inference.

For each test example, generate predictions using multiple permutations of the
element ordering in the input. Aggregate results via majority voting with
threshold = k/2.

Supports both the original NL/structured format (task-order permutation in prompt)
and the new mvp-markers format (marker ordering permutation in input).

Inspired by: Gou et al. (2023) "MvP: Multi-view Prompting Improves Aspect
Sentiment Tuple Prediction"
"""

from collections import Counter
from itertools import permutations

from constants import Task, TASK_TO_KEY, CANONICAL_KEY_ORDER
from src.data.data import to_generative_format
from src.eval.eval import parse_output


# All 6 orderings of the ASTE triplet task
TRIPLET_ORDERINGS = list(permutations([Task.ASPECT, Task.SENTIMENT, Task.POLARITY]))


def generate_multiview_inputs(
    canonical_example: dict,
    output_format: str = "natural-language",
    language: str = "en",
    top_k: int = 6,
    seed: int = 42,
) -> list[dict]:
    """
    Generate top-k task-order variants of a canonical example for multi-view inference.

    Returns list of generative-format dicts, each with a different task ordering.
    For mvp-markers format, the marker order in the input changes.
    """
    import random
    rng = random.Random(seed)

    all_orderings = TRIPLET_ORDERINGS
    if top_k < len(all_orderings):
        selected = rng.sample(all_orderings, top_k)
    else:
        selected = all_orderings

    variants = []
    for ordering in selected:
        gen = to_generative_format(
            canonical_example,
            list(ordering),
            output_format=output_format,
            language=language,
            nl_fraction=1.0 if output_format == "natural-language" else 0.0,
            ignore_order=False,
        )
        variants.append(gen)
    return variants


def vote_predictions(
    predictions_per_view: list[list[dict]],
    threshold: int = None,
) -> list[dict]:
    """
    Aggregate predictions from multiple views via majority voting.

    Args:
        predictions_per_view: list of parsed prediction lists (one per view/ordering)
        threshold: minimum vote count to include a triplet.
                   Default (None) = ceil(num_views / 2) i.e. majority.

    Returns:
        list of voted triplet dicts
    """
    if not predictions_per_view:
        return []

    if threshold is None:
        threshold = (len(predictions_per_view) + 1) // 2  # ceil(k/2)

    counts = Counter()
    for preds in predictions_per_view:
        for triplet in preds:
            # normalize to canonical key order for deduplication
            normalized = tuple(sorted(triplet.items()))
            counts[normalized] += 1

    voted = []
    for triplet_tuple, count in counts.items():
        if count >= threshold:
            voted.append(dict(triplet_tuple))
    return voted
