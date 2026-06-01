"""
MvP-style multi-prompt voting at inference.

For each test example, generate predictions using all permutations of the
task ordering in the input prompt. Aggregate results via majority voting.

Inspired by: Gou et al. (2023) "MvP: Multi-view Prompting Improves Aspect
Sentiment Tuple Prediction"
"""

from collections import Counter
from itertools import permutations

from constants import Task
from src.data.data import to_generative_format
from src.eval.eval import parse_output


# All 6 orderings of the ASTE triplet task
TRIPLET_ORDERINGS = list(permutations([Task.ASPECT, Task.SENTIMENT, Task.POLARITY]))


def generate_multiview_inputs(
    canonical_example: dict,
    output_format: str = "natural-language",
    language: str = "en",
) -> list[dict]:
    """
    Generate all 6 task-order variants of a canonical example for multi-view inference.

    Returns list of generative-format dicts, each with a different task ordering
    in the input prompt.
    """
    variants = []
    for ordering in TRIPLET_ORDERINGS:
        gen = to_generative_format(
            canonical_example,
            list(ordering),
            output_format=output_format,
            language=language,
            nl_fraction=1.0 if output_format == "natural-language" else 0.0,
            ignore_order=False,  # use the actual ordering for template selection
        )
        variants.append(gen)
    return variants


def vote_predictions(
    predictions_per_view: list[list[dict]],
    threshold: int = 1,
) -> list[dict]:
    """
    Aggregate predictions from multiple views via majority voting.

    Args:
        predictions_per_view: list of parsed prediction lists (one per view/ordering)
        threshold: minimum vote count to include a triplet (default 1 = union)

    Returns:
        list of voted triplet dicts
    """
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
