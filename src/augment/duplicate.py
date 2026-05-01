import copy
import random


def duplicate_examples(
    examples: list[dict],
    fraction: float = 0.5,
    seed: int = 42,
) -> list[dict]:
    """Append exact copies of a fraction of examples (no transformation)."""
    rng = random.Random(seed)
    selected = set(rng.sample(range(len(examples)), k=int(len(examples) * fraction)))

    result = []
    for i, ex in enumerate(examples):
        result.append(ex)
        if i in selected:
            result.append(copy.deepcopy(ex))
    return result
