import copy
import random


def duplicate_examples(
    examples: list[dict],
    fraction: float = 0.5,
    seed: int = 42,
    filter_polarity: str = None,
) -> list[dict]:
    """Append exact copies of examples. If filter_polarity is set, only duplicate
    examples containing at least one annotation with that polarity value.
    fraction: how many times to duplicate the selected examples (1.0 = 1 extra copy, 3.0 = 3 extra copies)."""
    rng = random.Random(seed)

    if filter_polarity:
        # duplicate all matching examples N times
        n_copies = max(1, int(fraction))
        result = list(examples)
        for ex in examples:
            if any(a.get("polarity") == filter_polarity for a in ex.get("annotations", [])):
                for _ in range(n_copies):
                    result.append(copy.deepcopy(ex))
        return result

    # original behavior: randomly select fraction of examples, duplicate once
    selected = set(rng.sample(range(len(examples)), k=int(len(examples) * fraction)))
    result = []
    for i, ex in enumerate(examples):
        result.append(ex)
        if i in selected:
            result.append(copy.deepcopy(ex))
    return result
