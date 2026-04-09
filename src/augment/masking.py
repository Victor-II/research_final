import random
import copy


def mask_aspects(
    examples: list[dict],
    fraction: float = 0.5,
    replace: bool = False,
    seed: int = 42,
) -> list[dict]:
    """
    Augment canonical examples by replacing aspect spans with <extra_id_X> sentinels.

    Args:
        examples: list of canonical dicts (must have 'tokens' and 'annotations' with 'aspect_idx')
        fraction: proportion of examples to augment
        replace:  if True, replace originals; if False, append masked copies to originals
        seed:     random seed

    Returns:
        augmented list of canonical dicts
    """
    rng = random.Random(seed)
    selected = set(rng.sample(range(len(examples)), k=int(len(examples) * fraction)))

    result = []
    for i, ex in enumerate(examples):
        if i not in selected:
            result.append(ex)
            continue

        masked = _mask_example(ex)
        if replace:
            result.append(masked)
        else:
            result.append(ex)
            result.append(masked)

    return result


def _mask_example(ex: dict) -> dict:
    ex = copy.deepcopy(ex)
    tokens = ex["tokens"][:]

    # collect all aspect spans, assign sentinel index per unique span
    sentinel_idx = 0
    masked_spans: dict[tuple, str] = {}

    for ann in ex["annotations"]:
        if ann.get("aspect_idx") is None:
            continue
        span_key = tuple(ann["aspect_idx"])
        if span_key not in masked_spans:
            masked_spans[span_key] = f"<extra_id_{sentinel_idx}>"
            sentinel_idx += 1

    if not masked_spans:
        return ex

    # build new token list, replacing each aspect span with its sentinel
    # process spans in reverse order to preserve indices
    sorted_spans = sorted(masked_spans.keys(), key=lambda s: s[0], reverse=True)
    for span in sorted_spans:
        sentinel = masked_spans[span]
        start, end = span[0], span[-1]
        tokens[start:end + 1] = [sentinel]

    ex["sentence"] = " ".join(tokens)
    ex["tokens"] = tokens

    # update aspect strings and indices in annotations
    for ann in ex["annotations"]:
        if ann.get("aspect_idx") is None:
            continue
        span_key = tuple(ann["aspect_idx"])
        ann["aspect"] = masked_spans[span_key]
        ann["aspect_idx"] = None  # indices no longer valid after masking

    return ex
