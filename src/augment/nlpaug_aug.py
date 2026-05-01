import copy
import random
import nlpaug.augmenter.word as naw

from src.data.utils import find_span_indices


_AUGMENTERS = {
    "synonym": lambda cfg: naw.SynonymAug(aug_src="wordnet", aug_p=cfg.get("aug_p", 0.1)),
    "random_swap": lambda cfg: naw.RandomWordAug(action="swap", aug_p=cfg.get("aug_p", 0.1)),
    "random_delete": lambda cfg: naw.RandomWordAug(action="delete", aug_p=cfg.get("aug_p", 0.1)),
    "random_crop": lambda cfg: naw.RandomWordAug(action="crop", aug_p=cfg.get("aug_p", 0.1)),
    "contextual": lambda cfg: naw.ContextualWordEmbsAug(
        model_path=cfg.get("model", "bert-base-uncased"), action="substitute", aug_p=cfg.get("aug_p", 0.1),
    ),
    "spelling": lambda cfg: naw.SpellingAug(aug_p=cfg.get("aug_p", 0.1)),
    "antonym": lambda cfg: naw.AntonymAug(aug_p=cfg.get("aug_p", 0.1)),
    "split": lambda cfg: naw.SplitAug(),
}

_ASPECT_PH = "XASPECTPLACEHOLDER"
_SENT_PH = "XSENTPLACEHOLDER"


def _collect_spans(ex: dict) -> list[tuple[str, list[int], str]]:
    """Collect all unique (text, indices, type) spans from annotations."""
    seen = set()
    spans = []
    for ann in ex["annotations"]:
        for field, idx_field, tag in [("aspect", "aspect_idx", "aspect"), ("sentiment", "sentiment_idx", "sent")]:
            text = ann.get(field)
            idx = ann.get(idx_field)
            if text and idx:
                key = (tuple(idx), tag)
                if key not in seen:
                    seen.add(key)
                    spans.append((text, idx, tag))
    # sort by start index descending so replacements don't shift earlier indices
    spans.sort(key=lambda s: s[1][0], reverse=True)
    return spans


def _protect_and_augment(ex: dict, augmenter) -> dict | None:
    """Replace aspect/sentiment spans with placeholders, augment, then restore."""
    ex = copy.deepcopy(ex)
    tokens = ex["tokens"][:]
    spans = _collect_spans(ex)

    # map: placeholder -> (original_text, span_len, type)
    ph_map = {}
    for i, (text, idx, tag) in enumerate(spans):
        ph = f"{_ASPECT_PH if tag == 'aspect' else _SENT_PH}{i}"
        ph_map[ph] = (text, len(idx), tag)
        start, end = idx[0], idx[-1]
        tokens[start:end + 1] = [ph]

    masked_sentence = " ".join(tokens)
    augmented = augmenter.augment(masked_sentence)
    if isinstance(augmented, list):
        augmented = augmented[0]
    if not augmented:
        return None

    new_tokens = augmented.split()

    # restore placeholders -> original text
    # nlpaug may attach punctuation to placeholders (e.g. "XASPECTPLACEHOLDER0,")
    restored = []
    for t in new_tokens:
        matched = False
        for ph, (orig_text, _, _) in ph_map.items():
            if ph in t:
                prefix = t[:t.index(ph)]
                suffix = t[t.index(ph) + len(ph):]
                if prefix:
                    restored.append(prefix)
                restored.extend(orig_text.split())
                if suffix:
                    restored.append(suffix)
                matched = True
                break
        if not matched:
            restored.append(t)

    ex["tokens"] = restored
    ex["sentence"] = " ".join(restored)

    # re-locate all spans
    for ann in ex["annotations"]:
        if ann.get("aspect") is not None:
            idx = find_span_indices(restored, ann["aspect"])
            if idx is None:
                return None
            ann["aspect_idx"] = idx
        if ann.get("sentiment") is not None:
            idx = find_span_indices(restored, ann["sentiment"])
            if idx is None:
                return None
            ann["sentiment_idx"] = idx

    return ex


def _full_augment(ex: dict, augmenter) -> dict | None:
    """Augment the full sentence including aspects, then try to recover spans."""
    ex = copy.deepcopy(ex)
    augmented = augmenter.augment(ex["sentence"])
    if isinstance(augmented, list):
        augmented = augmented[0]
    if not augmented or augmented == ex["sentence"]:
        return None

    new_tokens = augmented.split()
    ex["tokens"] = new_tokens
    ex["sentence"] = augmented

    for ann in ex["annotations"]:
        if ann.get("aspect") is not None:
            idx = find_span_indices(new_tokens, ann["aspect"])
            if idx is None:
                return None
            ann["aspect_idx"] = idx
        if ann.get("sentiment") is not None:
            idx = find_span_indices(new_tokens, ann["sentiment"])
            if idx is None:
                return None
            ann["sentiment_idx"] = idx

    return ex


def nlpaug_augment(
    examples: list[dict],
    method: str = "synonym",
    fraction: float = 0.5,
    replace: bool = False,
    protect_aspects: bool = True,
    seed: int = 42,
    **kwargs,
) -> list[dict]:
    if method not in _AUGMENTERS:
        raise ValueError(f"Unknown nlpaug method: {method}. Available: {list(_AUGMENTERS.keys())}")

    augmenter = _AUGMENTERS[method](kwargs)
    rng = random.Random(seed)
    selected = set(rng.sample(range(len(examples)), k=int(len(examples) * fraction)))

    aug_fn = _protect_and_augment if protect_aspects else _full_augment
    result = []
    discarded = 0
    for i, ex in enumerate(examples):
        if i not in selected:
            result.append(ex)
            continue

        augmented = aug_fn(ex, augmenter)
        if augmented is None:
            discarded += 1
            result.append(ex)
            continue

        if replace:
            result.append(augmented)
        else:
            result.append(ex)
            result.append(augmented)

    if discarded > 0:
        print(f"[nlpaug] {method}: {discarded}/{len(selected)} examples discarded (aspect/sentiment not recoverable)")

    return result
