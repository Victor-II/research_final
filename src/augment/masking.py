import random
import copy


def mask_aspects(
    examples: list[dict],
    fraction: float = 0.5,
    replace: bool = False,
    mask_target: bool = False,
    seed: int = 42,
) -> list[dict]:
    """
    Augment canonical examples by replacing aspect spans with <extra_id_X> sentinels.

    Args:
        examples: list of canonical dicts (must have 'tokens' and 'annotations' with 'aspect_idx')
        fraction: proportion of examples to augment
        replace:  if True, replace originals; if False, append masked copies to originals
        mask_target: if True, aspect text in annotations becomes the sentinel (original behaviour);
                     if False, keep original aspect text so the model learns to infer it from context
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

        masked = _mask_example(ex, mask_target=mask_target)
        if replace:
            result.append(masked)
        else:
            result.append(ex)
            result.append(masked)

    return result


def _mask_example(ex: dict, mask_target: bool = False) -> dict:
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

    # update annotations
    for ann in ex["annotations"]:
        if ann.get("aspect_idx") is None:
            continue
        span_key = tuple(ann["aspect_idx"])
        if mask_target:
            ann["aspect"] = masked_spans[span_key]
        ann["aspect_idx"] = None  # indices no longer valid after masking

    return ex



def mask_opinions(
    examples: list[dict],
    fraction: float = 0.5,
    replace: bool = False,
    mask_target: bool = False,
    seed: int = 42,
) -> list[dict]:
    """
    Augment canonical examples by replacing opinion/sentiment spans with <extra_id_X> sentinels.

    Forces the model to infer opinion expressions from context rather than copying from input.

    Args:
        examples: list of canonical dicts (must have 'tokens' and 'annotations' with 'sentiment_idx')
        fraction: proportion of examples to augment
        replace:  if True, replace originals; if False, append masked copies
        mask_target: if True, opinion text in annotations becomes the sentinel;
                     if False, keep original opinion text (model must predict it without seeing it)
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

        masked = _mask_opinion_example(ex, mask_target=mask_target)
        if replace:
            result.append(masked)
        else:
            result.append(ex)
            result.append(masked)

    return result


def _mask_opinion_example(ex: dict, mask_target: bool = False) -> dict:
    ex = copy.deepcopy(ex)
    tokens = ex["tokens"][:]

    sentinel_idx = 0
    masked_spans: dict[tuple, str] = {}

    for ann in ex["annotations"]:
        if ann.get("sentiment_idx") is None:
            continue
        span_key = tuple(ann["sentiment_idx"])
        if span_key not in masked_spans:
            masked_spans[span_key] = f"<extra_id_{sentinel_idx}>"
            sentinel_idx += 1

    if not masked_spans:
        return ex

    # collect all token positions to mask, map each to its sentinel
    pos_to_sentinel: dict[int, str] = {}
    for span_key, sentinel in masked_spans.items():
        for pos in span_key:
            pos_to_sentinel[pos] = sentinel

    # build new token list: replace first token of each sentinel group with
    # the sentinel, skip subsequent tokens belonging to the same group
    new_tokens = []
    last_sentinel = None
    for i, tok in enumerate(tokens):
        if i in pos_to_sentinel:
            s = pos_to_sentinel[i]
            if s != last_sentinel:
                new_tokens.append(s)
                last_sentinel = s
            # else: skip (consecutive tokens of same span)
        else:
            new_tokens.append(tok)
            last_sentinel = None

    ex["sentence"] = " ".join(new_tokens)
    ex["tokens"] = new_tokens

    for ann in ex["annotations"]:
        if ann.get("sentiment_idx") is None:
            continue
        span_key = tuple(ann["sentiment_idx"])
        if mask_target:
            ann["sentiment"] = masked_spans[span_key]
        ann["sentiment_idx"] = None

    return ex


def opinion_prediction_aux(
    examples: list[dict],
    fraction: float = 0.25,
    seed: int = 42,
) -> list[dict]:
    """
    Generate auxiliary opinion-prediction examples from canonical data.

    For each selected example, masks opinion spans in the input and creates a
    generative-format example with:
      Input: Task: opinion-prediction
             Input: <sentence with masked opinions>
             Polarity: <polarity>
      Target: <opinion span(s)>

    This teaches the model what opinion expressions look like for a given polarity
    and syntactic context, without requiring exact lexical match at test time.

    Args:
        examples: canonical dicts with 'tokens', 'annotations' (need sentiment_idx, polarity)
        fraction: proportion of examples to generate aux tasks from
        seed: random seed

    Returns:
        list of generative-format dicts (input/target/_keys/_format)
    """
    rng = random.Random(seed)
    n_select = int(len(examples) * fraction)
    if n_select == 0:
        return []
    selected = rng.sample(range(len(examples)), k=n_select)

    aux_examples = []
    for idx in selected:
        ex = examples[idx]
        # skip examples with no opinion spans
        has_opinions = any(a.get("sentiment_idx") for a in ex["annotations"])
        if not has_opinions:
            continue

        masked = _mask_opinion_example(ex, mask_target=False)

        # group annotations by polarity for the target
        # (all opinions in this sentence share the masked input)
        opinions_by_polarity: dict[str, list[str]] = {}
        for ann in ex["annotations"]:
            if ann.get("sentiment") and ann.get("polarity") and ann["sentiment"] != "IMPLICIT":
                pol = ann["polarity"]
                opinions_by_polarity.setdefault(pol, []).append(ann["sentiment"])

        # create one aux example per polarity group
        for polarity, opinions in opinions_by_polarity.items():
            target = " ; ".join(f"the opinion expressed is {op}" for op in opinions)
            input_text = (
                f"Task: opinion-prediction\n"
                f"Input: {masked['sentence']}\n"
                f"Polarity: {polarity}"
            )
            aux_examples.append({
                "input": input_text,
                "target": target,
                "_keys": ["sentiment"],
                "_format": "auxiliary",
            })

    return aux_examples
