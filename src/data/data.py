import ast
import re
import random
from itertools import permutations
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from constants import SENTIMENT_MAP, CANONICAL_KEY_ORDER, Task, TASK_TO_KEY
from src.data.utils import find_span_indices

# ---------------------------------------------------------------------------
# Canonical format
# ---------------------------------------------------------------------------
# {
#   "sentence": str,
#   "tokens":   list[str],
#   "annotations": [
#     {
#       "aspect":       str | None,
#       "aspect_idx":   list[int] | None,
#       "sentiment":    str | None,
#       "sentiment_idx":list[int] | None,
#       "polarity":     str | None,
#       "category":     str | None,
#     }, ...
#   ]
# }
# ---------------------------------------------------------------------------

def parse_aste_line(line: str) -> dict:
    text, raw_labels = line.strip().split("####")
    tokens = text.split()
    triplets = ast.literal_eval(raw_labels)
    annotations = []
    for aspect_idx, opinion_idx, sentiment in triplets:
        has_aspect = aspect_idx not in (None, [-1], (-1, -1))
        has_opinion = bool(opinion_idx)
        annotations.append({
            "aspect":        " ".join(tokens[j] for j in aspect_idx) if has_aspect else "IMPLICIT",
            "aspect_idx":    list(aspect_idx) if has_aspect else None,
            "sentiment":     " ".join(tokens[j] for j in opinion_idx) if has_opinion else "IMPLICIT",
            "sentiment_idx": list(opinion_idx) if has_opinion else None,
            "polarity":      SENTIMENT_MAP.get(sentiment, sentiment.lower()) if sentiment else None,
            "category":      None,
        })
    return {"sentence": text, "tokens": tokens, "annotations": annotations}


def load_aste_file(file_path: str) -> list[dict]:
    examples = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                examples.append(parse_aste_line(line))
    return examples


def _parse_silviolima_row(row: dict) -> dict:
    sentence = row["sentence"]
    tokens = sentence.split()
    triples = ast.literal_eval(row["triples"])
    annotations = []
    for aspect, opinion, polarity in triples:
        aspect_text = "IMPLICIT" if aspect == -1 else str(aspect)
        opinion_text = str(opinion) if opinion else "IMPLICIT"
        aspect_idx = find_span_indices(tokens, aspect_text) if aspect_text != "IMPLICIT" else None
        opinion_idx = find_span_indices(tokens, opinion_text) if opinion_text != "IMPLICIT" else None
        annotations.append({
            "aspect": aspect_text,
            "aspect_idx": aspect_idx,
            "sentiment": opinion_text,
            "sentiment_idx": opinion_idx,
            "polarity": SENTIMENT_MAP.get(polarity, polarity.lower()) if polarity else None,
            "category": None,
        })
    return {"sentence": sentence, "tokens": tokens, "annotations": annotations}


def load_silviolima(file_path: str, domain: str = None) -> list[dict]:
    import json as _json
    with open(file_path) as f:
        content = f.read()
    all_rows = []
    for m in re.finditer(r'"rows":\[(.*?)\],"num_rows_total"', content, re.DOTALL):
        rows = _json.loads('[' + m.group(1) + ']')
        all_rows.extend(rows)
    return [
        _parse_silviolima_row(r["row"])
        for r in all_rows
        if not domain or r["row"]["domain"].lower() == domain.lower()
    ]


def load_silviolima_domain(file_path: str) -> list[dict]:
    import json as _json
    with open(file_path) as f:
        rows = _json.load(f)
    return [_parse_silviolima_row(row) for row in rows]


def load_acos_jsonl(file_path: str) -> list[dict]:
    import json as _json
    examples = []
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = _json.loads(line)
            sentence = row["text"]
            tokens = sentence.split()
            annotations = []
            for label in row["labels"]:
                aspect_text = label["aspect"] if label["aspect"] != "NULL" else "IMPLICIT"
                opinion_text = label["opinion"] if label["opinion"] != "NULL" else "IMPLICIT"
                aspect_idx = find_span_indices(tokens, aspect_text) if aspect_text != "IMPLICIT" else None
                opinion_idx = find_span_indices(tokens, opinion_text) if opinion_text != "IMPLICIT" else None
                annotations.append({
                    "aspect": aspect_text,
                    "aspect_idx": aspect_idx,
                    "sentiment": opinion_text,
                    "sentiment_idx": opinion_idx,
                    "polarity": SENTIMENT_MAP.get(label["polarity"].upper(), label["polarity"].lower()),
                    "category": label.get("category") or "NONE",
                })
            examples.append({"sentence": sentence, "tokens": tokens, "annotations": annotations})
    return examples


def filter_implicit_aspects(examples: list[dict]) -> list[dict]:
    """Remove annotations with IMPLICIT aspect or sentiment. Drop examples with no remaining annotations."""
    filtered = []
    for ex in examples:
        anns = [a for a in ex["annotations"]
                if a.get("aspect") != "IMPLICIT" and a.get("sentiment") != "IMPLICIT"]
        if anns:
            filtered.append({**ex, "annotations": anns})
    return filtered


def enrich_syntax(examples: list[dict], mode: str) -> list[dict]:
    """Add syntactic annotations to canonical examples. Caches results in '_syntax' field.
    mode: 'dep-tree', 'dep-compact', 'dep-inline', 'pos-inline'
    """
    import spacy
    nlp = spacy.load("en_core_web_sm")
    CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    sentences = [ex["sentence"] for ex in examples]
    docs = list(nlp.pipe(sentences, batch_size=256))
    for ex, doc in zip(examples, docs):
        non_punct = [t for t in doc if not t.is_punct]
        if mode == "dep-tree":
            ex["_syntax"] = " ".join(f"{t.text}({t.dep_}->{t.head.text})" for t in non_punct)
        elif mode == "dep-compact":
            ex["_syntax"] = " ".join(
                f"{t.text}->{t.head.text}:{t.dep_}" for t in non_punct if t.pos_ in CONTENT_POS
            )
        elif mode == "dep-inline":
            ex["_syntax_tokens"] = [f"{t.text}/{t.dep_}" for t in non_punct]
        elif mode == "pos-inline":
            ex["_syntax_tokens"] = [f"{t.text}/{t.pos_}" for t in non_punct]
    return examples


# ---------------------------------------------------------------------------
# Generative format conversion
# ---------------------------------------------------------------------------

def _encode_target(items: list[dict]) -> str:
    return " ".join(
        "[" + ", ".join(str(v) for v in d.values()) + "]"
        for d in items
    )


_NL_TEMPLATES = {
    # singles
    frozenset(["aspect"]): "the aspect being discussed is {aspect}",
    frozenset(["sentiment"]): "the opinion expressed is {sentiment}",
    frozenset(["polarity"]): "the overall sentiment is {polarity}",
    frozenset(["category"]): "the category being discussed is {category}",
    # pairs
    frozenset(["aspect", "sentiment"]): "{aspect} is described as {sentiment}",
    frozenset(["aspect", "polarity"]): "the opinion about {aspect} is {polarity}",
    frozenset(["aspect", "category"]): "{aspect} falls under the category {category}",
    frozenset(["sentiment", "polarity"]): "the opinion {sentiment} conveys a {polarity} sentiment",
    frozenset(["sentiment", "category"]): "the opinion {sentiment} is about the category {category}",
    frozenset(["polarity", "category"]): "the sentiment toward {category} is {polarity}",
    # triples
    frozenset(["aspect", "sentiment", "polarity"]): "{aspect} is described as {sentiment}, expressing a {polarity} sentiment",
    frozenset(["aspect", "sentiment", "category"]): "{aspect} is described as {sentiment}, under the category {category}",
    frozenset(["aspect", "polarity", "category"]): "the {polarity} opinion about {aspect} falls under {category}",
    frozenset(["sentiment", "polarity", "category"]): "the opinion {sentiment} conveys a {polarity} sentiment about {category}",
    # quad
    frozenset(["aspect", "sentiment", "polarity", "category"]): "{aspect} is described as {sentiment}, expressing a {polarity} sentiment about {category}",
}


def _encode_target_nl(items: list[dict], keys_set: frozenset) -> str:
    template = _NL_TEMPLATES.get(keys_set)
    if not template:
        raise ValueError(f"No natural-language template for keys: {keys_set}")
    parts = []
    for d in items:
        d_resolved = dict(d)
        # handle implicit aspect inference
        if "aspect" in d_resolved and d_resolved["aspect"].startswith("IMPLIED:"):
            original_term = d_resolved["aspect"][len("IMPLIED:"):]
            d_resolved["aspect"] = original_term
            # use implicit template: replace "{aspect} is" with "the implied aspect is {aspect},"
            implicit_template = template.replace("{aspect} is", "the implied aspect is {aspect},")
            if implicit_template == template:
                # fallback for templates that don't start with "{aspect} is"
                implicit_template = template.replace("{aspect}", "the implied aspect {aspect}")
            parts.append(implicit_template.format(**d_resolved))
        else:
            parts.append(template.format(**d_resolved))
    return " ; ".join(parts)


def _decode_target(raw: str, keys: list[str]) -> list[dict]:
    results = []
    for match in re.finditer(r"\[([^\[\]]+)\]", raw):
        values = [v.strip() for v in match.group(1).split(",")]
        if len(values) == len(keys):
            results.append(dict(zip(keys, values)))
    return results


def to_generative_format(canonical: dict, tasks: list[Task], output_format: str = "structured", infer_implicit: bool = False) -> dict:
    if output_format == "natural-language":
        keys = [k for k in CANONICAL_KEY_ORDER if k in {TASK_TO_KEY[t] for t in tasks}]
        task_list = [t for t in [Task.ASPECT, Task.SENTIMENT, Task.POLARITY, Task.CATEGORY] if TASK_TO_KEY[t] in keys]
    else:
        keys = [TASK_TO_KEY[t] for t in tasks]
        task_list = tasks

    task_str = ", ".join(t.value for t in task_list)

    # build input text with optional syntax enrichment
    if "_syntax_tokens" in canonical:
        input_sentence = " ".join(canonical["_syntax_tokens"])
    else:
        input_sentence = canonical["sentence"]

    input_text = f"Task: {task_str}\nInput: {input_sentence}"

    if "_syntax" in canonical:
        input_text += f"\nSyntax: {canonical['_syntax']}"

    input_text += f"\nOutput: {'natural language' if output_format == 'natural-language' else 'structured'}"

    annotations = []
    for ann in canonical["annotations"]:
        d = {}
        for k in keys:
            val = ann.get(k)
            if val is None:
                break
            if k == "aspect" and val == "IMPLICIT" and infer_implicit and ann.get("aspect_original"):
                if output_format == "natural-language":
                    val = f"IMPLIED:{ann['aspect_original']}"
                else:
                    val = f"IMPLICIT:{ann['aspect_original']}"
            d[k] = val
        if len(d) == len(keys):
            annotations.append(d)

    if output_format == "natural-language":
        target = _encode_target_nl(annotations, frozenset(keys))
    else:
        target = _encode_target(annotations)

    return {
        "input":   input_text,
        "target":  target,
        "_keys":   keys,
        "_format": output_format,
    }


def filter_tasks(example: dict, tasks: list[Task]) -> dict:
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("tasks must be non-empty and unique")
    keys = [TASK_TO_KEY[t] for t in tasks]
    task_str = ", ".join(t.value for t in tasks)
    sentence = example["input"].split("Input: ", 1)[1]
    stored_keys = example.get("_keys", list(TASK_TO_KEY.values()))
    triplets = _decode_target(example["target"], stored_keys)
    filtered = [{k: t[k] for k in keys if k in t} for t in triplets]
    return {
        "input":  f"Task: {task_str}\nInput: {sentence}",
        "target": _encode_target(filtered),
        "_keys":  keys,
    }


def interpolate_curriculum(curriculum: list[dict], epoch: int) -> dict[str, float]:
    """Interpolate task partition weights between curriculum waypoints.
    Each waypoint: {"epoch": int, "tasks_partition": {task_key: fraction}}
    Returns interpolated tasks_partition for the given epoch.
    """
    if len(curriculum) == 1:
        return dict(curriculum[0]["tasks_partition"])

    # clamp to first/last waypoint
    if epoch <= curriculum[0]["epoch"]:
        return dict(curriculum[0]["tasks_partition"])
    if epoch >= curriculum[-1]["epoch"]:
        return dict(curriculum[-1]["tasks_partition"])

    # find surrounding waypoints
    for i in range(len(curriculum) - 1):
        a, b = curriculum[i], curriculum[i + 1]
        if a["epoch"] <= epoch < b["epoch"]:
            t = (epoch - a["epoch"]) / (b["epoch"] - a["epoch"])
            all_keys = set(a["tasks_partition"]) | set(b["tasks_partition"])
            result = {}
            for k in all_keys:
                w_a = a["tasks_partition"].get(k, 0.0)
                w_b = b["tasks_partition"].get(k, 0.0)
                w = w_a + (w_b - w_a) * t
                if w > 0:
                    result[k] = round(w, 6)
            # normalize to sum to 1.0
            total = sum(result.values())
            if total > 0:
                result = {k: v / total for k, v in result.items()}
            return result

    return dict(curriculum[-1]["tasks_partition"])


def split_by_task(
    file_path: str,
    tasks_partition: dict[tuple[Task, ...], float],
    seed: int = 42,
    shuffle_tasks: bool = False,
    examples: list[dict] = None,
    nl_fraction: float = 0.0,
    infer_implicit: bool = False,
) -> dict[tuple[Task, ...], list[dict]]:
    if not tasks_partition:
        raise ValueError("tasks_partition must not be empty")
    total = sum(tasks_partition.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    canonical = examples if examples is not None else load_aste_file(file_path)
    rng = random.Random(seed)
    indices = list(range(len(canonical)))
    rng.shuffle(indices)

    # select which examples get natural-language format
    nl_indices = set()
    if nl_fraction > 0:
        nl_indices = set(rng.sample(range(len(canonical)), k=int(len(canonical) * nl_fraction)))

    perms_by_group = {g: list(permutations(g)) for g in tasks_partition}
    keys = list(tasks_partition.keys())
    n = len(indices)
    partitions: dict[tuple[Task, ...], list[dict]] = {k: [] for k in keys}

    start = 0
    for i, task_group in enumerate(keys):
        end = n if i == len(keys) - 1 else start + round(tasks_partition[task_group] * n)
        perms = perms_by_group[task_group]
        for idx in indices[start:end]:
            fmt = "natural-language" if idx in nl_indices else "structured"
            ordered = list(rng.choice(perms) if shuffle_tasks and fmt == "structured" else task_group)
            partitions[task_group].append(to_generative_format(canonical[idx], ordered, output_format=fmt, infer_implicit=infer_implicit))
        start = end

    return partitions


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ABSADataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer: AutoTokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inputs = self.tokenizer(
            ex["input"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            ex["target"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = targets["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids":      inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels":         labels,
            "raw_target":     ex["target"],
            "raw_input":      ex["input"],
            "keys":           ",".join(ex.get("_keys", ["aspect", "sentiment", "polarity"])),
            "output_format":  ex.get("_format", "structured"),
        }
