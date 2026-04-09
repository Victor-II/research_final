import ast
import re
import random
from enum import Enum
from itertools import permutations
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class Task(str, Enum):
    ASPECT    = "aspect-extraction"
    SENTIMENT = "sentiment-extraction"
    POLARITY  = "polarity-inference"
    CATEGORY  = "category-extraction"

    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.name.lower() == value.lower():
                return member
        return None


TASK_TO_KEY = {
    Task.ASPECT:    "aspect",
    Task.SENTIMENT: "sentiment",
    Task.POLARITY:  "polarity",
    Task.CATEGORY:  "category",
}

SENTIMENT_MAP = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

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
        annotations.append({
            "aspect":        " ".join(tokens[j] for j in aspect_idx) if aspect_idx not in (None, [-1], (-1, -1)) else None,
            "aspect_idx":    list(aspect_idx) if aspect_idx not in (None, [-1], (-1, -1)) else None,
            "sentiment":     " ".join(tokens[j] for j in opinion_idx) if opinion_idx else None,
            "sentiment_idx": list(opinion_idx) if opinion_idx else None,
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


# ---------------------------------------------------------------------------
# Generative format conversion
# ---------------------------------------------------------------------------

def _encode_target(items: list[dict]) -> str:
    return " ".join(
        "[" + ", ".join(str(v) for v in d.values()) + "]"
        for d in items
    )


def _decode_target(raw: str, keys: list[str]) -> list[dict]:
    results = []
    for match in re.finditer(r"\[([^\[\]]+)\]", raw):
        values = [v.strip() for v in match.group(1).split(",")]
        if len(values) == len(keys):
            results.append(dict(zip(keys, values)))
    return results


def to_generative_format(canonical: dict, tasks: list[Task]) -> dict:
    keys = [TASK_TO_KEY[t] for t in tasks]
    task_str = ", ".join(t.value for t in tasks)
    input_text = f"Task: {task_str}\nInput: {canonical['sentence']}"
    annotations = [
        {k: ann[k] for k in keys if ann.get(k) is not None}
        for ann in canonical["annotations"]
    ]
    annotations = [a for a in annotations if len(a) == len(keys)]
    return {
        "input":   input_text,
        "target":  _encode_target(annotations),
        "_keys":   keys,
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


def split_by_task(
    file_path: str,
    tasks_partition: dict[tuple[Task, ...], float],
    seed: int = 42,
    shuffle_tasks: bool = False,
    examples: list[dict] = None,
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

    perms_by_group = {g: list(permutations(g)) for g in tasks_partition}
    keys = list(tasks_partition.keys())
    n = len(indices)
    partitions: dict[tuple[Task, ...], list[dict]] = {k: [] for k in keys}

    start = 0
    for i, task_group in enumerate(keys):
        end = n if i == len(keys) - 1 else start + round(tasks_partition[task_group] * n)
        perms = perms_by_group[task_group]
        for idx in indices[start:end]:
            ordered = list(rng.choice(perms) if shuffle_tasks else task_group)
            partitions[task_group].append(to_generative_format(canonical[idx], ordered))
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
        }
