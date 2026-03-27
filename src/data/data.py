import ast
import random
from enum import Enum
from itertools import permutations


class Task(str, Enum):
    ASPECT = "aspect-extraction"
    SENTIMENT = "sentiment-extraction"
    POLARITY = "polarity-inference"

    # Convenience: allow lookup by short key (e.g. Task["aspect"])
    @classmethod
    def _missing_(cls, value):
        for member in cls:
            if member.name.lower() == value.lower():
                return member
        return None


# Maps triplet dict keys -> Task enum
TRIPLET_KEY_TO_TASK = {
    "aspect": Task.ASPECT,
    "sentiment": Task.SENTIMENT,
    "polarity": Task.POLARITY,
}

SENTIMENT_MAP = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}


def parse_aste_line(line: str) -> dict:
    """Parse a single line from an ASTE .txt file into a dict with sentence and triplets."""
    text, raw_labels = line.strip().split("####")
    tokens = text.split()
    triplets = ast.literal_eval(raw_labels)

    parsed_triplets = []
    for aspect_idx, opinion_idx, sentiment in triplets:
        parsed_triplets.append({
            "aspect": " ".join(tokens[j] for j in aspect_idx),
            "sentiment": " ".join(tokens[j] for j in opinion_idx),
            "polarity": SENTIMENT_MAP.get(sentiment, sentiment.lower()),
        })

    return {"sentence": text, "triplets": parsed_triplets}


def _encode_target(items: list[dict]) -> str:
    """Encode a list of dicts to bracket notation: [v1, v2, v3] [v1, v2] ..."""
    return " ".join(
        "[" + ", ".join(str(v) for v in d.values()) + "]"
        for d in items
    )


def _decode_target(raw: str, keys: list[str]) -> list[dict]:
    """Decode bracket notation back to a list of dicts given the expected key order."""
    import re
    results = []
    for match in re.finditer(r"\[([^\[\]]+)\]", raw):
        values = [v.strip() for v in match.group(1).split(",")]
        if len(values) == len(keys):
            results.append(dict(zip(keys, values)))
    return results


def to_generative_format(sentence: str, triplets: list) -> dict:
    """Convert a sentence and its triplets to a T5-style input/target pair."""
    task_str = ", ".join(t.value for t in [Task.ASPECT, Task.SENTIMENT, Task.POLARITY])
    input_text = f"Task: {task_str}\nInput: {sentence}"
    target_text = _encode_target(triplets)
    return {"input": input_text, "target": target_text, "_keys": list(triplets[0].keys()) if triplets else ["aspect", "sentiment", "polarity"]}


def load_aste_file(file_path: str) -> list[dict]:
    """Load an ASTE .txt file and return a list of generative format examples."""
    examples = []
    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue
            parsed = parse_aste_line(line)
            example = to_generative_format(parsed["sentence"], parsed["triplets"])
            examples.append(example)
    return examples
    
def filter_tasks(example: dict, tasks: list[Task]) -> dict:
    """
    Re-format a generative example to only include the specified tasks.

    Args:
        example: a dict produced by to_generative_format, with 'input' and 'target' keys.
        tasks:   ordered list of Task enum members to include.

    Returns:
        A new dict with 'input' and 'target' scoped to the requested tasks.
    """
    if not tasks:
        raise ValueError("'tasks' must not be empty")
    if len(tasks) != len(set(tasks)):
        raise ValueError("'tasks' must not contain duplicates")

    task_to_key = {v: k for k, v in TRIPLET_KEY_TO_TASK.items()}

    task_str = ", ".join(t.value for t in tasks)
    sentence = example["input"].split("Input: ", 1)[1]

    # Decode from bracket notation using the stored key order
    stored_keys = example.get("_keys", list(TRIPLET_KEY_TO_TASK.keys()))
    triplets = _decode_target(example["target"], stored_keys)

    filtered_keys = [task_to_key[t] for t in tasks]
    filtered_triplets = [
        {k: triplet[k] for k in filtered_keys if k in triplet}
        for triplet in triplets
    ]

    return {
        "input": f"Task: {task_str}\nInput: {sentence}",
        "target": _encode_target(filtered_triplets),
        "_keys": filtered_keys,
    }

def split_by_task(
    file_path: str,
    tasks_partition: dict[tuple[Task, ...], float],
    seed: int = 42,
    shuffle_tasks: bool = False,
) -> dict[tuple[Task, ...], list[dict]]:
    """
    Load a dataset and split it into partitions, each scoped to a set of tasks.

    Args:
        file_path:       path to an ASTE .txt file.
        tasks_partition: mapping of task-tuple -> fraction (fractions must sum to 1.0).
                         Each key is a tuple of Task members that partition will expose.
                         e.g. {(Task.ASPECT,): 0.2, (Task.ASPECT, Task.POLARITY): 0.4}
        seed:            random seed for reproducibility.
        shuffle_tasks:   if True, the task order in each partition key is ignored and a
                         random permutation is sampled per example from the set of all
                         distinct permutations of that partition's tasks.

    Returns:
        dict mapping each task-tuple key to its list of filtered examples.
    """
    if not tasks_partition:
        raise ValueError("tasks_partition must not be empty")

    total = sum(tasks_partition.values())
    if not abs(total - 1.0) < 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {total}")

    examples = load_aste_file(file_path)

    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)

    # Pre-compute all distinct permutations per task group
    perms_by_group: dict[tuple[Task, ...], list[tuple[Task, ...]]] = {
        group: list(permutations(group)) for group in tasks_partition
    }

    keys = list(tasks_partition.keys())
    n = len(indices)
    partitions: dict[tuple[Task, ...], list[dict]] = {k: [] for k in keys}

    start = 0
    for i, task_group in enumerate(keys):
        end = n if i == len(keys) - 1 else start + round(tasks_partition[task_group] * n)
        perms = perms_by_group[task_group]
        for idx in indices[start:end]:
            ordered = list(rng.choice(perms) if shuffle_tasks else task_group)
            partitions[task_group].append(filter_tasks(examples[idx], ordered))
        start = end

    return partitions


