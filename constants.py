from pathlib import Path
from enum import Enum


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

BASE_CONFIG = Path("config/base.yaml")
EXPERIMENTS_DIR = Path("experiments")
REPLACE_KEYS = {"tasks_partition", "scopes", "datasets"}

SENTIMENT_MAP = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}

CANONICAL_KEY_ORDER = ["aspect", "sentiment", "polarity", "category"]

TASK_KEY_MAP = {
    # singles
    "aspect":    (Task.ASPECT,),
    "polarity":  (Task.POLARITY,),
    "sentiment": (Task.SENTIMENT,),
    "category":  (Task.CATEGORY,),
    # pairs
    "aspect+sentiment":   (Task.ASPECT, Task.SENTIMENT),
    "aspect+polarity":    (Task.ASPECT, Task.POLARITY),
    "aspect+category":    (Task.ASPECT, Task.CATEGORY),
    "sentiment+polarity": (Task.SENTIMENT, Task.POLARITY),
    "sentiment+category": (Task.SENTIMENT, Task.CATEGORY),
    "polarity+category":  (Task.POLARITY, Task.CATEGORY),
    # triples
    "full":      (Task.ASPECT, Task.SENTIMENT, Task.POLARITY),
    "aspect+sentiment+category":   (Task.ASPECT, Task.SENTIMENT, Task.CATEGORY),
    "aspect+polarity+category":    (Task.ASPECT, Task.POLARITY, Task.CATEGORY),
    "sentiment+polarity+category": (Task.SENTIMENT, Task.POLARITY, Task.CATEGORY),
    # quad
    "quad":      (Task.ASPECT, Task.SENTIMENT, Task.POLARITY, Task.CATEGORY),
}
