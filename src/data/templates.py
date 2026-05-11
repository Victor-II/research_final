import re

# ---------------------------------------------------------------------------
# English NL Templates (singular)
# ---------------------------------------------------------------------------

_NL_TEMPLATES_EN = {
    # singles
    frozenset(["aspect"]): "the aspect being discussed is {aspect}",
    frozenset(["sentiment"]): "the opinion expressed is {sentiment}",
    frozenset(["polarity"]): "the overall sentiment is {polarity}",
    frozenset(["category"]): "the category being discussed is {category}",
    # pairs
    frozenset(["aspect", "sentiment"]): "{aspect} is described as {sentiment}",
    frozenset(["aspect", "polarity"]): "the opinion about {aspect} is {polarity}",
    frozenset(["aspect", "category"]): "{aspect}, related to {category}, is being discussed",
    frozenset(["sentiment", "polarity"]): "the opinion {sentiment} conveys a {polarity} sentiment",
    frozenset(["sentiment", "category"]): "the opinion {sentiment} is about the category {category}",
    frozenset(["polarity", "category"]): "the sentiment toward {category} is {polarity}",
    # triples
    frozenset(["aspect", "sentiment", "polarity"]): "{aspect} is described as {sentiment}, expressing a {polarity} sentiment",
    frozenset(["aspect", "sentiment", "category"]): "{aspect}, related to {category}, is described as {sentiment}",
    frozenset(["aspect", "polarity", "category"]): "{aspect}, related to {category}, expresses a {polarity} sentiment",
    frozenset(["sentiment", "polarity", "category"]): "the opinion {sentiment} conveys a {polarity} sentiment about {category}",
    # quad
    frozenset(["aspect", "sentiment", "polarity", "category"]): "{aspect}, related to {category}, is described as {sentiment}, expressing a {polarity} sentiment",
}

# Plural forms for single-key tasks (2+ annotations → comma-separated list)
_NL_TEMPLATES_PLURAL_EN = {
    frozenset(["aspect"]): "the aspects being discussed are {list}",
    frozenset(["sentiment"]): "the opinions expressed are {list}",
    frozenset(["polarity"]): "the overall sentiments are {list}",
    frozenset(["category"]): "the categories being discussed are {list}",
}

_NL_TEMPLATES_IMPLICIT_SENTIMENT_EN = {
    frozenset(["aspect", "sentiment", "polarity", "category"]): "{aspect}, related to {category}, carries an implied {polarity} opinion",
    frozenset(["aspect", "sentiment", "polarity"]): "{aspect} carries an implied {polarity} opinion",
    frozenset(["aspect", "sentiment", "category"]): "{aspect}, related to {category}, carries an implied opinion",
    frozenset(["aspect", "sentiment"]): "{aspect} carries an implied opinion",
    frozenset(["sentiment", "polarity"]): "an implied opinion conveys a {polarity} sentiment",
    frozenset(["sentiment", "polarity", "category"]): "an implied opinion conveys a {polarity} sentiment about {category}",
}

# ---------------------------------------------------------------------------
# Romanian NL Templates (singular)
# ---------------------------------------------------------------------------

_NL_TEMPLATES_RO = {
    frozenset(["polarity", "category"]): "Parerea despre {category} este una {polarity}",
    frozenset(["polarity"]): "parerea generală este {polarity}",
    frozenset(["category"]): "categoria discutată este {category}",
    # ordered variants for shuffle_tasks
    ("category", "polarity"): "Parerea despre {category} este una {polarity}",
    ("polarity", "category"): "review-ul exprimă o părere {polarity} despre {category}",
}

_NL_TEMPLATES_PLURAL_RO = {
    frozenset(["polarity"]): "parerile generale sunt {list}",
    frozenset(["category"]): "categoriile discutate sunt {list}",
}

_NL_TEMPLATES_IMPLICIT_SENTIMENT_RO = {}

# ---------------------------------------------------------------------------
# Polarity maps
# ---------------------------------------------------------------------------

POLARITY_EN_TO_RO = {
    "positive": "pozitivă",
    "negative": "negativă",
    "neutral": "neutră",
}

POLARITY_RO_TO_EN = {v: k for k, v in POLARITY_EN_TO_RO.items()}

# ---------------------------------------------------------------------------
# Category maps
# ---------------------------------------------------------------------------

CATEGORY_EN_TO_RO = {
    "experience": "experiență",
    "durability": "durabilitate",
    "performance": "performanță",
    "battery": "baterie",
    "price_quality": "raportul calitate-preț",
    "design": "design",
    "camera": "cameră",
    "audio": "audio",
    "software": "software",
    "brand": "brand",
    "service": "service",
}

CATEGORY_RO_TO_EN = {v: k for k, v in CATEGORY_EN_TO_RO.items()}


# ---------------------------------------------------------------------------
# Template dispatch
# ---------------------------------------------------------------------------

def get_templates(language: str = "en"):
    if language == "ro":
        return _NL_TEMPLATES_RO, _NL_TEMPLATES_IMPLICIT_SENTIMENT_RO
    return _NL_TEMPLATES_EN, _NL_TEMPLATES_IMPLICIT_SENTIMENT_EN


def get_ordered_template(keys_tuple: tuple, language: str = "en") -> str | None:
    """Look up an order-dependent template by tuple key. Returns None if not found."""
    if language == "ro":
        return _NL_TEMPLATES_RO.get(keys_tuple)
    return _NL_TEMPLATES_EN.get(keys_tuple)


def get_all_templates_for_keys(keys_set: frozenset, language: str = "en") -> list[str]:
    """Get all template variants (frozenset + all tuple orderings) for a key set."""
    templates, _ = get_templates(language)
    source = _NL_TEMPLATES_RO if language == "ro" else _NL_TEMPLATES_EN
    result = []
    # frozenset entry
    if keys_set in source:
        result.append(source[keys_set])
    # tuple entries
    for k, v in source.items():
        if isinstance(k, tuple) and frozenset(k) == keys_set and v not in result:
            result.append(v)
    return result


def get_plural_templates(language: str = "en"):
    if language == "ro":
        return _NL_TEMPLATES_PLURAL_RO
    return _NL_TEMPLATES_PLURAL_EN


def translate_to_output(value: str, key: str, language: str) -> str:
    """Translate a canonical (English) value to the output language."""
    if language == "en":
        return value
    if key == "polarity":
        return POLARITY_EN_TO_RO.get(value, value)
    if key == "category":
        return CATEGORY_EN_TO_RO.get(value, value)
    return value


def translate_from_output(value: str, key: str, language: str) -> str:
    """Translate a parsed output value back to canonical (English)."""
    if language == "en":
        return value
    if key == "polarity":
        return POLARITY_RO_TO_EN.get(value, value)
    if key == "category":
        return CATEGORY_RO_TO_EN.get(value, value)
    return value


# ---------------------------------------------------------------------------
# Prompt labels per language
# ---------------------------------------------------------------------------

_PROMPT_LABELS = {
    "en": {
        "task": "Task",
        "input": "Input",
        "syntax": "Syntax",
        "categories": "Categories",
        "output": "Output",
        "output_nl": "natural language",
        "output_struct": "structured",
    },
    "ro": {
        "task": "Sarcină",
        "input": "Text",
        "syntax": "Sintaxă",
        "categories": "Categorii",
        "output": "Format",
        "output_nl": "limbaj natural",
        "output_struct": "structurat",
    },
}

_TASK_NAMES = {
    "en": {
        "aspect-extraction": "aspect-extraction",
        "sentiment-extraction": "sentiment-extraction",
        "polarity-inference": "polarity-inference",
        "category-extraction": "category-inference",
    },
    "ro": {
        "aspect-extraction": "extragere-aspect",
        "sentiment-extraction": "extragere-opinie",
        "polarity-inference": "inferență-polaritate",
        "category-extraction": "inferență-categorie",
    },
}


def get_prompt_labels(language: str = "en") -> dict:
    return _PROMPT_LABELS.get(language, _PROMPT_LABELS["en"])


def get_task_name(task_value: str, language: str = "en") -> str:
    names = _TASK_NAMES.get(language, _TASK_NAMES["en"])
    return names.get(task_value, task_value)
