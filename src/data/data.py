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


def load_emag_csv(file_path: str, include_title: bool = True) -> list[dict]:
    import csv
    examples = []
    with open(file_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        # detect column names (train vs test format)
        title_col = "title" if "title" in fields else "Title"
        review_col = "review" if "review" in fields else "Review"
        aspects_col = "aspects" if "aspects" in fields else "Aspects"
        for row in reader:
            title = (row.get(title_col) or "").strip()
            review = (row.get(review_col) or "").strip()
            if not review and not title:
                continue
            sentence = f"{title}. {review}" if include_title and title else review
            tokens = sentence.split()
            raw_aspects = (row.get(aspects_col) or "").strip()
            annotations = []
            if raw_aspects:
                for pair in raw_aspects.split(";"):
                    pair = pair.strip()
                    if ":" not in pair:
                        continue
                    category, polarity = pair.rsplit(":", 1)
                    annotations.append({
                        "aspect": None,
                        "aspect_idx": None,
                        "sentiment": None,
                        "sentiment_idx": None,
                        "polarity": polarity.strip(),
                        "category": category.strip(),
                    })
            if annotations:
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


def _parse_semeval_xml_opinion(opinion_el, tokens: list[str]) -> dict:
    """Parse a single <Opinion> element into canonical annotation format."""
    target = opinion_el.get("target")
    aspect_text = "IMPLICIT" if target == "NULL" else target
    aspect_idx = find_span_indices(tokens, aspect_text) if aspect_text != "IMPLICIT" else None
    polarity = opinion_el.get("polarity", "").lower()
    category = opinion_el.get("category") or "NONE"
    return {
        "aspect": aspect_text,
        "aspect_idx": aspect_idx,
        "sentiment": None,  # SemEval XML doesn't have opinion terms
        "sentiment_idx": None,
        "polarity": polarity,
        "category": category,
    }


def load_semeval_xml(file_path: str) -> list[dict]:
    """Load SemEval ABSA XML into flat sentence-level canonical format."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_path)
    examples = []
    for sentence in tree.findall(".//sentence"):
        text = sentence.find("text").text or ""
        tokens = text.split()
        opinions = sentence.findall(".//Opinion")
        if not opinions:
            continue
        annotations = [_parse_semeval_xml_opinion(op, tokens) for op in opinions]
        examples.append({"sentence": text, "tokens": tokens, "annotations": annotations})
    return examples


def load_semeval_xml_reviews(file_path: str) -> list[dict]:
    """Load SemEval ABSA XML preserving review-level grouping.

    Returns list of review dicts:
    {
        "review_id": str,
        "sentences": [
            {
                "sentence_id": str,
                "sentence": str,
                "tokens": list[str],
                "annotations": [canonical annotation dicts],
            }, ...
        ]
    }
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_path)
    reviews = []
    for review in tree.findall(".//Review"):
        rid = review.get("rid")
        sents = []
        for sentence in review.findall(".//sentence"):
            text = sentence.find("text").text or ""
            tokens = text.split()
            opinions = sentence.findall(".//Opinion")
            annotations = [_parse_semeval_xml_opinion(op, tokens) for op in opinions]
            sents.append({
                "sentence_id": sentence.get("id"),
                "sentence": text,
                "tokens": tokens,
                "annotations": annotations,
            })
        reviews.append({"review_id": rid, "sentences": sents})
    return reviews


def enrich_syntax(examples: list[dict], mode: str, spacy_model: str = "en_core_web_sm") -> list[dict]:
    """Add syntactic annotations to canonical examples. Caches results in '_syntax' field.
    mode: 'dep-tree', 'dep-compact', 'dep-inline', 'pos-inline'
    """
    import spacy
    nlp = spacy.load(spacy_model)
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
        elif mode == "pos-compact":
            ex["_syntax"] = " ".join(
                f"{t.text}/{t.pos_}" for t in non_punct if t.pos_ in CONTENT_POS
            )
        elif mode == "dep-nl":
            DEP_NL = {
                "nsubj": "is the subject of", "dobj": "is the object of",
                "amod": "modifies", "advmod": "modifies",
                "acomp": "describes", "attr": "is an attribute of",
                "nsubjpass": "is the passive subject of", "pobj": "is the object of",
                "conj": "is coordinated with", "ROOT": "is the main verb",
                "compound": "is part of", "neg": "negates",
                "xcomp": "complements", "ccomp": "complements",
                "prep": "is a preposition in", "det": "determines",
                "aux": "is auxiliary to", "relcl": "is modified by a clause about",
            }
            parts = []
            for t in non_punct:
                if t.pos_ in CONTENT_POS:
                    rel = DEP_NL.get(t.dep_, f"is related to")
                    if t.dep_ == "ROOT":
                        parts.append(f'"{t.text}" is the root')
                    else:
                        parts.append(f'"{t.text}" {rel} "{t.head.text}"')
            ex["_syntax"] = "; ".join(parts)
        elif mode == "pos-nl":
            POS_NL = {
                "NOUN": "noun", "VERB": "verb", "ADJ": "adjective",
                "ADV": "adverb", "PROPN": "proper noun",
            }
            parts = []
            for t in non_punct:
                if t.pos_ in CONTENT_POS:
                    pos_label = POS_NL.get(t.pos_, t.pos_.lower())
                    parts.append(f'"{t.text}" is a {pos_label}')
            ex["_syntax"] = "; ".join(parts)
    return examples


# ---------------------------------------------------------------------------
# Auxiliary syntax prediction tasks
# ---------------------------------------------------------------------------

def to_syntax_auxiliary(canonical: dict, task: str = "dep") -> dict | None:
    """Generate an auxiliary syntax prediction example.
    task: 'dep', 'pos', 'dep-nl', 'pos-nl'.
    Returns a generative format dict or None if syntax info unavailable."""
    sentence = canonical["sentence"]

    task_map = {
        "dep": ("dependency-prediction", "_syntax_dep", "_syntax"),
        "pos": ("pos-tagging", "_syntax_pos", None),
        "dep-nl": ("describe-dependencies", "_syntax_dep_nl", None),
        "pos-nl": ("describe-word-types", "_syntax_pos_nl", None),
    }
    if task not in task_map:
        return None

    task_name, primary_field, fallback_field = task_map[task]
    target = canonical.get(primary_field)
    if not target and fallback_field:
        target = canonical.get(fallback_field)
    if not target:
        return None

    return {
        "input": f"Task: {task_name}\nInput: {sentence}\nOutput:",
        "target": target,
        "_keys": [],
        "_format": "auxiliary",
    }


def enrich_syntax_auxiliary(examples: list[dict]) -> list[dict]:
    """Add dep and pos strings in both compact and NL formats for auxiliary tasks."""
    import spacy
    nlp = spacy.load("en_core_web_sm")
    CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
    DEP_NL = {
        "nsubj": "is the subject of", "dobj": "is the object of",
        "amod": "modifies", "advmod": "modifies",
        "acomp": "describes", "attr": "is an attribute of",
        "nsubjpass": "is the passive subject of", "pobj": "is the object of",
        "conj": "is coordinated with", "ROOT": "is the root",
        "compound": "is part of", "neg": "negates",
        "xcomp": "complements", "ccomp": "complements",
    }
    POS_NL = {"NOUN": "noun", "VERB": "verb", "ADJ": "adjective", "ADV": "adverb", "PROPN": "proper noun"}
    sentences = [ex["sentence"] for ex in examples]
    docs = list(nlp.pipe(sentences, batch_size=256))
    for ex, doc in zip(examples, docs):
        non_punct = [t for t in doc if not t.is_punct]
        content = [t for t in non_punct if t.pos_ in CONTENT_POS]
        ex["_syntax_dep"] = " ".join(f"{t.text}->{t.head.text}:{t.dep_}" for t in content)
        ex["_syntax_pos"] = " ".join(f"{t.text}/{t.pos_}" for t in content)
        # NL versions
        dep_parts = []
        for t in content:
            rel = DEP_NL.get(t.dep_, "is related to")
            dep_parts.append(f'"{t.text}" {rel} "{t.head.text}"' if t.dep_ != "ROOT" else f'"{t.text}" is the root')
        ex["_syntax_dep_nl"] = "; ".join(dep_parts)
        ex["_syntax_pos_nl"] = "; ".join(f'"{t.text}" is a {POS_NL.get(t.pos_, t.pos_.lower())}' for t in content)
    return examples


# ---------------------------------------------------------------------------
# Generative format conversion
# ---------------------------------------------------------------------------

def _encode_target(items: list[dict]) -> str:
    return " ".join(
        "[" + ", ".join(str(v) for v in d.values()) + "]"
        for d in items
    )


from src.data.templates import get_templates, get_plural_templates, get_ordered_template, get_all_templates_for_keys, translate_to_output, translate_from_output, get_prompt_labels, get_task_name

# backward-compat aliases used by constrained.py and eval.py
_NL_TEMPLATES, _NL_TEMPLATES_IMPLICIT_SENTIMENT = get_templates("en")


def _encode_target_nl(items: list[dict], keys_set: frozenset, language: str = "en", keys_ordered: tuple = None) -> str:
    templates, implicit_templates = get_templates(language)
    plural_templates = get_plural_templates(language)

    # try ordered template first if provided
    template = None
    if keys_ordered:
        template = get_ordered_template(keys_ordered, language)
    if not template:
        template = templates.get(keys_set)
    if not template:
        raise ValueError(f"No natural-language template for keys: {keys_set} (language={language})")

    # single-key tasks with multiple annotations → use plural form
    if len(keys_set) == 1 and len(items) > 1:
        plural_tmpl = plural_templates.get(keys_set)
        if plural_tmpl:
            key = next(iter(keys_set))
            values = [translate_to_output(d[key], key, language) for d in items]
            return plural_tmpl.replace("{list}", ", ".join(values))

    implicit_sent_template = implicit_templates.get(keys_set)
    parts = []
    for d in items:
        d_resolved = {}
        for k, v in d.items():
            d_resolved[k] = translate_to_output(v, k, language)

        aspect_implicit = False
        sentiment_implicit = False

        # detect implicit aspect
        if "aspect" in d_resolved:
            val = d_resolved["aspect"]
            if val.startswith("IMPLIED:"):
                d_resolved["aspect"] = val[len("IMPLIED:"):]
                aspect_implicit = "annotated"
            elif val == "IMPLICIT":
                d_resolved["aspect"] = "an unspecified aspect"
                aspect_implicit = "unknown"

        # detect implicit sentiment
        if "sentiment" in d_resolved and d_resolved["sentiment"] == "IMPLICIT":
            sentiment_implicit = True

        # choose template
        if sentiment_implicit and implicit_sent_template:
            tmpl = implicit_sent_template
        else:
            tmpl = template

        # apply aspect prefix for annotated implicit
        if aspect_implicit == "annotated":
            tmpl = tmpl.replace("{aspect}", "the implied aspect {aspect}")

        parts.append(tmpl.format(**d_resolved))
    return " ; ".join(parts)


def _decode_target(raw: str, keys: list[str]) -> list[dict]:
    results = []
    for match in re.finditer(r"\[([^\[\]]+)\]", raw):
        values = [v.strip() for v in match.group(1).split(",")]
        if len(values) == len(keys):
            results.append(dict(zip(keys, values)))
    return results


def to_generative_format(canonical: dict, tasks: list[Task], output_format: str = "structured", infer_implicit: bool = False, category_set: list[str] = None, bare_prompt: bool = False, language: str = "en", include_categories: bool = False, nl_fraction: float = 1.0, ignore_order: bool = True) -> dict:
    if output_format == "natural-language":
        if ignore_order:
            keys = [k for k in CANONICAL_KEY_ORDER if k in {TASK_TO_KEY[t] for t in tasks}]
            task_list = [t for t in [Task.ASPECT, Task.SENTIMENT, Task.POLARITY, Task.CATEGORY] if TASK_TO_KEY[t] in keys]
        else:
            keys = [TASK_TO_KEY[t] for t in tasks]
            task_list = tasks
    else:
        keys = [TASK_TO_KEY[t] for t in tasks]
        task_list = tasks

    task_str = ", ".join(get_task_name(t.value, language) for t in task_list)

    # build input text with optional syntax enrichment
    if "_syntax_tokens" in canonical:
        input_sentence = " ".join(canonical["_syntax_tokens"])
    else:
        input_sentence = canonical["sentence"]

    labels = get_prompt_labels(language)
    input_text = f"{labels['task']}: {task_str}\n{labels['input']}: {input_sentence}"

    if "_syntax" in canonical:
        input_text += f"\n{labels['syntax']}: {canonical['_syntax']}"

    if include_categories and category_set and "category" in keys:
        translated_cats = [translate_to_output(c, "category", language) for c in category_set]
        input_text += f"\n{labels['categories']}: {', '.join(translated_cats)}"

    # only show output format when mixed training (0 < nl_fraction < 1)
    if 0 < nl_fraction < 1:
        output_label = labels['output_nl'] if output_format == 'natural-language' else labels['output_struct']
        input_text += f"\n{labels['output']}: {output_label}"

    if bare_prompt:
        input_text = input_sentence

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
        keys_ordered = tuple(keys) if not ignore_order else None
        target = _encode_target_nl(annotations, frozenset(keys), language=language, keys_ordered=keys_ordered)
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
    category_set: list[str] = None,
    bare_prompt: bool = False,
    language: str = "en",
    include_categories: bool = False,
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
            ordered = list(rng.choice(perms) if shuffle_tasks else task_group)
            # shuffle category order per example to prevent positional bias
            cats = list(category_set) if category_set else None
            if cats:
                rng.shuffle(cats)
            # when shuffle_tasks is active, use the actual task order for NL template selection
            use_ignore_order = not shuffle_tasks
            partitions[task_group].append(to_generative_format(canonical[idx], ordered, output_format=fmt, infer_implicit=infer_implicit, category_set=cats, bare_prompt=bare_prompt, language=language, include_categories=include_categories, nl_fraction=nl_fraction, ignore_order=use_ignore_order))
        start = end

    return partitions


def extract_categories(examples: list[dict]) -> list[str]:
    cats = set()
    for ex in examples:
        for ann in ex["annotations"]:
            c = ann.get("category")
            if c and c not in ("NONE", "IMPLICIT"):
                cats.add(c)
    return sorted(cats)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ABSADataset(Dataset):
    def __init__(self, examples: list[dict], tokenizer: AutoTokenizer, max_length: int = 256, structured_attention: bool = False):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.structured_attention = structured_attention

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
        result = {
            "input_ids":      inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels":         labels,
            "raw_target":     ex["target"],
            "raw_input":      ex["input"],
            "keys":           ",".join(ex.get("_keys", ["aspect", "sentiment", "polarity"])),
            "output_format":  ex.get("_format", "structured"),
        }
        if self.structured_attention:
            from src.data.syntax_mask import build_syntax_attention_mask
            mask_2d = build_syntax_attention_mask(ex["input"], self.tokenizer, self.max_length)
            if mask_2d is not None:
                result["attention_mask"] = mask_2d
        return result
