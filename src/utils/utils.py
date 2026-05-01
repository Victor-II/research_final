import copy
import os
from pathlib import Path

import yaml

from constants import REPLACE_KEYS, BASE_CONFIG, EXPERIMENTS_DIR


def deep_merge(base: dict, overlay: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in REPLACE_KEYS:
            result[k] = copy.deepcopy(v)
        elif k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def set_nested(d: dict, dotted_key: str, value: str):
    keys = dotted_key.split(".")
    for k in keys[:-1]:
        if k not in d or d[k] is None or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    try:
        parsed = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed = value
    d[keys[-1]] = parsed


def resolve_output_dir(cfg: dict) -> Path:
    from datetime import date
    resolved = os.environ.get("_ABSA_OUTPUT_DIR")
    if resolved:
        return Path(resolved)
    name = cfg.get("name", "unnamed")
    day_dir = EXPERIMENTS_DIR / date.today().isoformat()
    base = day_dir / name
    if not base.exists():
        output_dir = base
    else:
        i = 2
        while (day_dir / f"{name}_{i}").exists():
            i += 1
        output_dir = day_dir / f"{name}_{i}"
    os.environ["_ABSA_OUTPUT_DIR"] = str(output_dir)
    return output_dir


def resolve_config(config_path: str | None, overrides: list[str]) -> dict:
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    if config_path:
        with open(config_path) as f:
            overlay = yaml.safe_load(f)
        cfg = deep_merge(cfg, overlay)

    for item in overrides:
        key, _, val = item.partition("=")
        set_nested(cfg, key.strip(), val.strip())

    return cfg
