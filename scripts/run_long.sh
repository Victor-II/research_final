#!/bin/bash
set -e

VENV="/home/victor-ii/research/research_final/venv/bin/python"

# Phase 1: baseline + task splits
echo "=== Phase 1: Baseline + Task Splits ==="
$VENV main.py --config config/overlays/baseline.yaml
$VENV main.py --config config/overlays/split-10-90.yaml
$VENV main.py --config config/overlays/split-30-70.yaml
$VENV main.py --config config/overlays/split-50-50.yaml
$VENV main.py --config config/overlays/split-70-30.yaml
$VENV main.py --config config/overlays/split-90-10.yaml

# Phase 2: find best split by average OOD triplet F1, generate masking overlays, run them
echo "=== Phase 2: Picking best split and running masking ==="
$VENV - <<'PYSCRIPT'
import json, yaml
from pathlib import Path

experiments_dir = Path("experiments")
names = ["baseline", "split-10-90", "split-30-70", "split-50-50", "split-70-30", "split-90-10"]

best_name = None
best_avg = -1

for name in names:
    # search in date subdirs
    results_file = None
    for sub in sorted(experiments_dir.iterdir()):
        candidate = sub / name / "results" / "results.json"
        if candidate.exists():
            results_file = candidate
            break
    if not results_file:
        # try top-level
        candidate = experiments_dir / name / "results" / "results.json"
        if candidate.exists():
            results_file = candidate
    if not results_file:
        print(f"  {name}: no results found, skipping")
        continue

    with open(results_file) as f:
        data = json.load(f)

    f1s = []
    for entry in data.get("test", []):
        triplet = entry.get("aspect+sentiment+polarity", {}).get("micro", {}).get("f1")
        if triplet is not None:
            f1s.append(triplet)

    avg = sum(f1s) / len(f1s) if f1s else 0
    print(f"  {name}: avg OOD triplet F1 = {avg:.4f} ({len(f1s)} datasets)")
    if avg > best_avg:
        best_avg = avg
        best_name = name

print(f"\nBest: {best_name} (avg F1 = {best_avg:.4f})")

# Load the best experiment's config to get its task partition
best_cfg_file = None
for sub in sorted(experiments_dir.iterdir()):
    candidate = sub / best_name / "config.yaml"
    if candidate.exists():
        best_cfg_file = candidate
        break
if not best_cfg_file:
    best_cfg_file = experiments_dir / best_name / "config.yaml"

with open(best_cfg_file) as f:
    best_cfg = yaml.safe_load(f)

test_block = {
    "datasets": [
        {"data": f"downloads/silviolima/{d}.json", "tasks": ["aspect", "sentiment", "polarity"]}
        for d in ["laptop", "beauty", "book", "electronics", "fashion", "grocery", "home", "pet", "toy"]
    ]
}

for frac in [0.10, 0.25, 0.50, 0.75]:
    name = f"mask-f{int(frac*100)}"
    cfg = {
        "name": name,
        "model": {"name": "google/flan-t5-base"},
        "data": {
            "train_file": "downloads/silviolima/restaurant.json",
            "filter_implicit": True,
            "tasks_partition": best_cfg["data"]["tasks_partition"],
            "shuffle_tasks": best_cfg["data"].get("shuffle_tasks", False),
            "augmentation": {
                "mask_aspects": {
                    "fraction": frac,
                    "replace": True,
                }
            },
        },
        "eval": {"val_split": 0.1},
        "test": {
            "datasets": test_block["datasets"],
        },
    }
    overlay_path = f"config/overlays/{name}.yaml"
    with open(overlay_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Created {overlay_path}")
PYSCRIPT

echo "=== Phase 2: Running masking experiments ==="
$VENV main.py --config config/overlays/mask-f10.yaml
$VENV main.py --config config/overlays/mask-f25.yaml
$VENV main.py --config config/overlays/mask-f50.yaml
$VENV main.py --config config/overlays/mask-f75.yaml

echo "All done."
