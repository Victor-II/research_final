#!/bin/bash
set -e

PYTHON="/home/victor-ii/research/research_final/venv/bin/python"

$PYTHON main.py --config config/overlays/cur-overlap.yaml
$PYTHON main.py --config config/overlays/cur-overlap-dep.yaml
$PYTHON main.py --config config/overlays/cur-fast-ramp.yaml
$PYTHON main.py --config config/overlays/cur-fast-ramp-dep.yaml
$PYTHON main.py --config config/overlays/cur-sandwich.yaml
$PYTHON main.py --config config/overlays/cur-sandwich-dep.yaml

echo "Done. Reference: nl-baseline 0.7166/0.5211, nl-dep-compact 0.7083/0.5414"
