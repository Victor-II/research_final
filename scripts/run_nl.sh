#!/bin/bash
set -e

PYTHON="/home/victor-ii/research/research_final/venv/bin/python"

echo "=== ASTE experiments ==="
$PYTHON main.py --config config/overlays/aste-baseline.yaml
$PYTHON main.py --config config/overlays/aste-baseline-nl10.yaml
$PYTHON main.py --config config/overlays/aste-baseline-nl30.yaml
$PYTHON main.py --config config/overlays/aste-baseline-nl50.yaml
$PYTHON main.py --config config/overlays/aste-split.yaml
$PYTHON main.py --config config/overlays/aste-split-nl10.yaml
$PYTHON main.py --config config/overlays/aste-split-nl30.yaml
$PYTHON main.py --config config/overlays/aste-split-nl50.yaml

echo "=== ACOS experiments ==="
$PYTHON main.py --config config/overlays/acos-baseline.yaml
$PYTHON main.py --config config/overlays/acos-baseline-nl10.yaml
$PYTHON main.py --config config/overlays/acos-baseline-nl30.yaml
$PYTHON main.py --config config/overlays/acos-baseline-nl50.yaml
$PYTHON main.py --config config/overlays/acos-split.yaml
$PYTHON main.py --config config/overlays/acos-split-nl10.yaml
$PYTHON main.py --config config/overlays/acos-split-nl30.yaml
$PYTHON main.py --config config/overlays/acos-split-nl50.yaml

echo "All done."
