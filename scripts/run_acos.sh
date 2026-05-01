#!/bin/bash
set -e

PYTHON="/home/victor-ii/research/research_final/venv/bin/python"

$PYTHON main.py --config config/overlays/acos-laptop-split-ours.yaml
$PYTHON main.py --config config/overlays/acos-laptop-split-star.yaml
$PYTHON main.py --config config/overlays/acos-rest16-split-ours.yaml
$PYTHON main.py --config config/overlays/acos-rest16-split-star.yaml

echo "Done. STAR reference: ACOS-Laptop 45.15 F1, ACOS-Rest16 61.07 F1"
