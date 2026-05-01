#!/bin/bash
set -e

PYTHON="/home/victor-ii/research/research_final/venv/bin/python"

$PYTHON main.py --config config/overlays/aste-baseline-dep-inline.yaml
$PYTHON main.py --config config/overlays/aste-split-nl10-dep-inline.yaml
$PYTHON main.py --config config/overlays/aste-baseline-pos-inline.yaml
$PYTHON main.py --config config/overlays/aste-split-nl10-pos-inline.yaml

echo "Done. Compare with:"
echo "  Reference: aste-baseline 0.6798/0.4296, aste-split-nl10 0.7135/0.4502"
