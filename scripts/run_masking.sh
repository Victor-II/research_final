#!/bin/bash
set -e

PYTHON="/home/victor-ii/research/research_final/venv/bin/python"

$PYTHON main.py --config config/overlays/nl-dep-mask10.yaml
$PYTHON main.py --config config/overlays/nl-dep-mask25-replace.yaml
$PYTHON main.py --config config/overlays/nl-dep-mask50-replace.yaml
$PYTHON main.py --config config/overlays/nl-mask25-replace.yaml

echo "Done. Reference: nl-dep-compact 0.7083/0.5414"
