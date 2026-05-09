#!/bin/bash
# ACOS experiment plan — step 1: structured vs NL
set -e

VENV="/home/victor-ii/research/research_final/venv"
source "$VENV/bin/activate"

echo "=========================================="
echo "  Step 1: Structured vs NL"
echo "=========================================="

configs=(
    "config/overlays/acos-rest16-quad-structured.yaml"
    "config/overlays/acos-rest16-quad-nl-baseline.yaml"
)

for cfg in "${configs[@]}"; do
    name=$(python -c "import yaml; print(yaml.safe_load(open('$cfg'))['name'])")
    echo ""
    echo "=========================================="
    echo "  Starting: $name"
    echo "=========================================="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python main.py --config "$cfg"
    echo "  Finished: $name"
done

echo ""
echo "Step 1 complete. Compare results, then run step 2."
