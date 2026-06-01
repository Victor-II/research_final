#!/bin/bash
set -e

PYTHON="venv/bin/python"

echo "============================================================"
echo "  1. STAR-style data multiplication (train + test)"
echo "============================================================"
$PYTHON main.py --config config/overlays/t5-nl-star.yaml

echo ""
echo "============================================================"
echo "  2. MvP-style multi-prompt voting (test only, template-aug checkpoint)"
echo "============================================================"
$PYTHON main.py --config config/overlays/t5-nl-template-aug.yaml --mode mvp \
  --checkpoint experiments/2026-05-26/t5-nl-template-aug/checkpoints/best.ckpt

echo ""
echo "============================================================"
echo "  3. MvP-style voting on nl-baseline checkpoint for comparison"
echo "============================================================"
$PYTHON main.py --config config/overlays/t5-nl-baseline.yaml --mode mvp \
  --checkpoint experiments/2026-05-22/t5-nl-baseline-s42/checkpoints/best.ckpt

echo ""
echo "All STAR/MvP experiments complete."
