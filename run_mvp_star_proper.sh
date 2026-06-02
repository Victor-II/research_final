#!/bin/bash
set -e

PYTHON="venv/bin/python"

echo "============================================================"
echo "  1. MvP proper (markers + 5 orderings + voting)"
echo "============================================================"
$PYTHON main.py --config config/overlays/t5-mvp-proper.yaml

echo ""
echo "============================================================"
echo "  2. MvP voting at inference (threshold=3)"
echo "============================================================"
$PYTHON main.py --config config/overlays/t5-mvp-proper.yaml --mode mvp \
  --checkpoint experiments/2026-06-02/t5-mvp-proper/checkpoints/best.ckpt

echo ""
echo "============================================================"
echo "  3. STAR proper (markers + pairwise + balanced loss)"
echo "============================================================"
$PYTHON main.py --config config/overlays/t5-star-proper.yaml

echo ""
echo "============================================================"
echo "  4. STAR voting at inference (threshold=3)"
echo "============================================================"
$PYTHON main.py --config config/overlays/t5-star-proper.yaml --mode mvp \
  --checkpoint experiments/2026-06-02/t5-star-proper/checkpoints/best.ckpt

echo ""
echo "All proper MvP/STAR experiments complete."
