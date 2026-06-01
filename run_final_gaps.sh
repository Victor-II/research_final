#!/bin/bash
set -e

PYTHON="venv/bin/python"

echo "============================================================"
echo "  1. Retest: t5-nl-template-aug (SemEval, fixed config)"
echo "============================================================"
$PYTHON main.py --config config/overlays/t5-nl-template-aug.yaml --mode test \
  --checkpoint experiments/2026-05-26/t5-nl-template-aug/checkpoints/best.ckpt

echo ""
echo "============================================================"
echo "  2. Domain-mix experiments (Rest14 + DMASTE domain fraction)"
echo "============================================================"

for DOMAIN in beauty electronics fashion home all; do
  echo ""
  echo "------------------------------------------------------------"
  echo "  Running: t5-nl-domainmix-${DOMAIN}"
  echo "------------------------------------------------------------"
  $PYTHON main.py --config "config/overlays/t5-nl-domainmix-${DOMAIN}.yaml"
done

echo ""
echo "All gap-closing experiments complete."
