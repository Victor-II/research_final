#!/bin/bash
set -e

PYTHON="/home/victor-ii/research/research_final/venv/bin/python"

for name in nl-baseline nl-split nl-dep-compact nl-split-dep-compact; do
  ckpt=$(find experiments/ -path "*/$name/checkpoints/best.ckpt" | head -1)
  cfg=$(find experiments/ -path "*/$name/config.yaml" | head -1)
  echo "=== $name (beam=4) ==="
  $PYTHON main.py --config "$cfg" --mode test --checkpoint "$ckpt" \
    --set generation.num_beams=4 --set name="${name}-beam4"
done

echo "Done."
