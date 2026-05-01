#!/bin/bash
set -e

PYTHON="/home/victor-ii/research/research_final/venv/bin/python"
CKPT=$(find experiments/ -path "*/nl-dep-compact/checkpoints/best.ckpt" | head -1)
CFG=$(find experiments/ -path "*/nl-dep-compact/config.yaml" | head -1)

echo "=== Temperature 0.7 ==="
$PYTHON main.py --config "$CFG" --mode test --checkpoint "$CKPT" \
  --set generation.do_sample=true --set generation.temperature=0.7 \
  --set name=nl-dep-compact-t07

echo "=== Temperature 0.9 ==="
$PYTHON main.py --config "$CFG" --mode test --checkpoint "$CKPT" \
  --set generation.do_sample=true --set generation.temperature=0.9 \
  --set name=nl-dep-compact-t09

echo "=== Voting: 5 seq, t=0.8, threshold=2 ==="
$PYTHON main.py --config "$CFG" --mode test --checkpoint "$CKPT" \
  --set generation.do_sample=true --set generation.temperature=0.8 \
  --set generation.num_return_sequences=5 --set generation.vote_threshold=2 \
  --set name=nl-dep-compact-vote5-t2

echo "=== Voting: 5 seq, t=0.8, threshold=3 ==="
$PYTHON main.py --config "$CFG" --mode test --checkpoint "$CKPT" \
  --set generation.do_sample=true --set generation.temperature=0.8 \
  --set generation.num_return_sequences=5 --set generation.vote_threshold=3 \
  --set name=nl-dep-compact-vote5-t3

echo "Done."
