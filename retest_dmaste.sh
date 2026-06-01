#!/bin/bash
set -e

PYTHON="venv/bin/python"

declare -A EXPERIMENTS=(
  # electronics (single-source)
  ["dmaste-electronics-nl"]="experiments/2026-05-26/dmaste-electronics-nl/checkpoints/best.ckpt"
  ["dmaste-electronics-nl-dep-compact"]="experiments/2026-05-26/dmaste-electronics-nl-dep-compact/checkpoints/best.ckpt"
  ["dmaste-electronics-structured"]="experiments/2026-05-26/dmaste-electronics-structured/checkpoints/best.ckpt"
  ["dmaste-electronics-nl-pos-compact"]="experiments/2026-05-27/dmaste-electronics-nl-pos-compact/checkpoints/best.ckpt"
  ["dmaste-electronics-nl-template-aug"]="experiments/2026-05-27/dmaste-electronics-nl-template-aug/checkpoints/best.ckpt"
  ["dmaste-electronics-roberta"]="experiments/2026-05-27/dmaste-electronics-roberta/checkpoints/best.ckpt"
  # allsource (multi-source)
  ["dmaste-allsource-nl"]="experiments/2026-05-27/dmaste-allsource-nl/checkpoints/best.ckpt"
  ["dmaste-allsource-nl-dep-compact"]="experiments/2026-05-27/dmaste-allsource-nl-dep-compact/checkpoints/best.ckpt"
  ["dmaste-allsource-nl-pos-compact"]="experiments/2026-05-27/dmaste-allsource-nl-pos-compact/checkpoints/best.ckpt"
  ["dmaste-allsource-nl-template-aug"]="experiments/2026-05-27/dmaste-allsource-nl-template-aug/checkpoints/best.ckpt"
  ["dmaste-allsource-roberta"]="experiments/2026-05-27/dmaste-allsource-roberta/checkpoints/best.ckpt"
  ["dmaste-allsource-structured"]="experiments/2026-05-27/dmaste-allsource-structured/checkpoints/best.ckpt"
)

TOTAL=${#EXPERIMENTS[@]}
COUNT=0

for NAME in "${!EXPERIMENTS[@]}"; do
  CKPT="${EXPERIMENTS[$NAME]}"
  CONFIG="config/overlays/${NAME}.yaml"
  COUNT=$((COUNT + 1))

  echo ""
  echo "============================================================"
  echo "  [$COUNT/$TOTAL] Testing: $NAME"
  echo "  Checkpoint: $CKPT"
  echo "============================================================"
  echo ""

  $PYTHON main.py --config "$CONFIG" --mode test --checkpoint "$CKPT"
done

echo ""
echo "All $TOTAL dmaste experiments retested."
