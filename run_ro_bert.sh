#!/bin/bash
set -e

echo "=== RoBERT-base (Romanian monolingual, 114M) ==="
python main.py --config config/overlays/robert-ro-baseline.yaml --mode train

echo "=== bert-base-romanian-cased-v1 (Romanian monolingual, 110M) ==="
python main.py --config config/overlays/bert-ro-baseline.yaml --mode train

echo "=== Done ==="
