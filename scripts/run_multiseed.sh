#!/bin/bash
# Multi-seed experiments: 4 configs × 2 setups × 5 seeds = 40 runs
set -e

source /home/victor-ii/research/research_final/venv/bin/activate
cd /home/victor-ii/research/research_final

ollama stop qwen2.5-coder:32b 2>/dev/null || true
ollama stop gemma2:27b 2>/dev/null || true
ollama stop command-r:latest 2>/dev/null || true
sleep 2

SEEDS="42 123 456 789 1337"
CONFIGS="t5-structured-baseline t5-nl-baseline t5-nl-dep-compact t5-nl-split"

for cfg in $CONFIGS; do
    for seed in $SEEDS; do
        echo "=== ${cfg} (Rest14 only, seed=${seed}) ==="
        python main.py --config config/overlays/${cfg}-s${seed}.yaml

        echo "=== ${cfg} (all-rest, seed=${seed}) ==="
        python main.py --config config/overlays/${cfg}-allrest-s${seed}.yaml
    done
done

echo "=== All multi-seed experiments complete ==="
