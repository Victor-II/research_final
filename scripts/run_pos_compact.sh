#!/bin/bash
# Multi-seed pos-compact experiments: 2 setups × 5 seeds = 10 runs
set -e

source /home/victor-ii/research/research_final/venv/bin/activate
cd /home/victor-ii/research/research_final

ollama stop qwen2.5-coder:32b 2>/dev/null || true
ollama stop gemma2:27b 2>/dev/null || true
ollama stop command-r:latest 2>/dev/null || true
sleep 2

SEEDS="42 123 456 789 1337"

for seed in $SEEDS; do
    echo "=== t5-nl-pos-compact (Rest14 only, seed=${seed}) ==="
    python main.py --config config/overlays/t5-nl-pos-compact-s${seed}.yaml

    echo "=== t5-nl-pos-compact (all-rest, seed=${seed}) ==="
    python main.py --config config/overlays/t5-nl-pos-compact-allrest-s${seed}.yaml
done

echo "=== All pos-compact experiments complete ==="
