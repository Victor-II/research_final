#!/bin/bash
# Run T5 split experiments (fixed configs with 50% full task)
set -e

source /home/victor-ii/research/research_final/venv/bin/activate
cd /home/victor-ii/research/research_final

ollama stop qwen2.5-coder:32b 2>/dev/null || true
ollama stop gemma2:27b 2>/dev/null || true
ollama stop command-r:latest 2>/dev/null || true
sleep 2

echo "=== T5 NL + split ==="
python main.py --config config/overlays/t5-nl-split.yaml

echo "=== T5 NL + split + dep-compact ==="
python main.py --config config/overlays/t5-nl-split-dep-compact.yaml

echo "=== Split experiments complete ==="
