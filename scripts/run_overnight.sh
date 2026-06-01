#!/bin/bash
# Overnight run: T5 split reruns + Romanian ollama experiments
set -e

source /home/victor-ii/research/research_final/venv/bin/activate
cd /home/victor-ii/research/research_final

# Free GPU memory from ollama before T5 training
ollama stop qwen2.5-coder:32b 2>/dev/null || true
ollama stop gemma2:27b 2>/dev/null || true
ollama stop command-r:latest 2>/dev/null || true
sleep 2

# --- T5 split experiments (fixed) ---

echo "=== T5 NL + split ==="
python main.py --config config/overlays/t5-nl-split.yaml

echo "=== T5 NL + split + dep-compact ==="
python main.py --config config/overlays/t5-nl-split-dep-compact.yaml

# --- Romanian ollama experiments ---

echo "=== Ollama: gemma2:27b Romanian 0-shot ==="
python main.py --config config/overlays/ollama-gemma27b-ro-0shot.yaml

echo "=== Ollama: gemma2:27b Romanian 6-shot ==="
python main.py --config config/overlays/ollama-gemma27b-ro-6shot.yaml

echo "=== Ollama: qwen2.5-coder:32b Romanian 0-shot ==="
python main.py --config config/overlays/ollama-qwen32b-ro-0shot.yaml

echo "=== Ollama: qwen2.5-coder:32b Romanian 6-shot ==="
python main.py --config config/overlays/ollama-qwen32b-ro-6shot.yaml

echo "=== Ollama: command-r Romanian 0-shot ==="
python main.py --config config/overlays/ollama-commandr-ro-0shot.yaml

echo "=== Ollama: command-r Romanian 6-shot ==="
python main.py --config config/overlays/ollama-commandr-ro-6shot.yaml

echo "=== All overnight experiments complete ==="
