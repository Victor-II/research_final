#!/bin/bash
# Run all pending experiments: RoBERTa, XLM-R, then ollama models
set -e

source /home/victor-ii/research/research_final/venv/bin/activate
cd /home/victor-ii/research/research_final

# Free GPU memory from ollama before training
ollama stop qwen2.5-coder:32b 2>/dev/null || true
ollama stop gemma2:27b 2>/dev/null || true
ollama stop command-r:latest 2>/dev/null || true
sleep 2

echo "=== RoBERTa baseline (English ASTE) ==="
python main.py --config config/overlays/roberta-baseline.yaml

echo "=== XLM-RoBERTa baseline (English ASTE) ==="
python main.py --config config/overlays/xlmr-baseline.yaml

echo "=== XLM-RoBERTa Romanian classifier ==="
python main.py --config config/overlays/xlmr-ro-baseline.yaml

echo "=== Ollama: gemma2:27b 0-shot ==="
python main.py --config config/overlays/ollama-gemma27b-0shot.yaml

echo "=== Ollama: gemma2:27b 6-shot ==="
python main.py --config config/overlays/ollama-gemma27b-6shot.yaml

echo "=== Ollama: qwen2.5-coder:32b 0-shot ==="
python main.py --config config/overlays/ollama-qwen32b-0shot.yaml

echo "=== Ollama: qwen2.5-coder:32b 6-shot ==="
python main.py --config config/overlays/ollama-qwen32b-6shot.yaml

echo "=== Ollama: command-r 0-shot ==="
python main.py --config config/overlays/ollama-commandr-0shot.yaml

echo "=== Ollama: command-r 6-shot ==="
python main.py --config config/overlays/ollama-commandr-6shot.yaml

echo "=== All experiments complete ==="
