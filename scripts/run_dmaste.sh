#!/bin/bash
# DMASTE cross-domain experiments: single-source and multi-source
set -e

source /home/victor-ii/research/research_final/venv/bin/activate
cd /home/victor-ii/research/research_final

ollama stop qwen2.5-coder:32b 2>/dev/null || true
ollama stop gemma2:27b 2>/dev/null || true
ollama stop command-r:latest 2>/dev/null || true
sleep 2

echo "=== DMASTE: electronics → targets (structured) ==="
python main.py --config config/overlays/dmaste-electronics-structured.yaml

echo "=== DMASTE: electronics → targets (NL) ==="
python main.py --config config/overlays/dmaste-electronics-nl.yaml

echo "=== DMASTE: electronics → targets (NL + dep-compact) ==="
python main.py --config config/overlays/dmaste-electronics-nl-dep-compact.yaml

echo "=== DMASTE: electronics → targets (NL + pos-compact) ==="
python main.py --config config/overlays/dmaste-electronics-nl-pos-compact.yaml

echo "=== DMASTE: all sources → targets (structured) ==="
python main.py --config config/overlays/dmaste-allsource-structured.yaml

echo "=== DMASTE: all sources → targets (NL) ==="
python main.py --config config/overlays/dmaste-allsource-nl.yaml

echo "=== DMASTE: all sources → targets (NL + dep-compact) ==="
python main.py --config config/overlays/dmaste-allsource-nl-dep-compact.yaml

echo "=== DMASTE: all sources → targets (NL + pos-compact) ==="
python main.py --config config/overlays/dmaste-allsource-nl-pos-compact.yaml

echo "=== DMASTE: electronics → targets (NL + template-aug) ==="
python main.py --config config/overlays/dmaste-electronics-nl-template-aug.yaml

echo "=== DMASTE: all sources → targets (NL + template-aug) ==="
python main.py --config config/overlays/dmaste-allsource-nl-template-aug.yaml

echo "=== DMASTE: electronics → targets (RoBERTa) ==="
python main.py --config config/overlays/dmaste-electronics-roberta.yaml

echo "=== DMASTE: all sources → targets (RoBERTa) ==="
python main.py --config config/overlays/dmaste-allsource-roberta.yaml

echo "=== All DMASTE experiments complete ==="
