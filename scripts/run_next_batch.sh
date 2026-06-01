#!/bin/bash
# Next batch: template-aug → DMASTE → domain-mix
set -e

source /home/victor-ii/research/research_final/venv/bin/activate
cd /home/victor-ii/research/research_final

ollama stop qwen2.5-coder:32b 2>/dev/null || true
ollama stop gemma2:27b 2>/dev/null || true
ollama stop command-r:latest 2>/dev/null || true
sleep 2

# --- Template augmentation (SemEval) ---

echo "=== T5 NL + template-aug (Rest14 → all) ==="
python main.py --config config/overlays/t5-nl-template-aug.yaml

# --- DMASTE cross-domain ---

echo "=== DMASTE: electronics → targets (structured) ==="
python main.py --config config/overlays/dmaste-electronics-structured.yaml

echo "=== DMASTE: electronics → targets (NL) ==="
python main.py --config config/overlays/dmaste-electronics-nl.yaml

echo "=== DMASTE: electronics → targets (NL + dep-compact) ==="
python main.py --config config/overlays/dmaste-electronics-nl-dep-compact.yaml

echo "=== DMASTE: electronics → targets (NL + pos-compact) ==="
python main.py --config config/overlays/dmaste-electronics-nl-pos-compact.yaml

echo "=== DMASTE: electronics → targets (NL + template-aug) ==="
python main.py --config config/overlays/dmaste-electronics-nl-template-aug.yaml

echo "=== DMASTE: electronics → targets (RoBERTa) ==="
python main.py --config config/overlays/dmaste-electronics-roberta.yaml

echo "=== DMASTE: all sources → targets (structured) ==="
python main.py --config config/overlays/dmaste-allsource-structured.yaml

echo "=== DMASTE: all sources → targets (NL) ==="
python main.py --config config/overlays/dmaste-allsource-nl.yaml

echo "=== DMASTE: all sources → targets (NL + dep-compact) ==="
python main.py --config config/overlays/dmaste-allsource-nl-dep-compact.yaml

echo "=== DMASTE: all sources → targets (NL + pos-compact) ==="
python main.py --config config/overlays/dmaste-allsource-nl-pos-compact.yaml

echo "=== DMASTE: all sources → targets (NL + template-aug) ==="
python main.py --config config/overlays/dmaste-allsource-nl-template-aug.yaml

echo "=== DMASTE: all sources → targets (RoBERTa) ==="
python main.py --config config/overlays/dmaste-allsource-roberta.yaml

# --- Domain mixing (Rest14 + DMASTE → Laptop14) ---

echo "=== Domain mix: Rest14 + beauty ==="
python main.py --config config/overlays/t5-nl-domainmix-beauty.yaml

echo "=== Domain mix: Rest14 + electronics ==="
python main.py --config config/overlays/t5-nl-domainmix-electronics.yaml

echo "=== Domain mix: Rest14 + fashion ==="
python main.py --config config/overlays/t5-nl-domainmix-fashion.yaml

echo "=== Domain mix: Rest14 + home ==="
python main.py --config config/overlays/t5-nl-domainmix-home.yaml

echo "=== Domain mix: Rest14 + all DMASTE ==="
python main.py --config config/overlays/t5-nl-domainmix-all.yaml

echo "=== All experiments complete ==="
