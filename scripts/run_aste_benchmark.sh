#!/bin/bash
# ASTE benchmark: per-dataset training & testing
# Runs 8 experiments sequentially (4 datasets × 2 configs)

PYTHON=/home/victor-ii/research/research_final/venv/bin/python

echo "=== ASTE Benchmark: NL baseline ==="
for ds in rest14 rest15 rest16 laptop14; do
    echo "--- aste-${ds}-nl ---"
    $PYTHON main.py --config config/overlays/aste-${ds}-nl.yaml
done

echo "=== ASTE Benchmark: NL + pos-aux ==="
for ds in rest14 rest15 rest16 laptop14; do
    echo "--- aste-${ds}-nl-pos-aux ---"
    $PYTHON main.py --config config/overlays/aste-${ds}-nl-pos-aux.yaml
done

echo "=== All done ==="
