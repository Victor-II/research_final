#!/bin/bash
# Run RoBERTa span extraction experiments
# Usage: bash scripts/run_roberta_experiments.sh

set -e
source venv/bin/activate

echo "=== RoBERTa Span Extraction Experiments ==="
echo ""

# OOD setup: train on Rest14+15+16, test on Rest14 (ID) + Laptop14 (OOD)
echo "--- RoBERTa OOD baseline (Rest14+15+16 → Laptop14) ---"
python main.py --mode train --config config/overlays/roberta-baseline.yaml

# Per-dataset: train on Rest14, test on Rest14 + Laptop14
echo "--- RoBERTa per-dataset (Rest14) ---"
python main.py --mode train --config config/overlays/roberta-rest14.yaml

echo ""
echo "=== All RoBERTa experiments complete ==="
