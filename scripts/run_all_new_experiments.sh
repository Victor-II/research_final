#!/bin/bash
# Run all Gemma + RoBERTa experiments
# Expected total time: ~4 hours
set -e
source venv/bin/activate

echo "============================================================"
echo "  Starting all experiments: $(date)"
echo "============================================================"

# --- Gemma 0-shot (no retriever needed, fastest) ---
echo ""
echo "=== Gemma 0-shot experiments ==="

echo "--- [1/10] gemma-0shot-structured ---"
python main.py --mode train --config config/overlays/gemma-0shot-structured.yaml

echo "--- [2/10] gemma-0shot-structured-syntax ---"
python main.py --mode train --config config/overlays/gemma-0shot-structured-syntax.yaml

echo "--- [3/10] gemma-0shot-nl ---"
python main.py --mode train --config config/overlays/gemma-0shot-nl.yaml

echo "--- [4/10] gemma-0shot-nl-syntax ---"
python main.py --mode train --config config/overlays/gemma-0shot-nl-syntax.yaml

echo "--- [5/10] gemma-0shot-acos ---"
python main.py --mode train --config config/overlays/gemma-0shot-acos.yaml

# --- Gemma 6-shot (hybrid retrieval) ---
echo ""
echo "=== Gemma 6-shot experiments ==="

echo "--- [6/10] gemma-6shot-structured ---"
python main.py --mode train --config config/overlays/gemma-6shot-structured.yaml

echo "--- [7/10] gemma-6shot-structured-syntax ---"
python main.py --mode train --config config/overlays/gemma-6shot-structured-syntax.yaml

echo "--- [8/10] gemma-6shot-nl ---"
python main.py --mode train --config config/overlays/gemma-6shot-nl.yaml

echo "--- [9/10] gemma-6shot-nl-syntax ---"
python main.py --mode train --config config/overlays/gemma-6shot-nl-syntax.yaml

echo "--- [10/10] gemma-6shot-acos ---"
python main.py --mode train --config config/overlays/gemma-6shot-acos.yaml

# --- RoBERTa ---
echo ""
echo "=== RoBERTa experiments ==="

echo "--- [1/3] roberta-baseline ---"
python main.py --mode train --config config/overlays/roberta-baseline.yaml

echo "--- [2/3] roberta-baseline-syntax ---"
python main.py --mode train --config config/overlays/roberta-baseline-syntax.yaml

echo "--- [3/3] roberta-rest14 ---"
python main.py --mode train --config config/overlays/roberta-rest14.yaml

echo ""
echo "============================================================"
echo "  All experiments complete: $(date)"
echo "============================================================"
