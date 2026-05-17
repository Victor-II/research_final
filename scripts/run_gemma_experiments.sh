#!/bin/bash
# Run all Gemma 4B inference experiments
# Usage: bash scripts/run_gemma_experiments.sh
#
# Prerequisites:
#   1. Accept Gemma license at https://huggingface.co/google/gemma-3-4b-it
#   2. Login: huggingface-cli login
#   (or set HF_TOKEN environment variable)

set -e
source venv/bin/activate

echo "=== Gemma 4B Experiments ==="
echo ""

# ASTE experiments
echo "--- ASTE: 0-shot structured ---"
python main.py --mode train --config config/overlays/gemma-0shot-structured.yaml

echo "--- ASTE: 0-shot NL ---"
python main.py --mode train --config config/overlays/gemma-0shot-nl.yaml

echo "--- ASTE: 6-shot structured ---"
python main.py --mode train --config config/overlays/gemma-6shot-structured.yaml

echo "--- ASTE: 6-shot NL ---"
python main.py --mode train --config config/overlays/gemma-6shot-nl.yaml

echo "--- ASTE: 6-shot NL + syntax ---"
python main.py --mode train --config config/overlays/gemma-6shot-nl-syntax.yaml

# ACOS experiments
echo "--- ACOS: 0-shot ---"
python main.py --mode train --config config/overlays/gemma-0shot-acos.yaml

echo "--- ACOS: 6-shot ---"
python main.py --mode train --config config/overlays/gemma-6shot-acos.yaml

echo ""
echo "=== All Gemma experiments complete ==="
