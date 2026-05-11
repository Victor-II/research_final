#!/bin/bash
source /home/victor-ii/research/research_final/venv/bin/activate

python main.py --config config/overlays/ro-baseline.yaml
python main.py --config config/overlays/ro-split.yaml
