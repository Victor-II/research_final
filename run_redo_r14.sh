#!/bin/bash
set -e

echo "=== Curriculum: overlap (Rest14 only) ==="
python main.py --config config/overlays/t5-nl-cur-overlap-r14.yaml --mode train

echo "=== Curriculum: fast-ramp (Rest14 only) ==="
python main.py --config config/overlays/t5-nl-cur-fast-ramp-r14.yaml --mode train

echo "=== Curriculum: sandwich (Rest14 only) ==="
python main.py --config config/overlays/t5-nl-cur-sandwich-r14.yaml --mode train

echo "=== Masking 20% + dep (Rest14 only) ==="
python main.py --config config/overlays/t5-nl-mask20-dep-r14.yaml --mode train

echo "=== Masking 20% no dep (Rest14 only) ==="
python main.py --config config/overlays/t5-nl-mask20-r14.yaml --mode train

echo "=== All done ==="
