#!/bin/bash
set -e

# Old settings: linear schedule, no label smoothing
python main.py --config config/overlays/sanity-baseline-no-ls.yaml

# New settings: cosine schedule, label smoothing 0.1
python main.py --config config/overlays/sanity-baseline.yaml

echo "Done. Compare with:"
echo "  python main.py --mode aggregate --experiments sanity-baseline-no-ls sanity-baseline --scope aspect+sentiment+polarity"
