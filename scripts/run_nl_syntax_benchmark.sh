#!/bin/bash
# NL syntax benchmark: 4 configs × 4 datasets = 16 experiments
# All use NL output, max_length=512, 50 epochs
# batch_size=8 + accumulate=4 to fit 512 seq len in memory

PYTHON=/home/victor-ii/research/research_final/venv/bin/python
EPOCHS=50
ML=512
BS=8
AG=4
COMMON="--set model.max_length=$ML --set trainer.max_epochs=$EPOCHS --set model.batch_size=$BS --set trainer.accumulate_grad_batches=$AG"

DATASETS=(rest14 rest15 rest16 laptop14)

# --- dep-nl enrichment only ---
for ds in "${DATASETS[@]}"; do
    echo "=== aste-${ds}-nl-dep-nl ==="
    $PYTHON main.py --config config/overlays/aste-${ds}-nl.yaml \
        --set data.syntax_enrichment=dep-nl \
        $COMMON \
        --set name=aste-${ds}-nl-dep-nl
done

# --- pos-nl enrichment only ---
for ds in "${DATASETS[@]}"; do
    echo "=== aste-${ds}-nl-pos-nl ==="
    $PYTHON main.py --config config/overlays/aste-${ds}-nl.yaml \
        --set data.syntax_enrichment=pos-nl \
        $COMMON \
        --set name=aste-${ds}-nl-pos-nl
done

# --- dep-nl enrichment + dep-nl auxiliary ---
for ds in "${DATASETS[@]}"; do
    echo "=== aste-${ds}-nl-dep-nl-aux ==="
    $PYTHON main.py --config config/overlays/aste-${ds}-nl.yaml \
        --set data.syntax_enrichment=dep-nl \
        --set data.syntax_auxiliary_fraction=0.1 \
        --set "data.syntax_auxiliary_tasks=[dep-nl]" \
        $COMMON \
        --set name=aste-${ds}-nl-dep-nl-aux
done

# --- pos-nl enrichment + pos-nl auxiliary ---
for ds in "${DATASETS[@]}"; do
    echo "=== aste-${ds}-nl-pos-nl-aux ==="
    $PYTHON main.py --config config/overlays/aste-${ds}-nl.yaml \
        --set data.syntax_enrichment=pos-nl \
        --set data.syntax_auxiliary_fraction=0.1 \
        --set "data.syntax_auxiliary_tasks=[pos-nl]" \
        $COMMON \
        --set name=aste-${ds}-nl-pos-nl-aux
done

echo "=== All done ==="
