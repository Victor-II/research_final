#!/bin/bash
set -e

PYTHON="/home/victor-ii/research/research_final/venv/bin/python"
BASE="--config config/overlays/semeval_rest_base.yaml"
LOG="experiments/experiment_log.txt"

run_exp() {
    local name="$1"
    shift
    echo "========================================" | tee -a "$LOG"
    echo "[$(date)] Starting: $name" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    if $PYTHON main.py $BASE "$@" 2>&1 | tee -a "$LOG"; then
        echo "[$(date)] DONE: $name" | tee -a "$LOG"
    else
        echo "[$(date)] FAILED: $name (exit code $?)" | tee -a "$LOG"
    fi
    # cooldown: wait if GPU temp > 80C
    if command -v nvidia-smi &>/dev/null; then
        while true; do
            TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -rn | head -1)
            if [ "$TEMP" -lt 75 ]; then
                break
            fi
            echo "[$(date)] GPU temp ${TEMP}C, cooling down..." | tee -a "$LOG"
            sleep 30
        done
    fi
    echo "" | tee -a "$LOG"
}

echo "Experiment run started at $(date)" > "$LOG"

# ============================================================
# Phase 1: Task Split Ablation (no augmentation)
# ============================================================

run_exp "split-90-10" \
    --set name=split-90-10 \
    --set data.shuffle_tasks=true \
    --set data.tasks_partition.aspect=0.30 \
    --set data.tasks_partition.sentiment=0.30 \
    --set data.tasks_partition.polarity=0.30 \
    --set data.tasks_partition.full=0.10

run_exp "split-70-30" \
    --set name=split-70-30 \
    --set data.shuffle_tasks=true \
    --set data.tasks_partition.aspect=0.233 \
    --set data.tasks_partition.sentiment=0.233 \
    --set data.tasks_partition.polarity=0.234 \
    --set data.tasks_partition.full=0.30

run_exp "split-50-50" \
    --set name=split-50-50 \
    --set data.shuffle_tasks=true \
    --set data.tasks_partition.aspect=0.167 \
    --set data.tasks_partition.sentiment=0.167 \
    --set data.tasks_partition.polarity=0.166 \
    --set data.tasks_partition.full=0.50

run_exp "split-30-70" \
    --set name=split-30-70 \
    --set data.shuffle_tasks=true \
    --set data.tasks_partition.aspect=0.10 \
    --set data.tasks_partition.sentiment=0.10 \
    --set data.tasks_partition.polarity=0.10 \
    --set data.tasks_partition.full=0.70

run_exp "split-0-100" \
    --set name=split-0-100 \
    --set data.shuffle_tasks=true \
    --set data.tasks_partition.aspect=0 \
    --set data.tasks_partition.sentiment=0 \
    --set data.tasks_partition.polarity=0 \
    --set data.tasks_partition.full=1.0

run_exp "baseline" \
    --set name=baseline \
    --set data.tasks_partition.aspect=0 \
    --set data.tasks_partition.sentiment=0 \
    --set data.tasks_partition.polarity=0 \
    --set data.tasks_partition.full=1.0

# ============================================================
# Phase 1 complete — determine best split by OOD performance
# Phase 2 uses the best split (manually check results or
# the script below picks it automatically)
# ============================================================

echo "" | tee -a "$LOG"
echo "Phase 1 complete. Determining best split by Laptop14 triplet micro F1..." | tee -a "$LOG"

BEST_SPLIT=$($PYTHON -c "
import json
from pathlib import Path

best_name, best_f1 = '', 0.0
for name in ['split-90-10', 'split-70-30', 'split-50-50', 'split-30-70', 'split-0-100', 'baseline']:
    rpath = Path('experiments') / name / 'results' / 'results.json'
    if not rpath.exists():
        continue
    data = json.loads(rpath.read_text())
    for entry in data.get('test', []):
        if 'Laptop14' in entry.get('data', ''):
            f1 = entry.get('aspect+sentiment+polarity', {}).get('micro', {}).get('f1', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
print(best_name)
")

echo "Best split: $BEST_SPLIT (by Laptop14 triplet micro F1)" | tee -a "$LOG"

# Read the best split's task partition values
BEST_A=$($PYTHON -c "
import yaml
with open('experiments/$BEST_SPLIT/config.yaml') as f:
    cfg = yaml.safe_load(f)
tp = cfg['data']['tasks_partition']
print(f\"{tp.get('aspect',0)} {tp.get('sentiment',0)} {tp.get('polarity',0)} {tp.get('full',0)}\")
")
read SPLIT_A SPLIT_S SPLIT_P SPLIT_F <<< "$BEST_A"

echo "Using partition: a=$SPLIT_A s=$SPLIT_S p=$SPLIT_P f=$SPLIT_F" | tee -a "$LOG"

# ============================================================
# Phase 2: Augmentation Comparison
# ============================================================

# --- Masking experiments ---

run_exp "mask-f25-append" \
    --set name=mask-f25-append \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.mask_aspects.fraction=0.25 \
    --set data.augmentation.mask_aspects.replace=false

run_exp "mask-f50-append" \
    --set name=mask-f50-append \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.mask_aspects.fraction=0.5 \
    --set data.augmentation.mask_aspects.replace=false

run_exp "mask-f75-append" \
    --set name=mask-f75-append \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.mask_aspects.fraction=0.75 \
    --set data.augmentation.mask_aspects.replace=false

run_exp "mask-f25-replace" \
    --set name=mask-f25-replace \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.mask_aspects.fraction=0.25 \
    --set data.augmentation.mask_aspects.replace=true

run_exp "mask-f50-replace" \
    --set name=mask-f50-replace \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.mask_aspects.fraction=0.5 \
    --set data.augmentation.mask_aspects.replace=true

run_exp "mask-f75-replace" \
    --set name=mask-f75-replace \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.mask_aspects.fraction=0.75 \
    --set data.augmentation.mask_aspects.replace=true

# --- nlpaug experiments ---

run_exp "nlpaug-synonym" \
    --set name=nlpaug-synonym \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.nlpaug.method=synonym \
    --set data.augmentation.nlpaug.fraction=0.3 \
    --set data.augmentation.nlpaug.protect_aspects=true

run_exp "nlpaug-swap" \
    --set name=nlpaug-swap \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.nlpaug.method=random_swap \
    --set data.augmentation.nlpaug.fraction=0.3 \
    --set data.augmentation.nlpaug.protect_aspects=true

run_exp "nlpaug-spelling" \
    --set name=nlpaug-spelling \
    --set data.tasks_partition.aspect=$SPLIT_A \
    --set data.tasks_partition.sentiment=$SPLIT_S \
    --set data.tasks_partition.polarity=$SPLIT_P \
    --set data.tasks_partition.full=$SPLIT_F \
    --set data.augmentation.nlpaug.method=spelling \
    --set data.augmentation.nlpaug.fraction=0.3 \
    --set data.augmentation.nlpaug.protect_aspects=true

# ============================================================
# Phase 3: Best masking config on pure triplet (split-0-100)
# ============================================================

echo "" | tee -a "$LOG"
echo "Phase 2 complete. Determining best masking config by Laptop14 triplet micro F1..." | tee -a "$LOG"

BEST_MASK=$($PYTHON -c "
import json
from pathlib import Path

best_name, best_f1 = '', 0.0
for name in ['mask-f25-append', 'mask-f50-append', 'mask-f75-append', 'mask-f25-replace', 'mask-f50-replace', 'mask-f75-replace']:
    rpath = Path('experiments') / name / 'results' / 'results.json'
    if not rpath.exists():
        continue
    data = json.loads(rpath.read_text())
    for entry in data.get('test', []):
        if 'Laptop14' in entry.get('data', ''):
            f1 = entry.get('aspect+sentiment+polarity', {}).get('micro', {}).get('f1', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_name = name
print(best_name)
")

echo "Best masking config: $BEST_MASK" | tee -a "$LOG"

# Extract fraction and replace from the best masking experiment's config
MASK_PARAMS=$($PYTHON -c "
import yaml
with open('experiments/$BEST_MASK/config.yaml') as f:
    cfg = yaml.safe_load(f)
m = cfg['data']['augmentation']['mask_aspects']
print(f\"{m['fraction']} {str(m['replace']).lower()}\")
")
read MASK_FRAC MASK_REPLACE <<< "$MASK_PARAMS"

echo "Using mask fraction=$MASK_FRAC replace=$MASK_REPLACE on split-0-100" | tee -a "$LOG"

run_exp "split-0-100-masked" \
    --set name=split-0-100-masked \
    --set data.tasks_partition.aspect=0 \
    --set data.tasks_partition.sentiment=0 \
    --set data.tasks_partition.polarity=0 \
    --set data.tasks_partition.full=1.0 \
    --set data.augmentation.mask_aspects.fraction=$MASK_FRAC \
    --set data.augmentation.mask_aspects.replace=$MASK_REPLACE

# ============================================================
# Final aggregation
# ============================================================

echo "" | tee -a "$LOG"
echo "All experiments complete. Running aggregation..." | tee -a "$LOG"

$PYTHON main.py --mode aggregate | tee -a "$LOG"
$PYTHON main.py --mode plot

echo "" | tee -a "$LOG"
echo "All done at $(date)" | tee -a "$LOG"
