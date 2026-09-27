#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

JSONL="${JSONL:-$REPO_ROOT/data/distance_region.jsonl}"
CLIPS="${CLIPS:-/data/projects/shot_classification/clips/}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/checkpoints_cosine}"

CACHE_DIR=/data/projects/distance_classification/shot_distance_cache

ARCH=swin_b

FPS=4
EPOCHS=10
BATCH_SIZE=32
LR=1e-4
HEAD_LR=1e-3
WEIGHT_DECAY=0.01
WEIGHT_DECAY_SCOPE=weights_only
MAX_GRAD_NORM=1.0
LABEL_SMOOTHING=0.1
FREEZE_EPOCHS=2
NUM_WORKERS=4
SEED=42

RESIZE_FILTER=area
REWARMUP="--rewarmup_on_unfreeze"

RESUME_FLAG=""
[ "${RESUME:-0}" != "0" ] && RESUME_FLAG="--resume"

LR_SCHEDULE=cosine

WARMUP_FRAC=0.05

PLATEAU_FACTOR=0.5
PLATEAU_PATIENCE=4
PLATEAU_THRESHOLD=1e-3

EARLY_STOP_PATIENCE=0

RUN_DIR="$OUTPUT_DIR/$ARCH"
LOG="$RUN_DIR/train.log"
mkdir -p "$RUN_DIR"
echo "Logging to $LOG"

export PYTHONUNBUFFERED=1

python3 "$SCRIPT_DIR/finetune.py" \
    --arch "$ARCH" \
    --jsonl "$JSONL" \
    --clips "$CLIPS" \
    --output_dir "$OUTPUT_DIR" \
    --cache_dir "$CACHE_DIR" \
    --fps $FPS \
    --epochs $EPOCHS \
    --lr_schedule $LR_SCHEDULE \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --head_lr $HEAD_LR \
    --weight_decay $WEIGHT_DECAY \
    --weight_decay_scope $WEIGHT_DECAY_SCOPE \
    --max_grad_norm $MAX_GRAD_NORM \
    --label_smoothing $LABEL_SMOOTHING \
    --freeze_epochs $FREEZE_EPOCHS \
    --num_workers $NUM_WORKERS \
    --resize_filter $RESIZE_FILTER \
    $REWARMUP \
    --warmup_frac $WARMUP_FRAC \
    --plateau_factor $PLATEAU_FACTOR \
    --plateau_patience $PLATEAU_PATIENCE \
    --plateau_threshold $PLATEAU_THRESHOLD \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --seed $SEED \
    $RESUME_FLAG \
    2>&1 | tee -a "$LOG"
