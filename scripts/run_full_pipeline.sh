#!/bin/bash
# Full pipeline: base session → incremental sessions → report
#
# Usage:
#   bash scripts/run_full_pipeline.sh [--seed SEED] [--device DEVICE]

set -euo pipefail

SEED=${SEED:-42}
DEVICE=${DEVICE:-cuda}
CONFIG=configs/cifar100_fscil.json
METHOD_CONFIG=configs/pc_meggroll.json
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=runs/pipeline_${TIMESTAMP}

echo "=========================================="
echo "PC-MEGGROLL Full Pipeline"
echo "=========================================="
echo "Seed: $SEED"
echo "Device: $DEVICE"
echo "Config: $CONFIG"
echo "Method: $METHOD_CONFIG"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Step 1: Base session training
echo ""
echo "Step 1/3: Base session training..."
PYTHONPATH=src python scripts/train_base.py \
    --config $CONFIG \
    --seed $SEED \
    --device $DEVICE \
    --output-dir ${OUTPUT_DIR}/base

# Step 2: Incremental sessions
echo ""
echo "Step 2/3: Incremental session training..."
PYTHONPATH=src python scripts/train_incremental.py \
    --config $CONFIG \
    --method-config $METHOD_CONFIG \
    --base-model ${OUTPUT_DIR}/base/model.pt \
    --base-prototypes ${OUTPUT_DIR}/base/prototypes.pt \
    --seed $SEED \
    --device $DEVICE \
    --output-dir ${OUTPUT_DIR}/incremental

# Step 3: Benchmark
echo ""
echo "Step 3/3: Speed/memory benchmark..."
PYTHONPATH=src python scripts/benchmark_speed.py \
    --device $DEVICE \
    --output ${OUTPUT_DIR}/benchmark.json

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "Results: ${OUTPUT_DIR}"
echo "=========================================="
