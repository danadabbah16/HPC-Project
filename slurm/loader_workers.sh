#!/bin/bash
# DataLoader worker sensitivity: 0 → 1 → 2 → 4 → 8 (Salome).
# Keeps 1 CPU training thread fixed so we isolate loader parallelism.
set -euo pipefail

DATA_DIR=${DATA_DIR:-data/processed}
EPOCHS=${EPOCHS:-10}
BATCH=${BATCH:-64}
LR=${LR:-0.001}
SEED=${SEED:-42}

for w in 0 1 2 4 8; do
    # Ask Slurm for enough CPUs for the training thread + worker processes.
    cpus=$((1 + w))
    sbatch \
        --cpus-per-task="$cpus" \
        --mem=12G \
        --job-name="loader_w${w}" \
        --export="ALL,DATA_DIR=${DATA_DIR},EPOCHS=${EPOCHS},BATCH_SIZE=${BATCH},LR=${LR},SEED=${SEED},THREADS=1,LOADER_WORKERS=${w},RUN=loader_w${w}" \
        slurm/train_loader.sbatch
done
