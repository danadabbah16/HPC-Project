#!/bin/bash

set -euo pipefail

for threads in 2 4 8 16; do
    sbatch \
        --cpus-per-task="$threads" \
        --export="THREADS=$threads,RUN=cpu_${threads}threads,DATA_DIR=data/processed,EPOCHS=10,BATCH_SIZE=64,LR=0.001,SEED=42" \
        slurm/train_cpu.sbatch
done
