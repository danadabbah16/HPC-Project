#!/bin/bash
#SBATCH --job-name=sleep-train
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

DATA_DIR=${DATA_DIR:-data/processed}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.001}
SEED=${SEED:-42}
RUN=${RUN:-baseline_1cpu}
THREADS=${THREADS:-$SLURM_CPUS_PER_TASK}
LOADER_WORKERS=${LOADER_WORKERS:-0}
DATA_FORMAT=${DATA_FORMAT:-npz}

mkdir -p results/logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

echo "Job $SLURM_JOB_ID on $(hostname) | $THREADS CPU thread(s) | loader_workers=$LOADER_WORKERS"

if [[ -x .venv/bin/python ]]; then
    .venv/bin/python -m src.train \
        --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --lr "$LR" --seed "$SEED" --threads "$THREADS" \
        --loader-workers "$LOADER_WORKERS" --data-format "$DATA_FORMAT" \
        --device cpu --run-name "$RUN" --results results/results.csv
else
    module purge
    module load pytorch/2.8.0
    torch-exec python -m src.train \
        --data "$DATA_DIR" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --lr "$LR" --seed "$SEED" --threads "$THREADS" \
        --loader-workers "$LOADER_WORKERS" --data-format "$DATA_FORMAT" \
        --device cpu --run-name "$RUN" --results results/results.csv
fi
