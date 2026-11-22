#!/bin/bash
#SBATCH --job-name=sleep-loader
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

# Single training run with a chosen DataLoader worker count (Salome).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

DATA_DIR=${DATA_DIR:-data/processed}
EPOCHS=${EPOCHS:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
LR=${LR:-0.001}
SEED=${SEED:-42}
THREADS=${THREADS:-1}
LOADER_WORKERS=${LOADER_WORKERS:-0}
RUN=${RUN:-loader_w${LOADER_WORKERS}}
DATA_FORMAT=${DATA_FORMAT:-npz}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

echo "Job $SLURM_JOB_ID on $(hostname) | train_threads=$THREADS loader_workers=$LOADER_WORKERS format=$DATA_FORMAT"

if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    module purge
    module load pytorch/2.8.0
    PY="torch-exec python"
fi

$PY -m src.train \
    --data "$DATA_DIR" \
    --data-format "$DATA_FORMAT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --seed "$SEED" \
    --threads "$THREADS" \
    --loader-workers "$LOADER_WORKERS" \
    --device cpu \
    --run-name "$RUN" \
    --results results/results.csv
