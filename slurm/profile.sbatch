#!/bin/bash
#SBATCH --job-name=sleep-profile
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# Profile one training loop on 1 CPU (matches the shared baseline setup).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs results/profiler

DATA_DIR=${DATA_DIR:-data/processed}
WORKERS=${WORKERS:-0}
THREADS=${THREADS:-$SLURM_CPUS_PER_TASK}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"

echo "Job $SLURM_JOB_ID on $(hostname) | threads=$THREADS loader_workers=$WORKERS"

if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    module purge
    module load pytorch/2.8.0
    PY="torch-exec python"
fi

$PY -m src.profile_train \
    --data "$DATA_DIR" \
    --device cpu \
    --threads "$THREADS" \
    --loader-workers "$WORKERS" \
    --batch-size 64 \
    --out results/profiler
