#!/bin/bash
#SBATCH --job-name=sleep-train
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Settings you can change when submitting (defaults = quick test)
DATA_DIR=${DATA_DIR:-data/sample}
EPOCHS=${EPOCHS:-3}
RUN=${RUN:-test_slurm}

echo "Job $SLURM_JOB_ID on $(hostname) with $SLURM_CPUS_PER_TASK CPU(s)"
srun python -m src.train --data "$DATA_DIR" --epochs "$EPOCHS" \
     --threads "$SLURM_CPUS_PER_TASK" --run-name "$RUN"
