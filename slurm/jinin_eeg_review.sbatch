#!/bin/bash
#SBATCH --job-name=jinin-eeg-review
#SBATCH --output=results/jinin/review/%x_%j.out
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

# Only EDF headers and labels are read, so 1 CPU and 2 GB are enough.
# Slurm needs the output folder to exist: mkdir -p results/jinin/review

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
VENV=${VENV:-$HOME/venvs/hypnoscale}
source "$VENV/bin/activate"
export PYTHONUNBUFFERED=1

echo "Job $SLURM_JOB_ID on $(hostname) with $SLURM_CPUS_PER_TASK CPU(s)"
echo "Commit: $(git rev-parse HEAD)"
python -m src.eeg_review
