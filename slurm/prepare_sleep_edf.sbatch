#!/bin/bash
#SBATCH --job-name=prepare-sleep-edf
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p data/processed results/logs

if [[ ! -f data/raw/SHA256SUMS.txt ]]; then
    echo "ERROR: download data/raw/sleep-cassette first on the login node:"
    echo "  bash scripts/download_data.sh data/raw"
    exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [[ -x .venv/bin/python ]]; then
    .venv/bin/python -m src.data.preprocess \
        --raw data/raw/sleep-cassette \
        --out data/processed
    .venv/bin/python -m src.data.make_splits \
        --data data/processed \
        --seed 42 \
        --sample data/sample
else
    module purge
    module load pytorch/2.8.0
    torch-exec python -m src.data.preprocess \
        --raw data/raw/sleep-cassette \
        --out data/processed
    torch-exec python -m src.data.make_splits \
        --data data/processed \
        --seed 42 \
        --sample data/sample
fi

test -f data/processed/splits.json
test -f data/processed/meta.json
