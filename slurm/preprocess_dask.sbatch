#!/bin/bash
#SBATCH --job-name=sleep-dask
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# Parallel preprocess with Dask; compare wall time to sequential (~165 s in docs/DATA.md).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs data/processed_dask

RAW=${RAW:-data/raw/sleep-cassette}
OUT=${OUT:-data/processed_dask}
N_WORKERS=${N_WORKERS:-4}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SLURM_SUBMIT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ ! -d "$RAW" ]]; then
    echo "ERROR: $RAW missing. Download on the login node first:"
    echo "  bash scripts/download_data.sh data/raw"
    exit 1
fi

if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    module purge
    module load pytorch/2.8.0
    PY="torch-exec python"
fi

echo "Job $SLURM_JOB_ID on $(hostname) | Dask workers=$N_WORKERS"
$PY -m src.data.preprocess_dask \
    --raw "$RAW" \
    --out "$OUT" \
    --n-workers "$N_WORKERS" \
    --threads-per-worker 1
