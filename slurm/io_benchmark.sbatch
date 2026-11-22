#!/bin/bash
#SBATCH --job-name=sleep-io
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Build HDF5 (if needed) and compare EDF vs .npz vs HDF5 load times.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results/logs

RAW=${RAW:-data/raw/sleep-cassette}
PROCESSED=${PROCESSED:-data/processed}
N_EDF=${N_EDF:-8}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
else
    module purge
    module load pytorch/2.8.0
    PY="torch-exec python"
fi

echo "Job $SLURM_JOB_ID on $(hostname)"
$PY -m src.data.to_hdf5 --data "$PROCESSED" --out "$PROCESSED/sleep_edf.h5"
$PY -m src.benchmark_io --raw "$RAW" --processed "$PROCESSED" --n-edf "$N_EDF" \
    --results results/io_benchmark.csv
