#!/bin/bash
#SBATCH --job-name=gpu-sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module purge
module load pytorch/2.8.0

mkdir -p logs

srun apptainer exec --nv "$PYTORCH_GPU_SIF" bash -lc '
  cd "$SLURM_SUBMIT_DIR"
  python -m src.train \
    --data data/sample \
    --epochs 2 \
    --batch-size 32 \
    --threads 1 \
    --device cuda \
    --run-name gabriel_sample_gpu \
    --results results/results.csv
'
