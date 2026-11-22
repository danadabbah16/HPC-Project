#!/bin/bash
#SBATCH --job-name=gpu-sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

module purge
module load pytorch/2.8.0

mkdir -p results/logs
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

srun apptainer exec --nv "$PYTORCH_GPU_SIF" bash -lc '
  cd "$SLURM_SUBMIT_DIR"
  RUN=${RUN:-gpu_fp32}
  python -m src.train \
    --data data/processed \
    --epochs 10 \
    --batch-size 64 \
    --lr 0.001 \
    --seed 42 \
    --threads 1 \
    --device cuda \
    --run-name "$RUN" \
    --results results/results.csv
'
