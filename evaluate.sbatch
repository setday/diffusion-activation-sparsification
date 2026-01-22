#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=evaluate-%a.out
#SBATCH --job-name=evaluate
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=0-3

MODES_ACT=(
    "GeLU"
    "GeLU"
    "ReLU"
    "ReLU"
)
MODES_NORM=(
    "LayerNorm"
    "LayerNorm-MeanQuantile-50"
    "LayerNorm"
    "LayerNorm-MeanQuantile-50"
)

python evaluator.py VIRTUAL_imagenet256_labeled.npz samples/DiT-S-2-${MODES_ACT[$SLURM_ARRAY_TASK_ID]}-${MODES_NORM[$SLURM_ARRAY_TASK_ID]}-0300000-size-256-vae-ema-cfg-1.5-seed-0.npz