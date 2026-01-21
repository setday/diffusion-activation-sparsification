#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=dit-train-%a.out
#SBATCH --job-name=dit-train
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

accelerate launch --mixed_precision fp16 train.py --model DiT-S/2 --epochs 60 --feature-path ./features --act-layer ${MODES_ACT[$SLURM_ARRAY_TASK_ID]} --norm-layer ${MODES_NORM[$SLURM_ARRAY_TASK_ID]}