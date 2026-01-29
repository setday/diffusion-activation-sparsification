#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=finetune-%a.out
#SBATCH --job-name=finetune
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=2-3

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

accelerate launch --mixed_precision fp16 finetune.py --model DiT-S/2 --feature-path ./features --act-layer ${MODES_ACT[$SLURM_ARRAY_TASK_ID]} --norm-layer ${MODES_NORM[$SLURM_ARRAY_TASK_ID]}  --ckpt ./results/DiT-S-2-GeLU-${MODES_NORM[$SLURM_ARRAY_TASK_ID]}/checkpoints/0300000.pt