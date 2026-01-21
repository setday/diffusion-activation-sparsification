#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=sample-%a.out
#SBATCH --job-name=sample
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

torchrun --nnodes=1 --nproc_per_node=1 sample_ddp.py --model DiT-S/2 --act-layer ${MODES_ACT[$SLURM_ARRAY_TASK_ID]} --norm-layer ${MODES_NORM[$SLURM_ARRAY_TASK_ID]} --num-fid-samples 50000 --ckpt ./results/DiT-S-2-${MODES_ACT[$SLURM_ARRAY_TASK_ID]}-${MODES_NORM[$SLURM_ARRAY_TASK_ID]}/checkpoints/0300000.pt