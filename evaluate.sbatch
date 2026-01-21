#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --output=sample-eval-dit-%a.out
#SBATCH --job-name=sample-eval-dit
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --array=1,3

MODES_ACT=(
    "GeLU"
    "GeLU"
    "ReLU"
    "ReLU"
)
MODES_NORM=(
    "LN"
    "LN_MQ50"
    "LN"
    "LN_MQ50"
)

source venv128/bin/activate

python evaluator.py VIRTUAL_imagenet256_labeled.npz samples/DiT-S-2-${MODES_ACT[$SLURM_ARRAY_TASK_ID]}-${MODES_NORM[$SLURM_ARRAY_TASK_ID]}-0300000-size-256-vae-ema-cfg-1.5-seed-0.npz