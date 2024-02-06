#!/bin/bash

#SBATCH --job-name=train-cifar10-linear-1
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out

script_dir=$(dirname "$0")

apptainer run --nv $script_dir/../pml.sif \
    python3 train_denoiser.py cifar10 \
    --run-name "cifar10-linear[0.0001,0.02]" \
    --image-size 32 \
    --time-steps 1000 \
    --epochs 100 \
    --subset-size 6250 \
    --train-split 0.8 \
    --batch-size 32 \
    --loss-function smoothl1 \
    --learning-rate 0.0002 \
    --dropout-rate 0.1 \
    --schedule linear \
    --schedule-start 0.0001 \
    --schedule-end 0.02 \
    --seed 42