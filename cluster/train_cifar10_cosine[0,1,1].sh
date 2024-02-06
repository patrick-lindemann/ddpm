#!/bin/bash

#SBATCH --job-name=train-cifar10-cosine-1
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out

apptainer run --nv pml.dif \
    python3 train_denoiser.py cifar10 \
    --run--name "cifar10-cosine[0,1,1]" \
    --image-size 32 \
    --time-steps 1000 \
    --epochs 100 \
    --subset-size 6250 \
    --train-split 0.8 \
    --batch-size 32 \
    --loss-function smoothl1 \
    --learning-rate 0.0002 \
    --dropout-rate 0.1 \
    --schedule cosine \
    --schedule-start 0.0 \
    --schedule-end 1.0 \
    --schedule-tau 1.0 \
    --seed 42