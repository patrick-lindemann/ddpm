#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --partition=gpu-2d
#SBATCH --gpus-per-node=40gb:1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/job-%j.out

apptainer run --nv pml.sif python diffusion-Flowers102.py
# apptainer run --nv pml.sif python create-FGVCAircraft.py

