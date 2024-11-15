#!/bin/bash
#SBATCH --partition=M1
#SBATCH --qos=q_d8_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --job-name=pt1
#SBATCH --output=output_%x_%j.out
#SBATCH --time=06:00:00

# Two partitions available
# partition=M1 & qos=q_d8_norm
# partition=M2 & qos=q_a_norm

# This is the installed cuda library, check via nvidia-smi / nvcc --version
module load cuda/12.2.2

# Documentation suggests loading the anaconda module here
# This didn't seem to work for me, it can access the env later on anyway
# module load anaconda/24.5.1

eval "$(conda shell.bash hook)"
conda activate RL_env4
# Add gpu-specific imports via pip here if necessary
python3 main.py