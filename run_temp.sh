#!/bin/bash -l

#SBATCH -n 1
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:16384m
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16384

module load eth_proxy
module load gcc/8.2.0
conda activate PvP
wandb login

python3 onehot_probe.py \
