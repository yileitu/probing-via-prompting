#!/bin/bash

#SBATCH -n 1
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:16384m
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384

module load eth_proxy
srun run_pp_ner.sh
