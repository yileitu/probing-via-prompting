#!/bin/bash -l

#SBATCH -n 1
#SBATCH --gpus=rtx_3090:1
#SBATCH --gres=gpumem:16384m
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=8192

module load eth_proxy
module load gcc/8.2.0
conda activate PvP
wandb login

export TASK_NAME=ner

python3 run_dp.py \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gpt2_name_or_path gpt2 \
  --data_dir ontonotes/dp/ \
  --task $TASK_NAME \
  --output_dir outputs/dp/lr/$TASK_NAME/ \
  --overwrite_output_dir \
  --cache_dir cache/ \
  --save_strategy no \
  --mlp_dropout 0.0 \
  --num_train_epochs 1.0 \
  --learning_rate 5e-3 \
  --weight_decay 0.0 \
  --randomized \
  --dev \
  --mlp_dim 4096 \
  --mlp_layers 32 \
  --fp16

