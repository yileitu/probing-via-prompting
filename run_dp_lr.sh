#!/bin/bash -l

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384

module load eth_proxy
module load gcc/8.2.0
conda activate PvP
wandb login

export TASK_NAME=ner
export CUDA_LAUNCH_BLOCKING=1

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
  --use_mlp False \
  --num_train_epochs 256.0 \
  --learning_rate 5e-5 \
  --weight_decay 0.0 \
  --fp16 \
  --evaluation_strategy epoch \
  --dev \
  --randomized \
