#!/bin/bash -l

#SBATCH -n 1
#SBATCH --gpus=a100-pcie-40gb:1
#SBATCH --gres=gpumem:16384m
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=16384

module load eth_proxy
module load gcc/8.2.0
conda activate PvP

export TASK_NAME=ner
export CUDA_LAUNCH_BLOCKING=1

python3 run_pp.py \
  --seed 21946520 \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gpt2_name_or_path gpt2 \
  --data_dir ontonotes/pp/ \
  --task $TASK_NAME \
  --output_dir outputs/pp/$TASK_NAME/ \
  --overwrite_output_dir \
  --use_fast_tokenizer False \
  --cache_dir cache/ \
  --num_train_epochs 256 \
  --learning_rate 1e-4 \
  --prefix_len 20 \
  --weight_decay 0.0 \
  --fp16 \
  --evaluation_strategy epoch \
  --dev \
  --randomized \
  --save_strategy steps \
  --save_steps 1029920 \

