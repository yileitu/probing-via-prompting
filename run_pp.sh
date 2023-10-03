#!/bin/bash -l

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=16384

module load eth_proxy
module load gcc/9.3.0
module load cuda/11.7.0
conda activate PvP
wandb login

export TASK_NAME=ner

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
  --save_strategy no \
  --num_train_epochs 192 \
  --learning_rate 1e-5 \
  --prefix_len 20 \
  --weight_decay 0.0 \
  --fp16 \
  --evaluation_strategy epoch \
  --randomized \
  --dev