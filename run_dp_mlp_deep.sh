#!/bin/bash -l

#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16384
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10240m

module load eth_proxy
module load gcc/9.3.0
module load cuda/11.7.0
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
  --output_dir outputs/dp/mlp/$TASK_NAME/ \
  --overwrite_output_dir \
  --cache_dir cache/ \
  --mlp_dropout 0.0 \
  --num_train_epochs 256.0 \
  --learning_rate 1e-4 \
  --weight_decay 0.0 \
  --mlp_dim 512 \
  --mlp_layers 4 \
  --fp16 \
  --evaluation_strategy epoch \
  --use_mlp False \
  --randomized \
  --dev \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --metric_for_best_model eval_loss \
#  --onehot \

