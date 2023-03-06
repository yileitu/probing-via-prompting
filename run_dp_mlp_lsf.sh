#export TASK_NAME=ner
#
#python3 run_dp.py \
#  --do_train \
#  --do_eval \
#  --per_device_train_batch_size 32 \
#  --per_device_eval_batch_size 32 \
#  --gpt2_name_or_path gpt2 \
#  --data_dir ontonotes/dp/ \
#  --task $TASK_NAME \
#  --output_dir outputs/dp/lr/$TASK_NAME/ \
#  --overwrite_output_dir \
#  --cache_dir cache/ \
#  --save_strategy no \
#  --mlp_dropout 0.0 \
#  --num_train_epochs 1.0 \
#  --learning_rate 1e-2 \
#  --weight_decay 0.0 \
#  --dev \
#  --init_mean 0.0 \
#  --init_std 0.02 \
#  --mod_randomized

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
  --weight_decay 0.0 \
  --dev \
  --num_train_epochs 1 \
  --learning_rate 1e-2 \
  --mlp_dim 512 \
  --mlp_layers 1 \
  --verbose 2 \
  --saturated \
