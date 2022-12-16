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
  --num_train_epochs 8.0 \
  --learning_rate 1e-5 \
  --prefix_len 20 \
  --flat \
  --weight_decay 0.0 \
  --randomized \
  --dev \
  --fp16
