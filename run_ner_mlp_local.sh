python3 run_ner.py \
  --n_gpu 1 \
  --task NanmedEntityRecognition \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --gpt2_name_or_path gpt2 \
  --output_dir outputs/ner/dp/ \
  --overwrite_output_dir \
  --cache_dir cache/ \
  --num_train_epochs 2 \
  --learning_rate 1e-5 \
  --weight_decay 0.0 \
  --mlp_dropout 0.0 \
  --mlp_dim 512 \
  --mlp_layers 8 \
  --use_mlp True \
  --dev \
  --save_strategy epoch \
  --evaluation_strategy epoch \


