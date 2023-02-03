python -m torch.distributed.launch --nproc_per_node=2 ft_t5.py \
  --fp16 False \
  --bf16 True \
  --model_name_or_path "google/flan-t5-small" \
  --output_dir "/nlp/scr/lxuechen/tests/ft_t5" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --eval_steps 5 \
  --save_strategy "steps" \
  --save_steps 100 \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 1 \
  --fsdp "full_shard auto_wrap offload" \
  --fsdp_transformer_layer_cls_to_wrap "T5Block"

#  --model_name_or_path "google/flan-t5-xxl" \
