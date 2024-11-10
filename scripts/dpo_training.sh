set -x

read -r -d '' training_commands <<EOF
train_dpo \
   --save_path ./checkpoint/llama3-8b-instruct-dpo-lora \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 64 \
   --micro_train_batch_size 1 \
   --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 3 \
   --learning_rate 5e-6 \
   --beta 0.1 \
   --dataset json@./data/gsm8k/gpt3.5/collect \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --gradient_checkpointing \
   --label_smoothing 0.1 \
   --lora_rank 8 \
   --lora_alpha 16 \
   --lora_dropout 0.05
EOF

# --use_wandb

if [[ ${1} != "slurm" ]]; then
    deepspeed  --master_port=29400 --include localhost:4,5,6,7 --module $training_commands
fi
