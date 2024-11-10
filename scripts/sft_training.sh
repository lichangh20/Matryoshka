set -x

read -r -d '' training_commands <<EOF
train_sft \
   --max_len 4096 \
   --dataset json@./data/gsm8k/pal_sft \
   --apply_chat_template \
   --input_key prompt \
   --output_key response \
   --train_batch_size 64 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain meta-llama/Meta-Llama-3-8B-Instruct \
   --save_path ./pal/llama3-8b-instruct-sft-lora-trainset \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --learning_rate 2e-5 \
   --load_checkpoint \
   --gradient_checkpointing \
   --ckpt_path ./pal/llama3-8b-instruct-sft-lora-trainset/checkpoints_sft \
   --lora_rank 8 \
   --lora_alpha 16 \
   --lora_dropout 0.05 \
   --max_ckpt_num 10
EOF

# --use_wandb

if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port=29600 --include localhost:4,5,6,7 --module $training_commands
fi
