# For GSM task
# model can be [gpt-3.5-turbo, gpt4o_mini]
# Inference on GSM8K
python gsm_pipeline.py \
    --port 8000 \
    --whitebox lora \
    --dataset gsm \
    --model gpt4o_mini \
    --closeloop \
    --compare_answer

# Inference on GSM-Hard
python gsm_pipeline.py \
    --port 8000 \
    --whitebox lora \
    --dataset gsmhardv2 \
    --model gpt4o_mini \
    --closeloop \
    --compare_answer

# For LLaMP task
python llamp_pipeline.py \
    --whitebox meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --few_gpt_shot 1 \
    --gpt-engine gpt4o_mini

# For Alfworld task
python alfworld_pipeline.py \
	--whitebox meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --use_gpt