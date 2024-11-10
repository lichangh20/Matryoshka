# For GSM task
python gsm_generate_data.py \
    --whitebox lora \
    --port 8000

# For LLaMP task
python llamp_generate_data.py \
    --few_gpt_shot 1

# For Alfworld task
python alfworld_generate_data.py \
    --disable_closeloop