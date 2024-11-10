PORT=${1:-8000}
CUDA_VISIBLE_DEVICES=${2:-0}
MODEL=${3:-"meta-llama/Meta-Llama-3-8B-Instruct"}


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

python -m vllm.entrypoints.openai.api_server --model ${MODEL} --trust-remote-code --dtype=half --port=${PORT}