# Copyright 2023 PAL Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import json
import argparse
import tqdm
import os
import random

from pal import interface
from pal.prompt import math_prompts
from openai import OpenAI
import time
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="vllm whiteBox port",
)
parser.add_argument("--whitebox", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Use for vllm client. If use lora, set to lora id, otherwise set to base model name")
parser.add_argument('--dataset', default='gsm_train', type=str)
parser.add_argument('--model', default='gpt-3.5-turbo', choices= ['gpt-3.5-turbo', 'gpt4o_mini', 'gpt4o'], type=str)
parser.add_argument('--temperature' , default=0.0, type=float)
parser.add_argument('--top_p', default=1.0, type=float)
parser.add_argument('--max_tokens', default=512, type=int)

args = parser.parse_args()

whiteBox_api_key = "whitebox"
whiteBox_api_base = f"http://localhost:{args.port}/v1"

whiteBox_client = OpenAI(
    api_key=whiteBox_api_key,
    base_url=whiteBox_api_base,
)
model_lists = [data.id for data in whiteBox_client.models.list().data]
assert args.whitebox in model_lists, f"Model {args.whitebox} not in the model list"
whiteBox_model = args.whitebox
print(f'Choose model: {whiteBox_model} from list: {model_lists}')

def parse_solution(solution):
    solution_steps = solution.split('\n')
    parsed_decomposition = "Let's break down this problem:"
    for step in solution_steps:
        step = step.strip()
        if step.startswith('#') and '?' in step and 'initialization' not in step.lower():
            parsed_decomposition += "\n" + step[1:].strip()

    return parsed_decomposition

DATA_PATH = f'data/gsm8k/pal/{args.dataset}.jsonl'
DPO_PATH = f'data/gsm/gpt3.5/collect_backup/train.json'

os.makedirs(os.path.dirname(DPO_PATH), exist_ok=True)

examples = list(map(json.loads, open(DATA_PATH)))

itf = interface.ProgramChatInterface(
    stop=None,
    get_answer_expr='solution()',
    model=args.model,
    verbose=False,
    system_message=math_prompts.MATH_CHAT_ANNOTATION_DECOMPOSE_BETA_SYSTEM_MESSAGE,
)

num_skip_exps = 0

NUM_RETURNS = 10
cnt = 0
DPO_PAIR = []
pbar = tqdm.tqdm(examples[num_skip_exps:], initial=num_skip_exps, total=len(examples))
for x in pbar:
    question = x['input']
    llama_prompt = math_prompts.LLAMA_DECOMPOSE_PROMPT.format(question=question)
    decomposes = itf.decompose(math_prompts.LLAMA_SYSTEM_MESSAGE, whiteBox_client, whiteBox_model, llama_prompt, temperature=1, max_tokens=args.max_tokens, num_returns=NUM_RETURNS)
    
    contrastive_pair = {}
    contrastive_pair['question'] = question
    contrastive_pair['positive'] = []
    contrastive_pair['negative'] = []
    cnt += 1
    for decompose in decomposes:
        
        try:
            ans = itf.run(
                math_prompts.MATH_CHAT_ANNOTATION_DECOMPOSE_BETA_PROMPT.format(question=question, decompose=decompose),
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens
            )
            ans = float(ans)
            score = 1 if abs(ans - x['target']) < 1e-3 else 0
        except Exception as e:
            print(e)
            ans = ''
            score = 0
        reserve_dict = {'response': parse_solution(itf.history[0]), 'solution': itf.history[0]}
        steps = parse_solution(itf.history[0]).split('\n')
        if len(steps) > 1:
            if score == 1:
                contrastive_pair['positive'].append(reserve_dict)
            else:
                contrastive_pair['negative'].append(reserve_dict)
        
        itf.clear_history()
    DPO_PAIR.append(contrastive_pair)
    with open(DPO_PATH, 'w') as f:
        json.dump(DPO_PAIR, f, indent=2)

# convert to dpo training format

# convert to chat mode
def wrap_dataset(question: str, response: str) -> dict:
    chats = []
    chats.append({"role": "system", "content": math_prompts.LLAMA_SYSTEM_MESSAGE})
    chats.append({"role": "user", "content": math_prompts.LLAMA_DECOMPOSE_PROMPT.format(question=question)})
    chats.append({"role": "assistant", "content": response})
    return chats

with open(DPO_PATH, "r") as f:
    dpo_pairs = json.load(f)

jsons_file = os.path.join(f'data/gsm/gpt3.5', 'collect', 'train.jsonl')
os.makedirs(os.path.join('data/gsm/gpt3.5', 'collect'), exist_ok=True)
with open(jsons_file, "w") as f:
    for data in dpo_pairs:
        if len(data["positive"]) == 0 or len(data["negative"]) == 0:
            continue
        cnt = min(len(data["positive"]), len(data["negative"]))
        positive_decompose = random.sample(data["positive"], cnt)
        positive_decompose = [wrap_dataset(data["question"], d["response"]) for d in positive_decompose]
        negative_decompose = random.sample(data["negative"], cnt)
        negative_decompose = [wrap_dataset(data["question"], d["response"]) for d in negative_decompose]
        for positive, negative in zip(positive_decompose, negative_decompose):
            f.write(json.dumps({"rejected": negative, "chosen": positive}) + "\n")