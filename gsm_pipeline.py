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

from pal import interface
from pal.prompt import math_prompts
from openai import OpenAI
import time
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
PR_FILE = 'Prompt_Response.txt'

parser = argparse.ArgumentParser()
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="vllm whiteBox port",
)
parser.add_argument("--whitebox", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Use for vllm client. If use lora, set to lora id, otherwise set to base model name")
parser.add_argument('--dataset', default='gsmhardv2', type=str)
parser.add_argument('--model', default='gpt-3.5-turbo', choices= ['gpt-3.5-turbo', 'gpt4o_mini', 'gpt4o'], type=str)
parser.add_argument('--temperature' , default=0.0, type=float)
parser.add_argument('--top_p', default=1.0, type=float)
parser.add_argument('--max_tokens', default=512, type=int)
parser.add_argument('--closeloop', action='store_true')
parser.add_argument('--compare_answer', action='store_true')

args = parser.parse_args()

def save_to_file(directory, filename, content):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory, filename)
    with open(filename, 'a') as file:
        file.write(content + '\n')

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

whiteBox = "LLaMA3" if args.whitebox == 'meta-llama/Meta-Llama-3-8B-Instruct' else args.whitebox
gpt = 'chatgpt' if args.model == 'gpt-3.5-turbo' else 'gpt4o_mini' if args.model == 'gpt4o_mini' else 'gpt4o'
DATA_PATH = f'datasets/{args.dataset}.jsonl'
OUTPUT_PATH = f'eval_results/{args.dataset}_gpt{gpt}_{current_time}_{whiteBox}.chat.jsonl'
save_to_folder = os.path.dirname(OUTPUT_PATH)
PR_FILE = os.path.join(save_to_folder, PR_FILE)
save_to_file(save_to_folder, PR_FILE, 'Start: ' + str(current_time) + '\n')

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

examples = list(map(json.loads, open(DATA_PATH)))

itf = interface.ProgramChatInterface(
    stop=None,
    get_answer_expr='solution()',
    model=args.model,
    verbose=False,
    system_message=math_prompts.MATH_CHAT_ANNOTATION_DECOMPOSE_BETA_SYSTEM_MESSAGE,
)

num_skip_exps = 0
scores = []
MAX_TRY = 6 if args.closeloop else 1
cnt = 0
with open(OUTPUT_PATH, 'w') as f:
    pbar = tqdm.tqdm(examples[num_skip_exps:], initial=num_skip_exps, total=len(examples))
    for x in pbar:
        cnt += 1
        if cnt % 50 == 0:
            f.write(f'Accuracy - {sum(scores) / len(scores)}')
        question = x['input']
        result = copy.copy(x)
        llama_prompt = math_prompts.LLAMA_DECOMPOSE_PROMPT.format(question=question)
        decompose = itf.decompose(math_prompts.LLAMA_SYSTEM_MESSAGE, whiteBox_client, whiteBox_model, llama_prompt, temperature=0, top_p=1, max_tokens=args.max_tokens)
        save_to_file(save_to_folder, PR_FILE, f'Task {cnt} Decompose Prompt: \n{llama_prompt}\nResponse: \n{decompose}\n' + '*'*50 + '\n')
        for i in range(MAX_TRY):
            try:
                if i == 0:
                    math_prompt = math_prompts.MATH_CHAT_ANNOTATION_DECOMPOSE_BETA_PROMPT.format(question=question, decompose=decompose)
                else:
                    math_prompt = math_prompts.MATH_CHAT_MODIFY_ERROR_PROMPT.format(question=question, error=error_msg)
                ans = itf.run(
                    math_prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens
                )
                ans = float(ans)
                score = 1 if abs(ans - x['target']) < 1e-3 else 0
                if score == 0:
                    error_msg = 'You executed the solution function successfully but the answer is not correct. Please check your solution function.'
                save_to_file(save_to_folder, PR_FILE, f'Task {cnt} Try {i} Prompt: \n{math_prompt}\nResponse: \n{itf.history[0]}\nAnswer: \n{ans}\n' + '='*50 + '\n')
                if not args.compare_answer:
                    break
            except Exception as e:
                # print('Execution Error')
                error_msg = e
                print(e)
                ans = ''
                score = 0
            if score == 1:
                break
        scores.append(score)
        
        result['answer'] = ans
        result['score'] = score
        result['generation'] = itf.history
        f.write(json.dumps(result) + '\n')
        
        itf.clear_history()
        f.flush()
    f.write(f'Accuracy - {sum(scores) / len(scores)}')
print(f'Accuracy - {sum(scores) / len(scores)}')
