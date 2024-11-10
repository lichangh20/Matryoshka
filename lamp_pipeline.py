import argparse
import random
import os
from tqdm import tqdm
from utils.json_utils import load_jsonlines
from utils.general_utils import seed_everything
from pal.prompt.lamp_prompts import create_black_prompts, create_white_prompts
import concurrent.futures
import re
import json
import torch
import numpy as np
from models.credentials import chatgpt_api_key_list, chatgpt_0125_api_key_list, gpt_4o_api_key_list, gpt_4o_mini_api_key_list

import time
from openai import OpenAI, AzureOpenAI

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def parse_opt():
    parser = argparse.ArgumentParser(description='whiteBox3 decompose question, chatgpt inference on math problems')
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vllm whiteBox port",
    )
    parser.add_argument(
        "--few_shot",
        type=int,
        default=0,
        help="few shot num for LLaMA",
    )
    parser.add_argument(
        "--few_gpt_shot",
        type=int,
        default=0,
        help="few shot num for GPT",
    )
    parser.add_argument("--whitebox", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Use for vllm client. If use lora, set to lora id, otherwise set to base model name")
    parser.add_argument("--intermediate-prompt", type=str, default='decompose', choices=['decompose', 'cot'], help="choose the intermediate prompt kind")
    parser.add_argument("--add_answer", action="store_true", default=False, help="whether add answer in the decomposed intermediate prompt")
    parser.add_argument("--gpt-engine", type=str, default='gpt3.5_0125', choices=['gpt4o', 'gpt4o_mini', 'gpt3.5_0125', 'gpt3.5_1106'], help="choose the gpt engine version to use")
    args = parser.parse_args()
    return args

args = parse_opt()
seed_everything(args.seed)

#===================== Params to change =====================#
whiteBox_api_key = "whitebox"
whiteBox_api_base = f"http://localhost:{args.port}/v1"

task = "LaMP-1"
index = task.split('-')[1].lower()
test_set = load_jsonlines(f"data/lamp/formalized/lamp{index}/formalized_dev_data.jsonl")
os.makedirs("logs/CacheTmp2", exist_ok=True)
whitebox = 'Llama3' if args.whitebox == 'meta-llama/Meta-Llama-3-8B-Instruct' else 'Llama3.1' if args.whitebox == 'meta-llama/Meta-Llama-3.1-8B-Instruct' else args.whitebox
json_file = f"data/lamp/inference/lamp{index}/inference_dev_data.jsonl"
#============================================================#

if index == '1':
    num_profs = 10000
if index == '2n':
    num_profs = 120
if index == '3':
    num_profs = 30
elif index == '2m':
    num_profs = 150
elif index == '4':
    num_profs = 50


class generate_pipeline():
    def __init__(self):
        self.api_idx = 0
        if args.gpt_engine == 'gpt4o':
            self.api_key_list = gpt_4o_api_key_list
            self.model_type = 'gpt-4o'
        elif args.gpt_engine == 'gpt4o_mini':
            self.api_key_list = gpt_4o_mini_api_key_list
            self.model_type = 'gpt-4o-mini'
        elif args.gpt_engine == 'gpt3.5_0125':
            self.api_key_list = chatgpt_0125_api_key_list
            self.model_type = 'gpt-3.5-turbo'
        elif args.gpt_engine == 'gpt3.5_1106':
            self.api_key_list = chatgpt_api_key_list
            self.model_type = 'gpt-3.5-turbo'
        self.gpt_client = AzureOpenAI(
            azure_endpoint = self.api_key_list[self.api_idx]['azure_endpoint'],
            api_key = self.api_key_list[self.api_idx]['api_key'],
            api_version = self.api_key_list[self.api_idx]['api_version'],
        )
        self.engine = self.api_key_list[self.api_idx]['engine']
        self.max_length = 512
        self.data = test_set
        self.whiteBox_responses = [] 
        # self.log_file = log_file
        self.num_returns = 1
        self.cnt = self.corr = 0
        self.question_responses = []
        self.system_choice = 1 if args.add_answer else 0
        self.question_summary = []

        if args.intermediate_prompt == 'decompose':
            self.whiteBox_client = OpenAI(
                api_key=whiteBox_api_key,
                base_url=whiteBox_api_base,
            )
            model_lists = [data.id for data in self.whiteBox_client.models.list().data]
            assert args.whitebox in model_lists, f"Model {args.whitebox} not in the model list"
            self.whiteBox_model = args.whitebox
            print(f'Choose model: {self.whiteBox_model} from list: {model_lists}')

    def switch_api_key(self):
        self.api_idx = (self.api_idx + 1) % len(self.api_key_list)
        self.gpt_client = AzureOpenAI(
            api_key = self.api_key_list[self.api_idx]['api_key'],
            api_version = self.api_key_list[self.api_idx]['api_version'],
            azure_endpoint = self.api_key_list[self.api_idx]['azure_endpoint'],
        )
        self.engine = self.api_key_list[self.api_idx]['engine']

    def generate(self):
        print('='*50)
        print('Start inferencing {} data...'.format(len(self.data)))
        whitebox_p_bar = tqdm(range(len(self.data)), desc="WhiteBox processing")
        gpt_p_bar = tqdm(range(len(self.data)), desc="GPT processing")

        def whiteBox_query(datapoint):
            num_trials = 0
            max_trials = 8
            if args.intermediate_prompt == 'decompose':
                messages_generator = create_white_prompts(datapoint['question'], num_profs, is_ranked=False, use_all=False)
                messages = messages_generator(datapoint['question'], datapoint['profile'], task)
                # print('messages:', messages, '\n\n')
            else:
                raise NotImplementedError
            # print('messages:', messages, '\n\n')
            messages = [{'role':'user', 'content': messages}]
            while num_trials < max_trials:
                try:
                    raw_response = self.whiteBox_client.chat.completions.create(
                        model=self.whiteBox_model,
                        messages=messages,
                        max_tokens=self.max_length,
                        n=self.num_returns,
                        temperature=0.0
                    )
                    response = raw_response.choices[0].message.content
                    # self.whiteBox_responses.append(response)
                    question_responses = datapoint.copy()
                    question_responses['whiteBox_responses'] = response
                    self.question_responses.append(question_responses)
                    whitebox_p_bar.update(1)
                    break
                except Exception as e:
                    num_trials += 1
                    print(e)
                    if num_trials == max_trials:
                        print(f"Failed to get responses for {datapoint['id']} after {max_trials} trials")
                        break
                    time.sleep(5)


        def gpt_query(datapoint):
            question_point = {}
            question_point['question'] = datapoint['question']
            question_point['answer'] = datapoint['answer']
            def _gpt_call(white_response):
                if args.intermediate_prompt == 'decompose':
                    messages_generator = create_black_prompts(1, is_ranked=False, use_all=False, is_rag=False)
                    messages = messages_generator(datapoint['question'], datapoint['profile'], task, white_response)
                    messages = [{'role':'user', 'content': messages}]
                    num_trials = 0
                    max_trials = 8
                while num_trials < max_trials:
                    try:
                        raw_response = self.gpt_client.chat.completions.create(
                            model=self.engine,
                            messages=messages,
                            max_tokens=self.max_length,
                            n=self.num_returns,
                            temperature=0.0
                        )
                        response = raw_response.choices[0].message.content
                        # only extract the answer with format "[i]", i is 1 or 2, return [i]
                        if task == "LaMP-1":
                            match = re.search(r'\[(\d+)\]', response)
                            if match:
                                answer = match.group(0)  # This includes the brackets [i]
                            else:
                                answer = None  # or some default value if no match is found
                            if answer == datapoint['answer']:
                                self.corr += 1
                        if task == "LaMP-2N" or task == "LaMP-2M" or task == "LaMP-3":
                            if response == datapoint['answer']:
                                self.corr += 1
                        self.cnt += 1
                        question_point['gpt_response'] = response
                        question_point['whiteBox_responses'] = white_response
                        self.question_summary.append(question_point)
                        gpt_p_bar.update(1)
                        break
                    except Exception as e:
                        self.switch_api_key()
                        num_trials += 1
                        if num_trials == max_trials:
                            print(f"Failed to get responses for {datapoint['id']} after {max_trials} trials")
                            break
                        time.sleep(5)
            _gpt_call(datapoint['whiteBox_responses'])

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(whiteBox_query, self.data)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(gpt_query, self.question_responses)
        with open(json_file, "w") as f:
            json.dump(self.question_summary, f, indent=2)
        print(f"Total Accuracy: {self.corr/self.cnt:.3f}")

generator = generate_pipeline()
generator.generate()