import argparse
import random
import os
from tqdm import tqdm
from utils.json_utils import load_jsonlines
from utils.general_utils import seed_everything
from pal.prompt.lamp_prompts import create_black_prompts, create_white_prompts
import concurrent.futures
from models.credentials import chatgpt_api_key_list, chatgpt_0125_api_key_list, gpt_4o_api_key_list, gpt_4o_mini_api_key_list
# set up time format
import time
import re
from openai import OpenAI, AzureOpenAI
import json
import torch
import numpy as np

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

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
    parser.add_argument("--dataset", type=str, choices=['train', 'test'], default='train', help="choose which dataset to generate")
    parser.add_argument("--add_answer", action="store_true", default=False, help="whether add answer in the decomposed intermediate prompt")    
    parser.add_argument("--gpt-engine", type=str, default='gpt4o_mini', choices=['gpt4o', 'gpt4o_mini', 'gpt3.5_0125', 'gpt3.5_1106'], help="choose the gpt engine version to use")
    args = parser.parse_args()
    return args

args = parse_opt()
seed_everything(args.seed)

#===================== Params to change =====================#
whiteBox_api_key = "whitebox"
whiteBox_api_base = f"http://localhost:{args.port}/v1"

task = "LaMP-1"
index = task.split('-')[1].lower()
train_set = load_jsonlines(f"data/lamp/formalized/lamp{index}/formalized_train_data.jsonl")
test_set = load_jsonlines(f"data/lamp/formalized/lamp{index}/formalized_dev_data.jsonl")
output_dir = f"data/lamp/generated/lamp{index}"
save_path = os.path.join(output_dir)
os.makedirs(save_path, exist_ok=True)
json_file = f"{save_path}/{task}.json"
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
        self.api_key_list = chatgpt_0125_api_key_list
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
        self.gpt_client_for_negative = AzureOpenAI(
            azure_endpoint = gpt_4o_mini_api_key_list[self.api_idx]['azure_endpoint'],
            api_key = gpt_4o_mini_api_key_list[self.api_idx]['api_key'],
            api_version = gpt_4o_mini_api_key_list[self.api_idx]['api_version'],
        )
        self.engine = self.api_key_list[self.api_idx]['engine']
        self.engine_for_negative = gpt_4o_mini_api_key_list[self.api_idx]['engine']
        self.whiteBox_client = OpenAI(
            api_key=whiteBox_api_key,
            base_url=whiteBox_api_base,
        )
        self.whiteBox_model = self.whiteBox_client.models.list().data[0].id
        print(self.whiteBox_model)
        self.max_length = 512
        self.data = train_set if args.dataset == 'train' else test_set
        self.question_responses = [] # question and decomposed subquestions
        self.json_file = json_file
        self.num_returns = 10
        self.cnt = self.corr = self.cost = 0
        print(f'Model: {self.whiteBox_model}')
        self.contrsative_pairs = []
        self.system_choice = 1 if args.add_answer else 0
        self.whiteBox_responses = {}

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
        print('Start generating {} data...'.format(len(self.data)))
        whitebox_p_bar = tqdm(range(len(self.data)), desc="WhiteBox processing")
        gpt_p_bar = tqdm(range(len(self.data)), desc="GPT processing")

        def whiteBox_query(datapoint):
            num_trials = 0
            max_trials = 8
            messages_generator = create_white_prompts(datapoint['question'], num_profs, is_ranked=False, use_all=False)
            messages = messages_generator(datapoint['question'], datapoint['profile'], task)
            messages = [{'role':'user', 'content': messages}]
            while num_trials < max_trials:
                try:
                    raw_response = self.whiteBox_client.chat.completions.create(
                        model=self.whiteBox_model,
                        messages=messages,
                        max_tokens=self.max_length,
                        n=self.num_returns,
                        temperature=1.0
                    )
                    responses = [choice.message.content for choice in raw_response.choices] 
                    question_responses = datapoint.copy()
                    # self.whiteBox_responses[datapoint['id']] = responses
                    question_responses['whiteBox_responses'] = responses
                    question_responses['whiteBox_prompt'] = messages
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
            contrastive_pair = {}
            contrastive_pair['question'] = datapoint['question']
            contrastive_pair['answer'] = datapoint['answer']
            # contrastive_pair['profile'] = datapoint['profile']
            contrastive_pair['whiteBox_prompt'] = datapoint['whiteBox_prompt']
            contrastive_pair['positive'] = []
            contrastive_pair['negative'] = []
            self.cnt += 1

            def _gpt_call(white_response):
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
                            n=1,
                            temperature=0.0
                        )
                        response = raw_response.choices[0].message.content
                        if task == 'LaMP-1':
                            match = re.search(r'\[(\d+)\]', response)
                            if match:
                                answer = match.group(0)  # This includes the brackets [i]
                            else:
                                answer = None  # or some default value if no match is found
                        elif task == 'LaMP-2N' or task == 'LaMP-2M':
                            answer = response
                        elif task == 'LaMP-3' or task == 'LaMP-4':
                            answer = datapoint['answer']
                        if answer == datapoint['answer']:
                            contrastive_pair['positive'].append({'decompose': white_response, 'response': response})
                        else:
                            contrastive_pair['negative'].append({'decompose': white_response, 'response': response})
                        gpt_p_bar.update(1)
                        break
                    except Exception as e:
                        self.switch_api_key()
                        num_trials += 1
                        if num_trials == max_trials:
                            print(f"Failed to get responses for {datapoint['id']} after {max_trials} trials")
                            break
                        time.sleep(5)
            for j in range(self.num_returns):
                _gpt_call(datapoint['whiteBox_responses'][j])
            if len(contrastive_pair['positive']) == 0:
                messages_generator = create_white_prompts(datapoint['question'], num_profs, is_ranked=False, use_all=False)
                messages = messages_generator(datapoint['question'], datapoint['profile'], task)
                messages = [{'role':'user', 'content': messages}]
                raw_response = self.gpt_client_for_negative.chat.completions.create(
                    model=self.engine_for_negative,
                    messages=messages,
                    max_tokens=self.max_length,
                    n=1,
                    temperature=1.0
                )
                white_response_strong = [choice.message.content for choice in raw_response.choices] 
                for j in range(5):
                    _gpt_call(white_response_strong[j])
            self.contrsative_pairs.append(contrastive_pair)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(whiteBox_query, self.data)
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(gpt_query, self.question_responses)
        with open(self.json_file, "w") as f:
            json.dump(self.contrsative_pairs, f, indent=2)
      

generator = generate_pipeline()
generator.generate()