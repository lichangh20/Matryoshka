import json
import os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import re
import sys
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")    
HF_TOKEN="your_huggingface_token"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pal.prompt.lamp_prompts import create_white_prompts

#======================= The params to change for different tasks =======================
task = "LaMP-1"
index = task.split('-')[1].lower()  
with open(f"data/lamp/generated/lamp{index}/{task}_processed.json") as f:
    data_all = json.load(f)
OUTPUT_FILE = f"data/lamp/processed/lamp{index}/{task}_processed.json"
output_dir = f"data/lamp/processed_length/lamp{index}"
#=======================================================================================

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

def wrap_dataset(datapoint, decompose) -> dict:
    white_prompt_generator = create_white_prompts(datapoint['question'], num_profs, is_ranked=False, use_all=False)
    white_prompt = white_prompt_generator(datapoint['question'], datapoint['profile'], task)
    chats = []
    chats.append({"role": "user", "content": white_prompt})
    chats.append({"role": "assistant", "content": decompose['decompose']})
    return chats

def wrap_dataset_with_white_prompt(datapoint, decompose) -> dict:
    chats = datapoint['whiteBox_prompt'].copy()
    chats.append({"role": "assistant", "content": decompose['decompose']})
    return chats
# KIND = ['train', 'test']

data_all = []
output_data = []

for i, data in tqdm(enumerate(data_all), desc="Processing data"):
    if len(data["positive"]) == 0 or len(data["negative"]) == 0:
        continue
    cnt = min(min(len(data["positive"]), len(data["negative"])), 2)
    positive_decompose = [data["negative"][-1]]
    # print(positive_decompose[0])
    positive_decompose = [wrap_dataset_with_white_prompt(data, d) for d in positive_decompose]
    negative_decompose = [data["positive"][0]]
    negative_decompose = [wrap_dataset_with_white_prompt(data, d) for d in negative_decompose]
    if data["positive"][0]['response'] == data["negative"][-1]['response']:
        continue
    for positive, negative in zip(positive_decompose, negative_decompose):
        output_data.append({"rejected": negative, "chosen": positive})
print(f"len of output_data: {len(output_data)}")
with open(OUTPUT_FILE, "w") as f:
    for data in output_data:
        f.write(json.dumps(data) + "\n")

import os


os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "train.jsonl")
sum = 0
with open(OUTPUT_FILE, "r") as f_in, open(output_file, "w") as f_out:
    for line in tqdm(f_in):
        data = json.loads(line)
        rejected = data['rejected']
        chosen = data['chosen']
        rejected_tokens = tokenizer.encode(str(rejected))
        chosen_tokens = tokenizer.encode(str(chosen))
        
        if len(rejected_tokens) <= 5500 and len(chosen_tokens) <= 5500:
            sum += 1
            f_out.write(line)  # Write the original line as is

print(f"Filtered data saved to {output_file}")
print(f"Number of data points: {sum}")