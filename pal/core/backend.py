# Copyright 2022 PAL Authors. All rights reserved.
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

import openai
from openai import AzureOpenAI
import time
import os
from .credentials import chatgpt_0125_api_key_list, gpt_4o_api_key_list, gpt_4o_mini_api_key_list

openai.api_key = os.getenv('OPENAI_API_KEY')

# GPT-3 API
def call_gpt(prompt, model='code-davinci-002', stop=None, temperature=0., top_p=1.0,
        max_tokens=128, majority_at=None):
    num_completions = majority_at if majority_at is not None else 1
    num_completions_batch_size = 5
    
        
    completions = []
    for i in range(20 * (num_completions // num_completions_batch_size + 1)):
        try:
            requested_completions = min(num_completions_batch_size, num_completions - len(completions))
            if model.startswith('gpt-4') or model.startswith('gpt-3.5-turbo'):
                ans = chat_api(
                            model=model,
                            max_tokens=max_tokens,
                            stop=stop,
                            prompt=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            n=requested_completions,
                            best_of=requested_completions)
            else:
                ans = completions_api(
                            model=model,
                            max_tokens=max_tokens,
                            stop=stop,
                            prompt=prompt,
                            temperature=temperature,
                            top_p=top_p,
                            n=requested_completions,
                            best_of=requested_completions)
            completions.extend(ans)
            if len(completions) >= num_completions:
                return completions[:num_completions]
        except openai.error.RateLimitError as e:
            time.sleep(min(i**2, 60))
    raise RuntimeError('Failed to call GPT API')

def completions_api(model, max_tokens, stop, prompt, temperature,
            top_p, n, best_of):
    ans = openai.Completion.create(
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        n=n,
        best_of=best_of)
    return [choice['text'] for choice in ans['choices']]

def chat_api(model, max_tokens, stop, prompt, temperature,
            top_p, n, best_of):
    ans = openai.ChatCompletion.create(
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant that can write Python code that solves mathematical reasoning questions similarly to the examples that you will be provided.'},
            {'role': 'user', 'content': prompt}],
        temperature=temperature,
        top_p=top_p,
        n=n)
    return [choice['message']['content'] for choice in ans['choices']]


def call_chat_gpt(messages, model='gpt-3.5-turbo', stop=None, temperature=0., top_p=1.0, max_tokens=128):
    wait = 1
    while True:
        try:
            ans = openai.ChatCompletion.create(
                model=model,
                max_tokens=max_tokens,
                stop=stop,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                n=1
            )
            return ans.choices[0]['message']['content']
        except openai.error.RateLimitError as e:
            time.sleep(min(wait, 60))
            wait *= 2
    raise RuntimeError('Failed to call chat gpt')


def call_whitebox(messages, client, model, temperature=0., top_p=1.0, max_tokens=128, num_returns=1):
    num_trials = 0
    max_trials = 3
    while num_trials < max_trials:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=num_returns
            )
            if num_returns > 1:
                return [choice.message.content.strip() for choice in response.choices]
            answer = response.choices[0].message.content.strip()
            # answer = extract_general_plan_and_steps(answer)
            return answer
        except Exception as e:
            num_trials += 1
            print(e)
            if num_trials > 3:
                print(f"Retry exceed the max_retries {num_trials} times.")
                break
            time.sleep(4)
    

class GPT_MODEL():
    def __init__(self, model='gpt-3.5-turbo'):
        self.api_idx = 0
        if model == 'gpt-3.5-turbo':
            self.api_key_list = chatgpt_0125_api_key_list
        elif model == 'gpt4o_mini':
            self.api_key_list = gpt_4o_mini_api_key_list
        elif model == 'gpt4o':
            self.api_key_list = gpt_4o_api_key_list
        self.gpt_client = AzureOpenAI(
            azure_endpoint = self.api_key_list[self.api_idx]['azure_endpoint'],
            api_key = self.api_key_list[self.api_idx]['api_key'],
            api_version = self.api_key_list[self.api_idx]['api_version'],
        )
        self.engine = self.api_key_list[self.api_idx]['engine']
    
    def switch_api_key(self):
        self.api_idx = (self.api_idx + 1) % len(self.api_key_list)
        self.gpt_client = AzureOpenAI(
            api_key = self.api_key_list[self.api_idx]['api_key'],
            api_version = self.api_key_list[self.api_idx]['api_version'],
            azure_endpoint = self.api_key_list[self.api_idx]['azure_endpoint'],
        )
        self.engine = self.api_key_list[self.api_idx]['engine']

    def call_chat_gpt(self, messages, stop=None, temperature=0., top_p=1.0, max_tokens=128):
        num_trials = 0
        max_trials = 3
        while num_trials < max_trials:
            try:
                ans = self.gpt_client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    n=1,
                )
                return ans.choices[0].message.content
            except Exception as e:
                self.switch_api_key()
                num_trials += 1
                print(e)
                if num_trials > 3:
                    print(f"Retry exceed the max_retries {num_trials} times.")
                    break
                time.sleep(5)