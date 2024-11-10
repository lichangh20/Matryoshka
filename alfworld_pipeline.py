# Indicate the LLM version that you want to use``
# 'gpt3.5' / 'gpt4o_mini'
GPT_MODEL = 'gpt3.5'
GPT_PATTERN =  r"```python(.*?)```"
ITER = 1 # Sampling Iterations (whether it's appropriate?)


# after NUM_TRY_RESET trials, the agent will try to start from step 1.
NUM_TRY_RESET = 3

# By default, the max number of tokens is 1400. In some tasks, the context limit of the language models is exceeded. Try to change the number of tokens in these cases.
MAX_TOKENS = 1400

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}

# save to files
LOG_FILE = 'interaction_log.txt'
PR_FILE = 'Prompt_Response.txt'
EXEC_FILE = 'execution_log.txt'

import os
from openai import AzureOpenAI, OpenAI
import re
import yaml
import alfworld
import alfworld.agents.environment
from datetime import datetime
import time
import sys
import io
import traceback
from models.credentials import gpt_4o_mini_api_key_list, chatgpt_0125_api_key_list, gpt_4o_api_key_list
import openai
import json

import argparse

def parse_opt():
    parser = argparse.ArgumentParser(description='whiteBox decompose task, chatgpt inference on Alfworld')
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vllm whiteBox port",
    )
    parser.add_argument(
        "--num-try",
        type=int,
        default=6,
        help="GPT try num",
    )
    parser.add_argument("--whitebox", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help="Use for vllm client. If use lora, set to lora id, otherwise set to base model name")
    parser.add_argument("--use_gpt", action="store_true", default=False, help="Whether exclude f(x) in closed-loop refinement")
    args = parser.parse_args()
    return args

args = parse_opt()

# max number of trials for each task
NUM_TRY = args.num_try

# decompose whitebox model
whiteBox_api_key = ""
whiteBox_api_base = f"http://localhost:{args.port}/v1"
whitebox = 'LLaMA3' if args.whitebox == 'meta-llama/Meta-Llama-3-8B-Instruct' else args.whitebox

os.environ['ALFWORLD_DATA'] = 'path/to/data'

with open('base_config.yaml') as reader:
    config = yaml.safe_load(reader)

split = "eval_out_of_distribution"

env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
game_num = len(env.game_files)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def start_episode(env):
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    ob = process_ob(ob)
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    return ob, name

def interact_with_env(action):
    observation, reward, done, info = env.step([action])
    observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
    return observation, reward, done

def extract_receptacles_and_task(text):
    # Extract list of receptacles
    receptacles_pattern = r"\b\w+ \d+\b"
    receptacles = re.findall(receptacles_pattern, text)

    # Extract the task
    task_pattern = r"Your task is to:.*"
    task = re.search(task_pattern, text).group(0)

    return receptacles, task

def extract_general_plan_and_steps(text):
    # Extract the general plan
    general_plan_pattern = r"# General.*"
    general_plan = re.findall(general_plan_pattern, text)

    # Extract the steps
    steps_pattern = r"# \[Step \d+\].*"
    steps = re.findall(steps_pattern, text)
    decompose = '\n'.join(general_plan) + '\n' + '\n'.join(steps)

    return decompose.strip()

def extract_answers(text, markers):
    answers = []
    for i, marker in enumerate(markers):
        if i < len(markers) - 1:
            next_marker = markers[i + 1]
            pattern = fr"{re.escape(marker)}\s*([\s\S]*?)\n*{re.escape(next_marker)}"
        else:
            pattern = fr"{re.escape(marker)}\s*([\s\S]*)"
        
        answer = re.search(pattern, text)
        if answer:
            answer = answer.group(1).strip()
            answers.append(answer)
        else:
            answers.append('Not found')
        
    return answers

def save_to_file(directory, filename, content):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    file_path = os.path.join(directory, filename)
    with open(filename, 'a') as file:
        file.write(content + '\n')

def clear_file_content(filename):
    with open(filename, 'w') as file:
        pass

def dict_to_traj(d):
    traj = ''
    for i, a in enumerate(d['actions']):
        traj += f'Act {i}: {a}\nObs {i}: {d["observations"][i]}\n'
    return traj

def get_line_starting_with(lines, start):
    for line in lines.split('\n'):
        if line.strip().startswith(start):
            return line
    return None

def get_error_step(error_message):
    pattern = r'\[Step (\d+)\]'
    match = re.search(pattern, error_message)
    if match:
        return int(match.group(1))
    else:
        return None
    
def get_first_digit(input_string):
    pattern = r'\d'
    match = re.search(pattern, input_string)
    if match:
        return int(match.group())
    else:
        return None
    
datetime = datetime.now().strftime("%m%d-%H%M%S")
save_to_folder = f'./r_white{whitebox}_black{GPT_MODEL}_Try{args.num_try}_{datetime}/'
PR_FILE =  save_to_folder + PR_FILE
LOG_FILE = save_to_folder + LOG_FILE
EXEC_FILE = save_to_folder + EXEC_FILE

save_to_file(save_to_folder, PR_FILE, 'Start: ' + str(datetime) + '\n')
save_to_file(save_to_folder, LOG_FILE, 'Start: ' + str(datetime) + '\n')
save_to_file(save_to_folder, EXEC_FILE, 'Start: ' + str(datetime) + '\n')

# this function captures the assertion error message and stores the local variables inside the solution at the breakpoint.
def capture_output(func, agent, step=1):
    # Store the original standard output and standard error
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Redirect the standard output and error to in-memory file-like objects
    temp_stdout = io.StringIO()
    temp_stderr = io.StringIO()
    sys.stdout = temp_stdout
    sys.stderr = temp_stderr

    checkpoint = None

    # import IPython
    # IPython.embed()
    
    # Run the function and capture exceptions
    try:
        func(agent, start_from=step)
    except Exception as e:
        traceback.print_exc()
        # Safely capture locals of the exception frame if traceback exists
        checkpoint = sys.exc_info()[2].tb_next.tb_frame.f_locals
    # Restore the original standard output and error
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # Get the output and error messages as strings
    output_string = temp_stdout.getvalue()
    error_string = temp_stderr.getvalue()

    # print('Ouput:')
    print(output_string)
    print(error_string)
    return error_string, checkpoint, output_string + error_string

# this function modifies the header of the solution to load the variables from the breakpoint
def modify_header(checkpoint):
    if not checkpoint:
        return 'def solution(agent, start_from=1):'
    load_checkpoint = ''
    skip_vars = ['agent', 'start_from']
    for k,v in checkpoint.items():
        if k not in skip_vars:
            if type(v) == str:
                load_checkpoint += f', {k}="{v}"'
            else:
                load_checkpoint += f', {k}={v}'
    header = f'def solution(agent, start_from{load_checkpoint}):'
    return header

def parse_solution_to_decompose(solution_func: str, add_general:bool = True) -> str:
    general_plan_pattern = r"# General [pP]lan.*"
    general_plan = re.findall(general_plan_pattern, solution_func)

    step_pattern = r'print\(\"(\[Step \d+\].*?)\"\)'
    steps = re.findall(step_pattern, solution_func)  
    if not steps:
        step_pattern = r'# (\[?Step \d+\]?).*'
        steps = re.findall(step_pattern, solution_func)  

    decompose = ''
    if general_plan and add_general:
        decompose = general_plan[0] + '\n'
    for step in steps:
        if '#' not in step:
            decompose += '# ' + step + '\n'
        else:
            decompose += step + '\n'
    return decompose.strip()



if GPT_MODEL == 'gpt4o_mini':
    api_key_list = gpt_4o_mini_api_key_list
    model_type = 'gpt-4o-mini'
elif GPT_MODEL == 'gpt3.5':
    api_key_list = chatgpt_0125_api_key_list
    model_type = 'gpt-3.5-turbo'
api_idx = 0
gpt_client = AzureOpenAI(
    azure_endpoint = api_key_list[api_idx]['azure_endpoint'],
    api_key = api_key_list[api_idx]['api_key'],
    api_version = api_key_list[api_idx]['api_version'],
)
engine = api_key_list[api_idx]['engine']
whitebox_clinet = OpenAI(
    api_key=whiteBox_api_key,
    base_url=whiteBox_api_base,
)
whitebox_model = args.whitebox
whiteox_list = [data.id for data in whitebox_clinet.models.list()]
assert whitebox_model in whiteox_list, f'whitebox model {whitebox_model} not found in {whiteox_list}'
print(f'Choose whitebox model {whitebox_model} from {whiteox_list}')

def switch_api_key():
    global api_idx, gpt_client, engine
    api_idx = (api_idx + 1) % len(api_key_list)
    gpt_client = AzureOpenAI(
        api_key = api_key_list[api_idx]['api_key'],
        api_version = api_key_list[api_idx]['api_version'],
        azure_endpoint = api_key_list[api_idx]['azure_endpoint'],
    )
    engine = api_key_list[api_idx]['engine']

def ask_whitebox(prompt, mark=''):
    prompt_chat = [
        {"role": "user", "content": prompt.strip()},
    ]
    num_trials = 0
    max_trials = 3
    while num_trials < max_trials:
        try:
            response = whitebox_clinet.chat.completions.create(
                model=whitebox_model,
                messages=prompt_chat,
                temperature=0.0,
                max_tokens=MAX_TOKENS,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            answer = response.choices[0].message.content.strip()
            # answer = extract_general_plan_and_steps(answer)
            save_to_file(save_to_folder,
                             PR_FILE, 
                            f'[{mark}] Prompt: \n' +
                            prompt +
                            f'Decompose: \n' +
                            answer + '\n' + '*'*20 + '\n')
            return answer
        except Exception as e:
            num_trials += 1
            print(e)
            if num_trials > 3:
                print(f"Retry exceed the max_retries {num_trials} times.")
                break
            time.sleep(10)

def ask(prompt, mark=''):
    prompt_chat = [
            {"role": "user", "content": prompt.strip()},
        ]
    cnt = 0
    while True:
        try:
            if GPT_MODEL == 'gpt3.5' or GPT_MODEL == 'gpt4o_mini':
                response = gpt_client.chat.completions.create(
                    model=engine,
                    messages=prompt_chat,
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                answer = response.choices[0].message.content.strip()
                matches = re.findall(GPT_PATTERN, answer, re.DOTALL)
                if len(matches) > 0:
                    answer = matches[0].strip()
                save_to_file(save_to_folder,
                             PR_FILE, 
                            f'[{mark}] Prompt: \n' +
                            prompt +
                            '\nResponse: \n' +
                            answer + '\n' + '='*20 + '\n')
        
                return answer
            else:
                raise Exception('Wrong GPT_MODEL')
        except openai.RateLimitError as e:
            switch_api_key()
            retry_after = 3
            print(f"Rate limit error: {e}. Retrying in {retry_after} seconds.")
            time.sleep(retry_after)
        except openai.BadRequestError as e:
            switch_api_key()
            # try to eliminate some parts of the prompt to reduce the number of tokens
            eliminate_context = "# for example: You have a list of receptacles, and you want to sort them by the likelihood of a soapbar appearing in them. You can do this by asking the assistant:\nreceptacles = ['countertop 1', 'garbagecan 1', 'sinkbasin 2', 'sinkbasin 1', 'toilet 1', 'toiletpaperhanger 1', 'towelholder 1']\nanswer = ask(f'Sort the list of receptacles, starting from the one a soapbar is most likely to appear: {receptacles}. You should return a Python list.')\n# answer = ['sinkbasin 1', 'sinkbasin 2', 'countertop 1', 'towelholder 1', 'toiletpaperhanger 1', 'garbagecan 1', 'toilet 1']"
            prompt = prompt.replace(eliminate_context, '')
            prompt_chat = [
                {"role": "user", "content": prompt.strip()},
            ]
            print(f"Exceed max: {e}.")
            cnt += 1
            if cnt > 3:
                return 'Exceed max limit. Tried 3 times. Skip this one.'
        except openai.APIError:
            switch_api_key()
            cnt += 1
            if cnt > 3:
                return 'APIError. Tried 3 times. Skip this one.'
        except Exception as e:
            switch_api_key()
            print(f"An unexpected error occurred: {e}")
            raise

from ast import literal_eval
from alfworld_prompt import simple_decompose, puttwo_decompose, examine_decompose, clean_decompose, heat_decompose, cool_decompose, whitebox_prompt, get_solution_prompt, code_check_prompt, feedback_fix_prompt, simple_example, puttwo_example, examine_example, clean_example, heat_example, cool_example, get_start_from_prompt, feedback_gptfix_prompt

# Agent class represents the state of the agent, including its location,
# what it's holding, and the states of receptacles and objects in the environment,
# as well as the actions it can take.
class Agent:
    def __init__(self, receptacles):
        self.location = None
        self.holding = None
        self.receptacles = receptacles
        self.interaction_history = {'actions': [], 'observations': []}
        self.is_success = False

    # Note down the history of interactions with the environment
    def add_to_history(self, action, observation):
        self.interaction_history['actions'].append(action)
        self.interaction_history['observations'].append(observation)
        
    # Get an observation from the environment after performing an action, and add it to the history
    def observation(self, action):
        observation, reward, done = interact_with_env(action)
        self.add_to_history(action, observation)
        print(f'Act: {action}\nObs: {observation}')
        if done:
            self.is_success = reward
            print('Done. Success:', reward)
        return observation

    # Go to a receptacle and update the agent's location. It returns an observation in natural language.
    # For example, 'On the countertop 1, you see a candle 1, a cloth 2, and a soapbar 1.' = goto('countertop 1')
    def goto(self, receptacle):
        self.location = receptacle
        return self.observation(f'go to {receptacle}')

    # Take an object from a receptacle if the agent is not holding anything. It returns an observation in natural language.
    # For example, 'You pick up the soapbar 1 from the towelholder 1.' = take('soapbar 1', 'towelholder 1')
    def take(self, object, receptacle):
        if self.holding is None:
            self.holding = object
            return self.observation(f'take {object} from {receptacle}')
        
    # Put an object in or on a receptacle if the agent is holding it. It returns an observation in natural language.
    # For example, 'You put the soapbar 1 in/on the cabinet 1.' = put('soapbar 1', 'cabinet 1')
    def put(self, object, receptacle):
        if self.holding == object:
            self.holding = None
            return self.observation(f'put {object} in/on {receptacle}')

    # Open a receptacle and observe its contents. It returns an observation in natural language.
    # For example, 'You open the cabinet 1. The cabinet 1 is open. In it, you see a cloth 1.' = open_receptacle('cabinet 1')
    def open_receptacle(self, receptacle):
        return self.observation(f'open {receptacle}')

    # Close an opened receptacle. It returns an observation in natural language.
    # For example, 'You close the safe 1.' = close_receptacle('safe 1')
    def close_receptacle(self, receptacle):
        return self.observation(f'close {receptacle}')

    # Clean an object with a receptacle. It returns an observation in natural language.
    # For example, 'You clean the soapbar 1 using the sinkbasin 1.' = clean('soapbar 1', 'sinkbasin 1')
    def clean(self, object, receptacle):
        return self.observation(f'clean {object} with {receptacle}')

    # Heat an object with a receptacle. It returns an observation in natural language.
    # For example, 'You heat the tomato 1 using the microwave 1.' = heat('tomato 1', 'microwave 1')
    def heat(self, object, receptacle):
        return self.observation(f'heat {object} with {receptacle}')

    # Cool an object with a receptacle. It returns an observation in natural language.
    # For example, 'You cool the pan 2 using the fridge 1.' = cool('pan 2', 'fridge 1')
    def cool(self, object, receptacle):
        return self.observation(f'cool {object} with {receptacle}')

    # Turn_on an object. It returns an observation in natural language.
    # For example, 'You turn on the desklamp 1.' = turn_on('desklamp 1')
    def turn_on(self, object):
        return self.observation(f'use {object}')
    
    # Report agent's current state, including its location, what it's holding, and last three actions and observations.
    # This function should only be used for assertion.
    def report(self):
        msg = \
f'''The last three interactions before error were:
Act: {self.interaction_history["actions"][-3]}
Obs: {self.interaction_history["observations"][-3]}
Act: {self.interaction_history["actions"][-2]}
Obs: {self.interaction_history["observations"][-2]}
Act: {self.interaction_history["actions"][-1]}
Obs: {self.interaction_history["observations"][-1]}
I am at {self.location} and holding {self.holding}.
'''.strip()
        return msg
    

# dict for storing the failed_tasks_id task numbers w.r.t. each task type

failed_tasks_id = {task_name: [] for task_name in prefixes.keys()}
num_refinement = {task_name: [] for task_name in prefixes.keys()}
all_tasks_id = {task_name: [] for task_name in prefixes.keys()}

# adapted from ReAct code.
cnts = [0] * 6
rs = [0] * 6

for iteration in range(ITER):
    # for _ in range(134):
    print(f'We sample {game_num} games in total.')
    for _ in range(game_num):
    # for _ in range(5):
        terminal_output = ''
        description, task_name = start_episode(env)
        receptacle_list, task = extract_receptacles_and_task(description)
        # print('description:', description)
        print('task_name:', task_name)
        # print('receptacle_list:', receptacle_list)
        print('task:', task)
        # define environment 
        agent = Agent(receptacle_list)

        if task_name.startswith('pick_two_obj'):
            example = puttwo_example
            decompose = puttwo_decompose
        elif task_name.startswith('look_at_obj'):
            example = examine_example
            decompose = examine_decompose
        elif task_name.startswith('pick_and_place'):
            example = simple_example
            decompose = simple_decompose
        elif task_name.startswith('pick_clean_then_place'):
            example = clean_example
            decompose = clean_decompose
        elif task_name.startswith('pick_heat_then_place'):
            example = heat_example
            decompose = heat_decompose
        elif task_name.startswith('pick_cool_then_place'):
            example = cool_example
            decompose = cool_decompose
        else:
            print('pass!')
            print(task_name)

        dpo_prompt = whitebox_prompt\
                    .replace('<decompose>', decompose)\
                    .replace('<receptacle_list>', str(receptacle_list))\
                    .replace('<task>', task)
        
        # get the decompose result
        decompose_prompt = whitebox_prompt\
                            .replace('<decompose>', decompose)\
                            .replace('<receptacle_list>', str(receptacle_list))\
                            .replace('<task>', task)
        decomposition = ask_whitebox(decompose_prompt, f'Get Task{_+1} Decomposition')

        # get the solution function
        prompt = get_solution_prompt\
                .replace('<receptacle_list>', str(receptacle_list))\
                .replace('<task>', task)\
                .replace('<example>', example)\
                .replace('<decomposition>', decomposition)
        response = ask(prompt, f'Get Task{_+1} Solution')

        # refine internally                
        solution_func = '''
        def solution(agent, start_from=1):
            <solution>
        '''.strip().replace('<solution>', response) if not response.startswith('def solution(agent, start_from=1):') else response

        prompt = code_check_prompt\
                .replace('<solution_func>', solution_func)
        response = ask(prompt, f'Code Check Task{_+1}')
        answers = extract_answers(response, ['[Decision]', '[Revised code]'])
        # if there is a No after [1]:
        if 'Yes' in answers[0]:
            print('Fix error in solution function')
            solution_func = answers[1].strip('```').replace('Revised code:', '').strip()

        # formalize the solution function
        solution_func = solution_func.replace('CD', 'cd').replace('solution(agent)','').replace('solution(agent)','').replace('<EOC>', '')\
                                        .replace('print("Task completed successfully!")', '').replace('print(agent.report())', '').replace('receptacles =', '# receptacles =')   
        matches = re.findall(GPT_PATTERN, solution_func, re.DOTALL)
        if len(matches) > 0:
            solution_func = matches[0].strip()
    
        start_num = None
        for num_try in range(NUM_TRY):
            if num_try < NUM_TRY_RESET:
                step = start_num if start_num else 1
            else:
                step = 1
            print('start_from_step:', start_num)
            # execute the solution function
            def_error = False
            try:
                # save_to_file(save_to_folder, EXEC_FILE, f'Task {_+1} Try{num_try+1}:\n' + f'Executing solution function with start_from={step}:\n' + solution_func)
                exec(solution_func)
            except Exception as e:
                error_msg = str(e)
                error_string = str(e)
                checkpoint = None
                def_error = True

            if not def_error:
                error_string, checkpoint, output_string = capture_output(solution, agent, step)
                terminal_output += output_string
                if error_string:
                    error_msg = error_string.split('\n')[4:]
                    error_msg = '\n'.join(error_msg)
                else:
                    error_msg = 'You executed the solution function successfully but the task is not completed. Please check your solution function.'
            
            start_num = None

            if agent.is_success:
                break

            prev_solution_func = solution_func
            print("Captured error:", error_string.strip())
            save_to_file(save_to_folder, EXEC_FILE, f'Task {_+1} Try{num_try+1}:\n' + f'Executing solution function with start_from={step}:\n' + solution_func + '\n' + 'Error Message:\n' + error_msg)
            error_step = get_error_step(error_string)

            # refine based on environment feedback
            if args.use_gpt:
                prompt = feedback_gptfix_prompt\
                    .replace('<example>', example)\
                    .replace('<receptacle_list>', str(receptacle_list))\
                    .replace('<task>', task)\
                    .replace('<error_msg>', error_msg)
            else:
                prompt = feedback_fix_prompt\
                        .replace('<example>', example)\
                        .replace('<receptacle_list>', str(receptacle_list))\
                        .replace('<task>', task)\
                        .replace('<decomposition>', decomposition)\
                        .replace('<error_msg>', error_msg)
            response = ask(prompt, f'Feedback Fix Task{_+1} Try{num_try+1}')
            solution_func = '''
                def solution(agent, start_from=1):
                    <solution>
                '''.strip().replace('<solution>', response) if not response.startswith('def solution(agent, start_from=1):') else response

            # formalize the solution function
            solution_func = solution_func.replace('CD', 'cd').replace('solution(agent)','').replace('solution(agent)','').replace('<EOC>', '')\
                                        .replace('print("Task completed successfully!")', '').replace('print(agent.report())', '').replace('receptacles =', '# receptacles =')

            matches = re.findall(GPT_PATTERN, solution_func, re.DOTALL)
            if len(matches) > 0:
                solution_func = matches[0].strip()
            
            prompt = get_start_from_prompt\
                    .replace('<previous_solution>', prev_solution_func)\
                    .replace('<revised_solution>', solution_func)
            response = ask(prompt, f'Get Start From Task{_+1} Try{num_try+1}')
            start_num = get_first_digit(response)
            solution_func = solution_func.replace('def solution(agent, start_from=1):', modify_header(checkpoint))

        # store results
        for i, (k, v) in enumerate(prefixes.items()):
            if task_name.startswith(k):
                rs[i] += agent.is_success
                cnts[i] += 1
                num_refinement[k].append(num_try if agent.is_success else -1)
                all_tasks_id[k].append(_)
                if not agent.is_success:
                    failed_tasks_id[k].append(_)
                break

        save_to_file(save_to_folder, LOG_FILE, 
                    f'Task {_+1}: {task_name}\n' + \
                    description + '\n' + \
                    terminal_output + '\n' + \
                    f'Success: {agent.is_success}\n' + \
                    f'Task {_+1}, rs: {rs} cnts {cnts} success_rate: {sum(rs) / sum(cnts)} \n')
        save_to_file(save_to_folder, LOG_FILE, f'failed_tasks_id {failed_tasks_id}')
        save_to_file(save_to_folder, LOG_FILE, f'num_refinement {num_refinement}')
        save_to_file(save_to_folder, LOG_FILE, f'all_tasks_id {all_tasks_id}')
        save_to_file(save_to_folder, LOG_FILE, f'------------\n')

        print(_+1, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
        print('------------\n')

    print('failed_tasks_id', failed_tasks_id)
    print('num_refinement', num_refinement)
    print('all_tasks_id', all_tasks_id)
