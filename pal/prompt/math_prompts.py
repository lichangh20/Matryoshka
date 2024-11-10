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

LLAMA_DECOMPOSE_PROMPT = '''
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Let's break down this problem:\nHow much does Olivia spend on bagels?\nHow much money does Olivia have left after the purchase?

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Let's break down this problem:\nHow many golf balls did Michael lose in total by the end of Wednesday?\nHow many golf balls does Michael have left after losing the total amount?

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: Let's break down this problem:\nHow many computers were added in total from Monday to Thursday?\nHow many computers are now in the server room after adding the new ones?

Q: {question}
A:
'''.strip()

LLAMA_SYSTEM_MESSAGE = '''
You will decompose a math problem into smaller parts. Follow the prompt instruction and do not generate redundant information.
'''

MATH_CHAT_ANNOTATION_DECOMPOSE_BETA_SYSTEM_MESSAGE = 'You will write python program to solve math problems. You will write annotations and code blocks following instructions. Annotations should be written in the form of a question.'

MATH_CHAT_MODIFY_ERROR_PROMPT = '''
Let's use python to solve math problems. Here are three successful cases on how to do it,
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
```
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
```

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
```
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result
```

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
```
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday
    computers_added = computers_per_day * num_days
    computers_total = computers_initial + computers_added
    result = computers_total
    return result
```

# Here is the actual question.
Q: {question}
You have generated code of solution() to solve the task. However, you executed the solution() function and get an error message:
{error}

Referring to the successful case and the error message, you should complete the solution function with the correct code.
'''.strip()


MATH_CHAT_ANNOTATION_DECOMPOSE_BETA_PROMPT = '''
Let's use python to solve math problems. Here are three examples how to do it,
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Let's break down this problem:\nHow much does Olivia spend on bagels?\nHow much money does Olivia have left after the purchase?
```
def solution():
    """Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"""
    # Initialization of variables
    money_initial = 23
    bagels = 5
    bagel_cost = 3

    # How much does Olivia spend on bagels?
    money_spent = bagels * bagel_cost

    # How much money does Olivia have left after the purchase?
    money_left = money_initial - money_spent
    result = money_left
    return result
```

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
Let's break down this problem:\nHow many golf balls did Michael lose in total by the end of Wednesday?\nHow many golf balls does Michael have left after losing the total amount?
```
def solution():
    """Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"""
    # Initialization of variables
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2

    # How many golf balls did Michael lose in total by the end of Wednesday?
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday

    # How many golf balls does Michael have left after losing the total amount?
    result = golf_balls_left
    return result
```

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Let's break down this problem:\nHow many computers were added in total from Monday to Thursday?\nHow many computers are now in the server room after adding the new ones?
```
def solution():
    """There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"""
    # Initialization of variables
    computers_initial = 9
    computers_per_day = 5
    num_days = 4  # 4 days between monday and thursday

    # How many computers were added in total from Monday to Thursday?
    computers_added = computers_per_day * num_days

    # How many computers are now in the server room after adding the new ones?
    computers_total = computers_initial + computers_added
    result = computers_total
    return result
```

How about this question?
Q: {question}
{decompose}
'''.strip()
# Please write annotations and code blocks to solve the problem. Annotations should be written in the form of a question.