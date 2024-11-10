from torch.utils.data import Dataset
from tqdm import tqdm
from utils.gsm8k_prompt import nshot_decompose_instructwhiteBox
from utils.gsm8k_prompt_singleturn import SingleTurn_decompose_instuction


def preprocess_data(data, input_template='decompose', input_key="input", apply_chat_template=None) -> str:
    assert apply_chat_template
    if input_template == 'decompose':
        # messages = nshot_decompose_instructwhiteBox(data[input_key])
        messages = SingleTurn_decompose_instuction(data[input_key], 1)
    else:
        raise NotImplementedError
    prompt = apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template='decompose',
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.n_samples_per_prompt = getattr(self.strategy.args, "n_samples_per_prompt", 1)

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        self.prompts = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt = preprocess_data(data, input_template, input_key, apply_chat_template)
            self.prompts.append(prompt)

        self.ground_truths = []
        for data in tqdm(dataset, desc="Loading ground truths", disable=not self.strategy.is_rank_0()):
            self.ground_truths.append(data["answer"])

        self.questions = []
        for data in tqdm(dataset, desc="Loading questions", disable=not self.strategy.is_rank_0()):
            self.questions.append(data["question"])

    def __len__(self):
        length = len(self.prompts)
        return length * self.n_samples_per_prompt

    def __getitem__(self, idx):
        return {'prompt': self.prompts[idx // self.n_samples_per_prompt], 'ground_truth': self.ground_truths[idx // self.n_samples_per_prompt], 'question': self.questions[idx // self.n_samples_per_prompt]}
