import random
from datasets import load_dataset
import re


def render_mc(question, letters, choices):
    """Render a multiple-choice question (format: choice before letter)."""
    query = f'Multiple choice question: {question}\n'
    query += "".join([f'- {choice}={letter}\n' for letter, choice in zip(letters, choices)])
    query += '\nRespond only with the letter of the correct answer.'
    return query


class SmolTalk:
    """General conversational data: HuggingFaceTB/smol-smoltalk (460K train, 24K test)."""

    def __init__(self, split):
        self.ds = load_dataset(
            'HuggingFaceTB/smol-smoltalk', split=split).shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return {'messages': self.ds[idx]['messages']}


class MMLUTask:
    letters = ('A', 'B', 'C', 'D')

    def __init__(self, subset, split):
        self.ds = load_dataset('cais/mmlu', subset, split=split).shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        user_msg = render_mc(row['question'], self.letters, row['choices'])
        assistant_msg = self.letters[row['answer']]
        return {
            'messages': [
                {'role': 'user', 'content': user_msg},
                {'role': 'assistant', 'content': assistant_msg}
            ]
        }


class GSM8KTask:
    def __init__(self, subset, split):
        self.ds = load_dataset('openai/gsm8k', subset, split=split).shuffle(seed=42)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        answer = re.sub(r'<<[^>]+>>', '', row['answer'])
        return {
            'messages': [
                {'role': 'user', 'content': row['question']},
                {'role': 'assistant', 'content': answer}
            ]
        }


class TaskMixture:
    def __init__(self, tasks):
        self.index_map = []
        for task_idx, task in enumerate(tasks):
            for local_idx in range(len(task)):
                self.index_map.append((task_idx, local_idx))

        rng = random.Random(42)
        rng.shuffle(self.index_map)
        self.tasks = tasks

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        task_idx, local_idx = self.index_map[idx]
        return self.tasks[task_idx][local_idx]
