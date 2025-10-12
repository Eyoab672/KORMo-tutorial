from dataclasses import dataclass
from transformers import PreTrainedTokenizer
import torch

K = 1024

@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizer
    
    def __init__(self):
        self.think_token = self.tokenizer.decode("<think>")

    def __call__(self, instances):
        input_ids = [instance["input_ids"][:4*K] for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels[labels == 125033] = -100 ## <think>

        return dict(
            input_ids=input_ids,
            labels=labels,
        )