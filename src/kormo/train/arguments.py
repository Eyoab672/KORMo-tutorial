from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    model_size: Optional[str] = field(default='400M')
    _attn_implementation: Optional[str] = field(default='flex_attention')
    tokenizer_name_or_path: Optional[str] = field(default="KORMo-Team/KORMo-tokenizer")

@dataclass
class DataArguments:
    root_dir: Optional[str] = field(default=None)
    dataset_name: Optional[str] = field(default=None)


@dataclass
class KORMoTrainingArguments(TrainingArguments):
    save_only_model: Optional[bool] = field(default=False)
    do_train : Optional[bool] = field(default=True)
    do_eval : Optional[bool] = field(default=False)
    bf16 : Optional[bool] = field(default=True)
    num_train_epochs : Optional[int] = field(default=1)
    eval_strategy : Optional[str] = field(default="no")
    eval_steps : Optional[int] = field(default=1000)
    logging_strategy: Optional[str] = field(default="steps")
    logging_steps: Optional[int] = field(default=1)
    logging_first_step: Optional[bool] = field(default=True)
    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=500) 
    save_total_limit: Optional[int] = field(default=10000)
    per_device_train_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    warmup_ratio: Optional[float] = field(default=0.03)
    ddp_find_unused_parameters: Optional[bool] = field(default=True)
    overwrite_output_dir: Optional[bool] = field(default=True)
    remove_unused_columns: Optional[bool] = field(default=True)
    gradient_checkpointing: Optional[bool] = field(default=False)
    optim: Optional[str] = field(default="adamw_torch")
    include_num_input_tokens_seen: Optional[bool] = field(default=True)
    lr_scheduler_type: Optional[str] = field(default="warmup_stable_decay")
    learning_rate : Optional[float] = field(default=5e-4)
    weight_decay : Optional[float] = field(default=0.033)
    adam_beta1: Optional[float] = field(default=0.9)
    adam_beta2: Optional[float] = field(default=0.95)
    adam_epsilon: Optional[float] = field(default=1e-8) # from OLMo2
    hf_hub_repo_id: Optional[str] = field(default=None)
    hf_hub_token: Optional[str] = field(default=None)

    resume_from_checkpoint: Optional[str] = field(default=None)


@dataclass
class Config:
    data: DataArguments = field(default_factory=DataArguments)
    model: ModelArguments = field(default_factory=ModelArguments)
    train: KORMoTrainingArguments = field(default_factory=KORMoTrainingArguments)
    def __post_init__(self):
        if isinstance(self.data, dict):
            self.data = DataArguments(**self.data)
        if isinstance(self.model, dict):
            self.model = ModelArguments(**self.model)
        if isinstance(self.train, dict):
            self.train = KORMoTrainingArguments(**self.train)