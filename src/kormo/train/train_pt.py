import os
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import datasets
from datasets import load_from_disk, concatenate_datasets
from transformers import HfArgumentParser
from kormo.utils import set_all_seeds
from kormo.modeling_configs.load_model import load_model_and_tokenizer
from kormo.train.arguments import Config
from kormo.train.trainer import KORMoTrainer
from kormo.utils import get_num_params, format_large_number, rank0_print
from kormo.train.callbacks import  PushToHubCallback
from kormo.data_utils.collator import DataCollatorForCausalLM
import os
from glob import glob
project_root = Path(__file__).resolve().parent.parent

set_all_seeds(42)

cli = argparse.ArgumentParser(description="KORMo trainer", allow_abbrev=False)
cli.add_argument("--config", required=True, type=str,help="Path to YAML config file")
args = cli.parse_args() 

parser = HfArgumentParser(Config)
(cfg,)  = parser.parse_yaml_file(args.config)

model_args, data_args, train_args = cfg.model, cfg.data, cfg.train

rank0_print(f"tokenizer :: {model_args.tokenizer_name_or_path}")
model, model_cfg, tokenizer = load_model_and_tokenizer(
    model_args.model_size,
    tokenizer_name_or_path=model_args.tokenizer_name_or_path,
    _attn_implementation=model_args._attn_implementation,
)
rank0_print(model)
rank0_print("### Attention implementation:: ", model.config._attn_implementation)


ds = datasets.load_from_disk('/home/work/mlp/mjkim/nemotron-cc-hq-synthetic-packed')
rank0_print(f"Train samples: {ds.num_rows}")
ds.set_format("torch")

train_args.lr_scheduler_kwargs = {
    'num_decay_steps': 0,
    'warmup_type': "linear",
    'min_lr_ratio': 0.05,
    'num_cycles': 1,
}

rank0_print("# Params: ", format_large_number(get_num_params(model)))
rank0_print("Model dtype: ", model.dtype)

callbacks = []

collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
)

trainer = KORMoTrainer(
    model=model,
    args=train_args,
    train_dataset=ds,
    processing_class=tokenizer,
    callbacks=callbacks,
    data_collator=collator,
)

if train_args.resume_from_checkpoint:
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

else:
    trainer.train()
