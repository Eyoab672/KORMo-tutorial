import os
import random
import numpy as np
import torch
from transformers import set_seed
import copy
import json 
import torch.distributed as dist
import os, torch, torch.distributed as dist

def rank0_print(*args, **kwargs):
    if not dist.is_available() or not dist.is_initialized():
        print(*args, **kwargs)
    else:
        if dist.get_rank() == 0:
            print(*args, **kwargs)

def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()

    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])

    nodes = int(os.getenv("SLURM_JOB_NUM_NODES", "1"))
    gpn   = int(os.getenv("GPUS_PER_NODE", torch.cuda.device_count() or 1))
    return nodes * gpn

def set_all_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_large_number(x: float) -> str:
    if x >= 1_000_000_000_000:
        return f"{x / 1_000_000_000_000:.2f}T"
    elif x >= 1_000_000_000:
        return f"{x / 1_000_000_000:.2f}B"
    elif x >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    else:
        return f"{x:.2f}"

def print_once(message: str) -> None:
    if not getattr(print_once, "_has_printed", False):
        print(message)
        print_once._has_printed = True

def get_num_params(model) -> int:
    numel = {n: p.numel() for n, p in model.named_parameters()}
    return sum(numel.values())