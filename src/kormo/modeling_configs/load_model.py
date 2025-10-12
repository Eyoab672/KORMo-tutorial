import yaml
from pathlib import Path
from kormo.model._configuration_kormo import KORMoConfig
from kormo.model._modeling_kormo import KORMoForCausalLM
from transformers import AutoTokenizer
from importlib.resources import files
import torch
import os

config_path = files("kormo.modeling_configs")

def load_model_from_config(num_parameters, **kwargs):
    with Path(f'{config_path}/kormo_{num_parameters}.yaml').open() as f:
        model_config = yaml.safe_load(f)

    model_config.update(**kwargs)
    model_path = os.path.join(config_path, 'initialized_models', f'kormo_{num_parameters}')
    try:
        model = KORMoForCausalLM.from_pretrained(
            model_path,
            **model_config
        )

    except:
        cfg = KORMoConfig(**model_config)
        model = KORMoForCausalLM._from_config(cfg)
        model.save_pretrained(model_path)

    return model, model_config

def load_tokenizer(tokenizer_name_or_path='kormo-lm/KORMo-tokenizer'):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model_and_tokenizer(
    num_parameters,
    **kwargs
):
    model, _ = load_model_from_config(num_parameters, **kwargs)
    return model, load_tokenizer()

def load_model_and_tokenizer_from_checkpoint(
    tokenizer_name_or_path,
    num_parameters='10B',
    ckpt_path=None,
    **kwargs
):
    with Path(f'{config_path}/kormo_{num_parameters}.yaml').open() as f:
        model_cfg = yaml.safe_load(f)
    model_cfg.update(**kwargs)

    torch_dtype = getattr(torch, 'bfloat16')
    model = KORMoForCausalLM.from_pretrained(ckpt_path, dtype=torch_dtype, **kwargs)
    return model, model_cfg, load_tokenizer(tokenizer_name_or_path)


def compute_param_count(config):
    V = config['vocab_size']
    H = config['hidden_size']
    I = config['intermediate_size']
    L = config['num_hidden_layers']
    N = config['num_attention_heads']

    K = config.get('num_key_value_heads', N)
    tie = config.get('tie_word_embeddings', False)
    use_bias = config.get('use_bias', False)

    D = config.get('head_dim', H // N)
    if 'head_dim' in config and H != N * D:
        print(f"Warning: hidden_size={H} != num_attention_heads({N})*head_dim({config['head_dim']}), using D={D}.")

    emb_params = V * H
    lm_head_params = 0 if tie else V * H

    q_params = H * (N * D)
    k_params = H * (K * D)
    v_params = H * (K * D)
    o_params = (N * D) * H
    attn_params = q_params + k_params + v_params + o_params

    mlp_params = 2 * (H * I) + (I * H)

    norm_params = 2 * H

    per_layer = attn_params + mlp_params + norm_params

    total = emb_params + lm_head_params + per_layer * L
    total += H

    return total