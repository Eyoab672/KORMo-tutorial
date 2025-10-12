from typing import Optional
from transformers.loss.loss_utils import ForCausalLMLoss, fixed_cross_entropy
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_lm_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
):  
    ce_loss = ForCausalLMLoss(
        logits,
        labels,
        vocab_size,
        num_items_in_batch,
        ignore_index,
        shift_labels,
        **kwargs,
    )
    return ce_loss, ce_loss, torch.tensor(0.0, device=logits.device)

def causal_lm_with_zloss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    *,
    z_loss_multiplier: float = 1e-5,
    **kwargs,
) -> torch.Tensor:
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    ### Compute cross-entropy ###
    logits_ce = logits.view(-1, vocab_size)
    shift_labels_ce = shift_labels.view(-1)
    shift_labels_ce = shift_labels_ce.to(logits_ce.device)

    ce_loss = fixed_cross_entropy(logits_ce, shift_labels_ce, num_items_in_batch, ignore_index, **kwargs)

    ### Compute z-loss ###
    log_z = torch.logsumexp(logits, dim=-1)
    mask  = (shift_labels != ignore_index).float()

    if num_items_in_batch is not None:
        z_loss = (log_z.pow(2) * mask).sum() / num_items_in_batch
    else:
        z_loss = (log_z.pow(2) * mask).mean()

    z_loss = z_loss_multiplier * z_loss
    total_loss = ce_loss + z_loss
    return total_loss, ce_loss, z_loss