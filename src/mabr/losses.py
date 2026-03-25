from __future__ import annotations

import torch
import torch.nn.functional as F


def focal_loss(inputs: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    log_probs = F.log_softmax(inputs, dim=-1)
    targets = targets.view(-1, 1)
    log_probs = log_probs.gather(1, targets)
    probs = torch.exp(log_probs)
    return (-((1 - probs) ** gamma) * log_probs).mean()


def debiased_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    bias_probs: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(inputs, dim=-1)
    gather_indices = targets.view(-1, 1)
    log_probs = log_probs.gather(1, gather_indices)
    probs = torch.exp(log_probs)
    weighted_loss = torch.exp(-bias_probs) * (-((1 - probs) ** gamma) * log_probs)
    return weighted_loss.mean()
