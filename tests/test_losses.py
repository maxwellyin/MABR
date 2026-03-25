import torch

from mabr.losses import debiased_focal_loss, focal_loss


def test_focal_loss_is_positive():
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.5]])
    labels = torch.tensor([0, 1])
    loss = focal_loss(logits, labels, gamma=2.0)
    assert loss.item() > 0


def test_debiased_focal_loss_returns_scalar():
    logits = torch.tensor([[2.0, 0.5], [0.1, 1.5]])
    labels = torch.tensor([0, 1])
    bias_probs = torch.tensor([[0.2], [0.8]])
    loss = debiased_focal_loss(logits, labels, gamma=2.0, bias_probs=bias_probs)
    assert loss.ndim == 0
