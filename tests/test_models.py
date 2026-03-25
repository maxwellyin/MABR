import torch

from mabr.models import BiasDetector, ReverseLayerF


def test_bias_detector_returns_logits():
    detector = BiasDetector(input_dim=4)
    outputs = detector(torch.randn(2, 4))
    assert outputs.shape == (2, 1)


def test_reverse_layer_flips_gradient_sign():
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    y = ReverseLayerF.apply(x, 1.0)
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([[-1.0, -1.0]]))
