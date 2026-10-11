import pytest
import torch
from losses.reconstruction import ReconstructionLoss


def test_reconstruction_loss():
    views = [
        torch.randn(8, 20),
        torch.randn(8, 30)
    ]

    reconstructions = [
        view + 0.1 for view in views
    ]

    loss = ReconstructionLoss()(reconstructions, views)

    assert loss.ndim == 0
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_reconstruction_gradients():
    views = [torch.randn(8, 20)]
    reconstruction = torch.randn(8, 20, requires_grad=True)

    loss = ReconstructionLoss()([reconstruction], views)
    loss.backward()

    assert reconstruction.grad is not None
    assert torch.isfinite(reconstruction.grad).all()


def test_reconstruction_shape_mismatch():
    with pytest.raises(ValueError):
        ReconstructionLoss()(
            [torch.randn(8, 10)],
            [torch.randn(8, 20)]
        )