import pytest
import torch

from losses.matching import MatchingLoss


def test_matching_loss_is_finite():
    loss_fn = MatchingLoss()
    embeddings = [torch.randn(16, 32), torch.randn(16, 32)]

    loss = loss_fn(embeddings)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_matching_loss_backpropagates():
    loss_fn = MatchingLoss()
    left = torch.randn(8, 16, requires_grad=True)
    right = torch.randn(8, 16, requires_grad=True)

    loss_fn([left, right]).backward()

    assert left.grad is not None
    assert right.grad is not None


def test_matching_loss_requires_two_views():
    with pytest.raises(ValueError):
        MatchingLoss()([torch.randn(8, 16)])


def test_matching_loss_rejects_invalid_temperature():
    with pytest.raises(ValueError):
        MatchingLoss(temperature=0)


def test_matching_loss_with_transport_plan():
    loss_fn = MatchingLoss()
    embeddings = [torch.randn(8, 16), torch.randn(8, 16)]
    plan = torch.eye(8) / 8

    loss = loss_fn(embeddings, [[None, plan], [None, None]])

    assert torch.isfinite(loss)


def test_matching_loss_runs_on_gpu():
    if not torch.cuda.is_available():
        return

    embeddings = [
        torch.randn(8, 16, device="cuda"),
        torch.randn(8, 16, device="cuda")
    ]

    loss = MatchingLoss()(embeddings)

    assert loss.is_cuda
    assert torch.isfinite(loss)