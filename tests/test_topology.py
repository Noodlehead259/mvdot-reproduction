import pytest
import torch

from losses.topology import TopologyLoss


def test_topology_loss_is_finite():
    embeddings = [torch.randn(8, 16), torch.randn(8, 16)]

    loss = TopologyLoss()(embeddings)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_topology_loss_backpropagates():
    left = torch.randn(8, 16, requires_grad=True)
    right = torch.randn(8, 16, requires_grad=True)

    TopologyLoss()([left, right]).backward()

    assert left.grad is not None
    assert right.grad is not None


def test_topology_loss_with_transport_plan():
    embeddings = [torch.randn(8, 16), torch.randn(8, 16)]
    plan = torch.eye(8) / 8

    loss = TopologyLoss()(
        embeddings,
        [[None, plan], [None, None]]
    )

    assert torch.isfinite(loss)


def test_topology_loss_requires_two_views():
    with pytest.raises(ValueError):
        TopologyLoss()([torch.randn(8, 16)])


def test_topology_loss_requires_two_samples():
    with pytest.raises(ValueError):
        TopologyLoss()([torch.randn(1, 16), torch.randn(1, 16)])


def test_topology_loss_runs_on_gpu():
    if not torch.cuda.is_available():
        return

    embeddings = [
        torch.randn(8, 16, device="cuda"),
        torch.randn(8, 16, device="cuda")
    ]

    loss = TopologyLoss()(embeddings)

    assert loss.is_cuda
    assert torch.isfinite(loss)