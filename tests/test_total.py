import pytest
import torch

from losses.total import TotalLoss


def create_inputs(batch_size=8, input_dims=(20, 30), latent_dim=16):
    views = [
        torch.randn(batch_size, dim)
        for dim in input_dims
    ]

    reconstructions = [
        torch.randn(batch_size, dim, requires_grad=True)
        for dim in input_dims
    ]

    embeddings = [
        torch.randn(batch_size, latent_dim, requires_grad=True)
        for _ in input_dims
    ]

    return reconstructions, views, embeddings


def test_total_loss_returns_all_components():
    reconstructions, views, embeddings = create_inputs()
    criterion = TotalLoss()

    losses = criterion(reconstructions, views, embeddings)

    assert set(losses) == {
        "total",
        "reconstruction",
        "matching",
        "topology",
    }

    for loss in losses.values():
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_total_loss_combines_weighted_components():
    reconstructions, views, embeddings = create_inputs()

    criterion = TotalLoss(
        reconstruction_weight=2.0,
        matching_weight=0.5,
        topology_weight=3.0,
    )

    losses = criterion(reconstructions, views, embeddings)

    expected = (
        2.0 * losses["reconstruction"]
        + 0.5 * losses["matching"]
        + 3.0 * losses["topology"]
    )

    assert torch.allclose(losses["total"], expected)


def test_total_loss_backpropagates():
    reconstructions, views, embeddings = create_inputs()
    criterion = TotalLoss()

    losses = criterion(reconstructions, views, embeddings)
    losses["total"].backward()

    for reconstruction in reconstructions:
        assert reconstruction.grad is not None
        assert torch.isfinite(reconstruction.grad).all()

    for embedding in embeddings:
        assert embedding.grad is not None
        assert torch.isfinite(embedding.grad).all()


def test_total_loss_runs_on_gpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    reconstructions, views, embeddings = create_inputs()

    reconstructions = [x.detach().cuda().requires_grad_(True) for x in reconstructions]
    views = [x.cuda() for x in views]
    embeddings = [x.detach().cuda().requires_grad_(True) for x in embeddings]

    criterion = TotalLoss().cuda()
    losses = criterion(reconstructions, views, embeddings)

    assert losses["total"].is_cuda
    assert torch.isfinite(losses["total"])

    losses["total"].backward()

    for embedding in embeddings:
        assert embedding.grad is not None
        assert torch.isfinite(embedding.grad).all()