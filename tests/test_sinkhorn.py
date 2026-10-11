import torch
from ot.sinkhorn import sinkhorn


def test_sinkhorn_marginals():
    cost = torch.tensor([
        [0.1, 0.8, 0.5],
        [0.7, 0.2, 0.9],
        [0.4, 0.6, 0.3]
    ])

    transport = sinkhorn(cost, epsilon=0.1, max_iter=1000)

    expected = torch.full((3,), 1.0 / 3)

    assert torch.allclose(
        transport.sum(dim=1), expected, atol=1e-4
    )

    assert torch.allclose(
        transport.sum(dim=0), expected, atol=1e-4
    )


def test_sinkhorn_on_gpu():
    if not torch.cuda.is_available():
        return

    cost = torch.rand(8, 5, device="cuda")
    transport = sinkhorn(cost)

    assert transport.is_cuda
    assert torch.isfinite(transport).all()