import torch
from models.mvdot import MvDOT


def test_mvdot_forward():
    model = MvDOT(
        input_dims=[784, 300],
        num_clusters=10,
        hidden_dim=256,
        latent_dim=64,
        mapping_dim=32,
        mapping_hidden_dim=128
    )

    views = [
        torch.randn(16, 784),
        torch.randn(16, 300)
    ]

    output = model(views)

    assert len(output["reconstructions"]) == 2
    assert output["reconstructions"][0].shape == (16, 784)
    assert output["reconstructions"][1].shape == (16, 300)
    assert output["embeddings"][0].shape == (16, 32)
    assert output["transport_costs"][0].shape == (16, 10)
    assert output["transport_costs"][1].shape == (16, 10)


def test_mvdot_backward():
    model = MvDOT(
        input_dims=[20, 30],
        num_clusters=4,
        hidden_dim=32,
        latent_dim=16,
        mapping_dim=8,
        mapping_hidden_dim=16
    )

    views = [
        torch.randn(8, 20),
        torch.randn(8, 30)
    ]

    output = model(views)

    loss = sum(
        reconstruction.square().mean()
        for reconstruction in output["reconstructions"]
    )

    loss += sum(cost.mean() for cost in output["transport_costs"])
    loss += sum(mass.square().sum() for mass in output["masses"])

    loss.backward()

    assert model.barycenter.centers.grad is not None
    assert model.mapping.network[0].weight.grad is not None


def test_mvdot_gpu():
    if not torch.cuda.is_available():
        return

    model = MvDOT(
        input_dims=[20, 30],
        num_clusters=4,
        hidden_dim=32,
        latent_dim=16,
        mapping_dim=8,
        mapping_hidden_dim=16
    ).cuda()

    views = [
        torch.randn(8, 20, device="cuda"),
        torch.randn(8, 30, device="cuda")
    ]

    output = model(views)

    assert output["embeddings"][0].is_cuda
    assert output["transport_costs"][0].is_cuda