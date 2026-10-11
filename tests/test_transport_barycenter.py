import torch
from models.transport_barycenter import TransportBarycenter


def test_barycenter_shapes_and_masses():
    model = TransportBarycenter(num_clusters=10, embedding_dim=64)
    embeddings = torch.randn(32, 64)

    cost, masses = model(embeddings)

    assert cost.shape == (32, 10)
    assert masses.shape == (10,)
    assert torch.all(masses > 0)
    assert torch.isclose(masses.sum(), torch.tensor(1.0))


def test_barycenter_parameters_receive_gradients():
    model = TransportBarycenter(num_clusters=5, embedding_dim=32)
    embeddings = torch.randn(16, 32)

    cost, masses = model(embeddings)
    loss = cost.mean() + masses.square().sum()
    loss.backward()

    assert model.centers.grad is not None
    assert model.mass_logits.grad is not None


def test_barycenter_gpu():
    if not torch.cuda.is_available():
        return

    model = TransportBarycenter(10, 64).cuda()
    embeddings = torch.randn(32, 64, device="cuda")

    cost, masses = model(embeddings)

    assert cost.is_cuda
    assert masses.is_cuda