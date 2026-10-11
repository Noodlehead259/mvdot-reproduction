import torch
from models.autoencoder import ViewAutoencoder


def test_autoencoder_shapes():
    model = ViewAutoencoder(input_dim=784, hidden_dim=256, latent_dim=64)
    x = torch.randn(32, 784)

    reconstruction, z = model(x)

    assert reconstruction.shape == x.shape
    assert z.shape == (32, 64)


def test_autoencoder_gpu():
    if not torch.cuda.is_available():
        return

    model = ViewAutoencoder(input_dim=784).cuda()
    x = torch.randn(16, 784, device="cuda")

    reconstruction, z = model(x)

    assert reconstruction.is_cuda
    assert z.is_cuda