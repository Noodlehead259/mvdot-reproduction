import torch
from models.mapping import MappingNetwork


def test_mapping_output_shape():
    model = MappingNetwork(input_dim=64, hidden_dim=128, output_dim=32)
    x = torch.randn(16, 64)

    output = model(x)

    assert output.shape == (16, 32)


def test_mapping_gpu():
    if not torch.cuda.is_available():
        return

    model = MappingNetwork().cuda()
    x = torch.randn(8, 128, device="cuda")

    output = model(x)

    assert output.shape == (8, 128)
    assert output.is_cuda