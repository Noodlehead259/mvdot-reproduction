import random
from pathlib import Path

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import functional as tf
from torchvision.transforms import InterpolationMode


def main():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)

    root = Path(__file__).resolve().parent / "raw"
    output_path = Path(__file__).resolve().parent / "noisymnist.pt"

    train = MNIST(root=root, train=True, download=True)
    test = MNIST(root=root, train=False, download=True)

    images = torch.cat([train.data, test.data]).float() / 255.0
    labels = torch.cat([train.targets, test.targets])

    generator = torch.Generator().manual_seed(seed)
    angles = torch.empty(len(images)).uniform_(-45, 45, generator=generator)

    view2 = torch.empty_like(images)

    for start in range(0, len(images), 512):
        end = min(start + 512, len(images))
        batch = images[start:end].unsqueeze(1)

        for i in range(len(batch)):
            rotated = tf.rotate(
                batch[i],
                float(angles[start + i]),
                interpolation=InterpolationMode.BILINEAR,
            )
            view2[start + i] = rotated.squeeze(0)

    noise = torch.randn(view2.shape, generator=generator) * 0.2
    view2 = (view2 + noise).clamp(0, 1)

    data = {
        "view1": images.flatten(start_dim=1).contiguous(),
        "view2": view2.flatten(start_dim=1).contiguous(),
        "labels": labels,
        "seed": seed,
        "rotation_range_degrees": [-45, 45],
        "gaussian_noise_std": 0.2,
        "description": (
            "Original MNIST paired with rotated and Gaussian-noised images. "
            "Transformation settings are implementation-defined."
        ),
    }

    torch.save(data, output_path)

    print(f"Saved: {output_path}")
    print(f"Samples: {len(labels)}")
    print(f"View 1 shape: {tuple(data['view1'].shape)}")
    print(f"View 2 shape: {tuple(data['view2'].shape)}")
    print(f"Labels shape: {tuple(labels.shape)}")
    print(f"Classes: {labels.unique().numel()}")


if __name__ == "__main__":
    main()