from pathlib import Path

import torch
from torch.utils.data import Dataset


class NoisyMNISTDataset(Dataset):
    def __init__(self, path=None):
        if path is None:
            path = Path(__file__).resolve().parent / "noisymnist.pt"

        data = torch.load(path, map_location="cpu", weights_only=True)

        self.view1 = data["view1"].float()
        self.view2 = data["view2"].float()
        self.labels = data["labels"].long()

        if self.view1.shape != (70000, 784):
            raise ValueError(f"Unexpected view1 shape: {self.view1.shape}")

        if self.view2.shape != (70000, 784):
            raise ValueError(f"Unexpected view2 shape: {self.view2.shape}")

        if self.labels.shape != (70000,):
            raise ValueError(f"Unexpected labels shape: {self.labels.shape}")

        if not torch.isfinite(self.view1).all():
            raise ValueError("view1 contains non-finite values")

        if not torch.isfinite(self.view2).all():
            raise ValueError("view2 contains non-finite values")

        if self.labels.min() < 0 or self.labels.max() > 9:
            raise ValueError("Labels must be digits from 0 to 9")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.view1[index], self.view2[index], self.labels[index]