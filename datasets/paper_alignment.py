from pathlib import Path

import torch
from torch.utils.data import Dataset


class PaperAlignedRateDataset(Dataset):
    def __init__(self, path="datasets/noisymnist.pt", aligned_rate=0.0, seed=42):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        if not 0.0 <= aligned_rate <= 1.0:
            raise ValueError("aligned_rate must be between 0 and 1")

        try:
            data = torch.load(self.path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(self.path, map_location="cpu")

        if isinstance(data, dict):
            self.view1 = self._get_tensor(data, ("view1", "view_1", "x1", "view1_data"))
            self.view2 = self._get_tensor(data, ("view2", "view_2", "x2", "view2_data"))
            self.labels = self._get_tensor(data, ("labels", "label", "y", "targets"))
        elif isinstance(data, (tuple, list)) and len(data) >= 3:
            self.view1, self.view2, self.labels = data[:3]
        else:
            raise ValueError("Expected a dictionary or tuple containing view1, view2 and labels")

        self.view1 = self._flatten(self.view1).float().contiguous()
        self.view2 = self._flatten(self.view2).float().contiguous()
        self.labels = self.labels.long().reshape(-1).contiguous()
        if self.view1.shape != self.view2.shape:
            raise ValueError("The two views must have equal sample count and feature dimension")
        if self.view1.shape[0] != self.labels.shape[0]:
            raise ValueError("Sample count does not match labels")
        if not torch.isfinite(self.view1).all() or not torch.isfinite(self.view2).all():
            raise ValueError("Dataset contains non-finite values")
        if self.view1.max() > 1.0 or self.view2.max() > 1.0:
            self.view1 = self.view1 / 255.0
            self.view2 = self.view2 / 255.0

        self.aligned_count = int(round(len(self.labels) * aligned_rate))
        self.aligned_rate = aligned_rate
        self.view2_indices = torch.arange(len(self.labels), dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        remaining = torch.arange(self.aligned_count, len(self.labels), dtype=torch.long)
        if remaining.numel() > 1:
            permutation = torch.randperm(remaining.numel(), generator=generator)
            while torch.any(permutation == torch.arange(remaining.numel())):
                permutation = torch.randperm(remaining.numel(), generator=generator)
            self.view2_indices[remaining] = remaining[permutation]

    @staticmethod
    def _get_tensor(data, candidates):
        for key in candidates:
            if key in data:
                value = data[key]
                if torch.is_tensor(value):
                    return value
        raise KeyError(f"Could not find a tensor for any key in {candidates}")

    @staticmethod
    def _flatten(value):
        tensor = torch.as_tensor(value)
        if tensor.ndim > 2:
            tensor = tensor.reshape(tensor.shape[0], -1)
        if tensor.ndim != 2:
            raise ValueError("Each view must have shape [samples, features]")
        return tensor

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        view2_index = int(self.view2_indices[index])
        return (
            self.view1[index],
            self.view2[view2_index],
            self.labels[index],
            self.labels[view2_index],
            index < self.aligned_count,
            index,
        )
