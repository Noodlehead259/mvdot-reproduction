import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, reconstructions, views):
        if len(reconstructions) != len(views):
            raise ValueError(
                "reconstructions and views must have the same length"
            )

        if len(views) == 0:
            raise ValueError("at least one view is required")

        losses = []

        for reconstruction, view in zip(reconstructions, views):
            if reconstruction.shape != view.shape:
                raise ValueError(
                    "each reconstruction must match its input shape"
                )

            losses.append(
                torch.mean((reconstruction - view) ** 2)
            )

        return torch.stack(losses).mean()