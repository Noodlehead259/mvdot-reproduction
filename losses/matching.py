import torch
import torch.nn as nn
import torch.nn.functional as f


class MatchingLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature

    def forward(self, embeddings, transport_plan=None):
        if len(embeddings) < 2:
            raise ValueError("at least two views are required")

        batch_size = embeddings[0].shape[0]

        if batch_size == 0:
            raise ValueError("embeddings must have a non-empty batch")

        for embedding in embeddings:
            if embedding.ndim != 2 or embedding.shape[0] != batch_size:
                raise ValueError("all embeddings must be 2D with equal batch sizes")

        loss = embeddings[0].new_zeros(())
        pairs = 0

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                left = f.normalize(embeddings[i], dim=1)
                right = f.normalize(embeddings[j], dim=1)

                logits = left @ right.T / self.temperature
                targets = torch.arange(batch_size, device=logits.device)

                if transport_plan is None:
                    pair_loss = (
                        f.cross_entropy(logits, targets)
                        + f.cross_entropy(logits.T, targets)
                    ) / 2
                else:
                    plan = transport_plan[i][j].to(
                        device=logits.device,
                        dtype=logits.dtype
                    )

                    if plan.shape != logits.shape:
                        raise ValueError(
                            "transport plan shape must match similarity matrix"
                        )

                    plan = plan / plan.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-12)

                    pair_loss = -(
                        plan * f.log_softmax(logits, dim=1)
                    ).sum(dim=1).mean()

                loss = loss + pair_loss
                pairs += 1

        return loss / pairs