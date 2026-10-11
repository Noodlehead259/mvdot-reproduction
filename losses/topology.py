import torch
import torch.nn as nn
import torch.nn.functional as f


class TopologyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature

    def forward(self, embeddings, transport_plan=None):
        if len(embeddings) < 2:
            raise ValueError("at least two views are required")

        left, right = embeddings[0], embeddings[1]

        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("embeddings must be two-dimensional")

        if left.shape[0] != right.shape[0]:
            raise ValueError("views must contain the same number of samples")

        if left.shape[0] < 2:
            raise ValueError("at least two samples are required")

        left = f.normalize(left, p=2, dim=1)
        right = f.normalize(right, p=2, dim=1)

        left_similarity = left @ left.T
        right_similarity = right @ right.T

        left_similarity = left_similarity / self.temperature
        right_similarity = right_similarity / self.temperature

        left_similarity = left_similarity - left_similarity.max(
            dim=1, keepdim=True
        ).values.detach()

        right_similarity = right_similarity - right_similarity.max(
            dim=1, keepdim=True
        ).values.detach()

        left_probabilities = f.softmax(left_similarity, dim=1)
        right_log_probabilities = f.log_softmax(right_similarity, dim=1)

        left_log_probabilities = f.log_softmax(left_similarity, dim=1)
        right_probabilities = f.softmax(right_similarity, dim=1)

        loss_left = f.kl_div(
            right_log_probabilities,
            left_probabilities,
            reduction="batchmean",
        )

        loss_right = f.kl_div(
            left_log_probabilities,
            right_probabilities,
            reduction="batchmean",
        )

        return 0.5 * (loss_left + loss_right)