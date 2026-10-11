import torch
import torch.nn as nn
import torch.nn.functional as F


class TransportBarycenter(nn.Module):
    def __init__(self, num_clusters, embedding_dim=128):
        super().__init__()

        self.centers = nn.Parameter(
            torch.randn(num_clusters, embedding_dim) * 0.02
        )

        self.mass_logits = nn.Parameter(torch.zeros(num_clusters))

    def get_masses(self):
        return F.softmax(self.mass_logits, dim=0)

    def pairwise_cost(self, embeddings):
        return torch.cdist(embeddings, self.centers, p=2).pow(2)

    def forward(self, embeddings):
        cost = self.pairwise_cost(embeddings)
        masses = self.get_masses()
        return cost, masses