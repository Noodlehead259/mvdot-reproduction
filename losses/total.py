import torch
import torch.nn as nn

from losses.reconstruction import ReconstructionLoss
from losses.matching import MatchingLoss
from losses.topology import TopologyLoss


class TotalLoss(nn.Module):
    def __init__(
        self,
        reconstruction_weight=1.0,
        matching_weight=1.0,
        topology_weight=1.0,
    ):
        super().__init__()

        self.reconstruction_weight = reconstruction_weight
        self.matching_weight = matching_weight
        self.topology_weight = topology_weight

        self.reconstruction_loss = ReconstructionLoss()
        self.matching_loss = MatchingLoss()
        self.topology_loss = TopologyLoss()

    def forward(
        self,
        reconstructions,
        views,
        embeddings,
        transport_plans=None,
    ):
        reconstruction = self.reconstruction_loss(
            reconstructions,
            views,
        )

        matching = self.matching_loss(
            embeddings,
            transport_plans,
        )

        topology = self.topology_loss(
            embeddings,
            transport_plans,
        )

        total = (
            self.reconstruction_weight * reconstruction
            + self.matching_weight * matching
            + self.topology_weight * topology
        )

        return {
            "total": total,
            "reconstruction": reconstruction,
            "matching": matching,
            "topology": topology,
        }