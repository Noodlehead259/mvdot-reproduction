import torch
import torch.nn as nn

from models.autoencoder import ViewAutoencoder
from models.mapping import MappingNetwork
from models.transport_barycenter import TransportBarycenter


class MvDOT(nn.Module):
    def __init__(
        self,
        input_dims,
        num_clusters,
        hidden_dim=512,
        latent_dim=128,
        mapping_dim=128,
        mapping_hidden_dim=256
    ):
        super().__init__()

        self.num_views = len(input_dims)

        self.autoencoders = nn.ModuleList([
            ViewAutoencoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim
            )
            for input_dim in input_dims
        ])

        self.mapping = MappingNetwork(
            input_dim=latent_dim,
            hidden_dim=mapping_hidden_dim,
            output_dim=mapping_dim
        )

        self.barycenter = TransportBarycenter(
            num_clusters=num_clusters,
            embedding_dim=mapping_dim
        )

    def forward(self, views):
        if len(views) != self.num_views:
            raise ValueError(
                f"Expected {self.num_views} views, received {len(views)}"
            )

        reconstructions = []
        latent_representations = []
        embeddings = []
        transport_costs = []
        masses = []

        for autoencoder, view in zip(self.autoencoders, views):
            reconstruction, latent = autoencoder(view)
            embedding = self.mapping(latent)
            cost, view_masses = self.barycenter(embedding)

            reconstructions.append(reconstruction)
            latent_representations.append(latent)
            embeddings.append(embedding)
            transport_costs.append(cost)
            masses.append(view_masses)

        return {
            "reconstructions": reconstructions,
            "latents": latent_representations,
            "embeddings": embeddings,
            "transport_costs": transport_costs,
            "masses": masses,
            "cluster_centers": self.barycenter.centers
        }