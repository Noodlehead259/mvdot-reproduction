import torch
from torch import nn
from torch.nn import functional as F


def normalize_rows(x, eps=1e-12):
    return F.normalize(x, p=2, dim=1, eps=eps)


class ViewAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=128, hidden_dims=(500, 500, 2000)):
        super().__init__()
        dims = [input_dim, *hidden_dims, latent_dim]
        encoder = []
        for i in range(len(dims) - 1):
            encoder.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*encoder)

        decoder_dims = [latent_dim, *reversed(hidden_dims), input_dim]
        decoder = []
        for i in range(len(decoder_dims) - 1):
            decoder.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:
                decoder.append(nn.ReLU(inplace=True))
            elif input_dim == 784:
                decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)


class PaperMvDOT(nn.Module):
    def __init__(self, input_dims, num_clusters=10, latent_dim=128, hidden_dims=(500, 500, 2000)):
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError("MvDOT requires at least two views")
        self.input_dims = list(input_dims)
        self.num_views = len(input_dims)
        self.num_clusters = num_clusters
        self.latent_dim = latent_dim
        self.autoencoders = nn.ModuleList([
            ViewAutoencoder(dim, latent_dim=latent_dim, hidden_dims=hidden_dims)
            for dim in input_dims
        ])
        self.mapping = nn.Linear(latent_dim, latent_dim)
        self.centers = nn.Parameter(normalize_rows(torch.randn(num_clusters, latent_dim)))
        self.register_buffer("masses", torch.full((num_clusters,), 1.0 / num_clusters))

    def encode_views(self, views):
        if len(views) != self.num_views:
            raise ValueError(f"expected {self.num_views} views, got {len(views)}")
        latents = []
        reconstructions = []
        common = []
        for view, autoencoder in zip(views, self.autoencoders):
            z, reconstruction = autoencoder(view)
            h = self.mapping(z)
            latents.append(z)
            reconstructions.append(reconstruction)
            common.append(h)
        return latents, reconstructions, common

    def transport_costs(self, common):
        return [torch.cdist(h, self.centers, p=2).square() for h in common]

    def posteriors(self, common, costs=None, sigma2=None):
        if costs is None:
            costs = self.transport_costs(common)
        if sigma2 is None:
            pooled_costs = torch.cat(costs, dim=0)
            sigma2 = pooled_costs.detach().mean(dim=0).clamp_min(1e-4)
        else:
            sigma2 = sigma2.to(device=common[0].device, dtype=common[0].dtype).clamp_min(1e-4)
        affinities = [torch.exp(-cost / (2.0 * sigma2.unsqueeze(0))) for cost in costs]
        posteriors = [affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1e-12) for affinity in affinities]
        return posteriors, affinities, sigma2

    @torch.no_grad()
    def project_centers_(self):
        self.centers.copy_(normalize_rows(self.centers))

    @torch.no_grad()
    def update_masses_from_plans_(self, plans, momentum=0.98):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        empirical = torch.stack([plan.sum(dim=0) for plan in plans], dim=0).mean(dim=0)
        empirical = empirical / empirical.sum().clamp_min(1e-12)
        updated = momentum * self.masses + (1.0 - momentum) * empirical
        self.masses.copy_(updated / updated.sum().clamp_min(1e-12))
        return empirical

    @torch.no_grad()
    def initialize_spherical_kmeans(self, latent_views, seed=42, iterations=50):
        samples = normalize_rows(torch.cat(latent_views, dim=0).detach())
        if samples.shape[0] < self.num_clusters:
            raise ValueError("number of samples must be at least the number of clusters")
        generator = torch.Generator(device=samples.device)
        generator.manual_seed(seed)
        chosen = torch.randperm(samples.shape[0], generator=generator, device=samples.device)[:self.num_clusters]
        centers = samples[chosen].clone()

        for _ in range(iterations):
            assignments = samples @ centers.T
            labels = assignments.argmax(dim=1)
            new_centers = torch.zeros_like(centers)
            counts = torch.bincount(labels, minlength=self.num_clusters)
            new_centers.index_add_(0, labels, samples)
            nonempty = counts > 0
            new_centers[nonempty] = normalize_rows(new_centers[nonempty])
            if (~nonempty).any():
                confidence = assignments.max(dim=1).values
                replacement = torch.argsort(confidence)[:int((~nonempty).sum().item())]
                new_centers[~nonempty] = samples[replacement]
            if torch.allclose(centers, new_centers, atol=1e-5, rtol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        self.centers.copy_(normalize_rows(centers))
        self.masses.fill_(1.0 / self.num_clusters)

    def forward(self, views, posterior_sigma2=None):
        latents, reconstructions, common = self.encode_views(views)
        costs = self.transport_costs(common)
        posteriors, affinities, sigma2 = self.posteriors(common, costs, sigma2=posterior_sigma2)
        return {
            "latents": latents,
            "reconstructions": reconstructions,
            "common": common,
            "costs": costs,
            "posteriors": posteriors,
            "anchor_affinities": affinities,
            "sigma2": sigma2,
            "centers": self.centers,
            "masses": self.masses,
        }
