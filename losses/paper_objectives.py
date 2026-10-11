import numpy as np
import torch
from torch.nn import functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from ot.paper_sinkhorn import sinkhorn


def reconstruction_loss(views, reconstructions):
    if len(views) != len(reconstructions):
        raise ValueError("views and reconstructions must have the same length")
    return sum(F.mse_loss(reconstruction, view) for view, reconstruction in zip(views, reconstructions))


def _knn_laplacian(latent, neighbors=10):
    n = latent.shape[0]
    if n < 2:
        return latent.new_zeros((n, n))
    k = min(max(1, neighbors), n - 1)
    with torch.no_grad():
        distances = torch.cdist(latent.detach(), latent.detach(), p=2).square()
        distances.fill_diagonal_(float("inf"))
        knn_distances, knn_indices = torch.topk(distances, k=k, dim=1, largest=False)
        mask = torch.zeros((n, n), dtype=torch.bool, device=latent.device)
        mask.scatter_(1, knn_indices, True)
        mask = mask | mask.T
        finite_knn = knn_distances[torch.isfinite(knn_distances)]
        scale = finite_knn.median().clamp_min(1e-8) if finite_knn.numel() else distances.new_tensor(1.0)
        weights = torch.exp(-distances / scale)
        weights = torch.where(mask & torch.isfinite(distances), weights, torch.zeros_like(weights))
        weights = (weights + weights.T) * 0.5
        weights.fill_diagonal_(0.0)
    return weights


def graph_laplacian_loss(latent, neighbors=10):
    weights = _knn_laplacian(latent, neighbors=neighbors)
    degree = weights.sum(dim=1)
    laplacian_product = degree.unsqueeze(1) * latent - weights @ latent
    return (latent * laplacian_product).sum() / max(latent.shape[0], 1)


def discrete_transport_loss(costs, plans, latents, graph_weight=0.05, graph_neighbors=10):
    if not (len(costs) == len(plans) == len(latents)):
        raise ValueError("costs, plans and latents must have the same length")
    transport = sum((plan * cost).sum() for plan, cost in zip(plans, costs))
    graph = sum(graph_laplacian_loss(z, graph_neighbors) for z in latents)
    return transport + graph_weight * graph, transport, graph


def _cross_view_hops(affinity, neighbors=10):
    affinity = affinity.detach()
    n, m = affinity.shape
    if n != m:
        raise ValueError("bipartite matching graph expects equal sample counts in both views")
    if n < 2:
        return np.ones((n, n), dtype=np.float32)
    k = min(max(1, neighbors), n)
    values, indices = torch.topk(affinity, k=k, dim=1, largest=True)
    reverse_values, reverse_indices = torch.topk(affinity, k=k, dim=0, largest=True)
    row_ids = torch.arange(n, device=affinity.device).unsqueeze(1).expand_as(indices)
    reverse_row_ids = reverse_indices
    reverse_col_ids = torch.arange(n, device=affinity.device).unsqueeze(0).expand_as(reverse_indices)

    adjacency = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    row_ids_np = row_ids.cpu().numpy().ravel()
    col_ids_np = indices.cpu().numpy().ravel() + n
    adjacency[row_ids_np, col_ids_np] = 1
    adjacency[col_ids_np, row_ids_np] = 1
    reverse_rows_np = reverse_row_ids.cpu().numpy().ravel()
    reverse_cols_np = reverse_col_ids.cpu().numpy().ravel() + n
    adjacency[reverse_rows_np, reverse_cols_np] = 1
    adjacency[reverse_cols_np, reverse_rows_np] = 1

    graph = csr_matrix(adjacency)
    hops = shortest_path(graph, directed=False, unweighted=True, indices=np.arange(n))[:, n:]
    hops[~np.isfinite(hops)] = np.inf
    topological_distance = np.empty_like(hops, dtype=np.float32)
    finite = np.isfinite(hops)
    topological_distance[finite] = 1.0 / (1.0 + np.exp(-hops[finite]))
    topological_distance[~finite] = 1.0
    return topological_distance


def semantic_topology_cost(affinity_m, affinity_n, posterior_m, posterior_n, graph_neighbors=10):
    cross_affinity = affinity_m @ affinity_n.T
    overlap = posterior_m @ posterior_n.T
    semantic_distance = (1.0 - overlap).clamp(0.0, 1.0)
    hops_distance = _cross_view_hops(cross_affinity, neighbors=graph_neighbors)
    topology_distance = torch.as_tensor(hops_distance, device=semantic_distance.device, dtype=semantic_distance.dtype)
    return semantic_distance * topology_distance, overlap


def unpaired_matching_loss(posteriors, affinities, costs, plans, aligned_mask=None, graph_neighbors=10, matching_epsilon=0.1, sinkhorn_iterations=60, log_epsilon=1e-8):
    """Paper-style sample-matching loss over the unaligned subset only.

    Equation (35) sums over unpaired samples. Known aligned pairs are not given
    an additional direct matching loss here; adding one would be an extra
    objective not specified by Eq. (35) or Eq. (38).
    """
    num_views = len(posteriors)
    if num_views < 2:
        return costs[0].new_zeros(()), [], []

    batch_size = posteriors[0].shape[0]
    device = posteriors[0].device
    if aligned_mask is None:
        aligned_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    else:
        aligned_mask = aligned_mask.to(device=device, dtype=torch.bool)
    unpaired_indices = torch.nonzero(~aligned_mask, as_tuple=False).flatten()
    if unpaired_indices.numel() < 1:
        return costs[0].new_zeros(()), [], []

    ot_distances = torch.stack([(plan * cost).sum().detach() for plan, cost in zip(plans, costs)])
    view_reliability = (ot_distances + 1e-8).reciprocal()
    view_reliability = view_reliability / view_reliability.sum().clamp_min(1e-12)

    total = costs[0].new_zeros(())
    matrices = []
    pair_costs = []
    size = unpaired_indices.numel()
    marginal = torch.full((size,), 1.0 / size, device=device, dtype=costs[0].dtype)

    for m in range(num_views):
        for n in range(m + 1, num_views):
            q_m = posteriors[m].index_select(0, unpaired_indices)
            q_n = posteriors[n].index_select(0, unpaired_indices)
            g_m = affinities[m].index_select(0, unpaired_indices)
            g_n = affinities[n].index_select(0, unpaired_indices)
            matching_cost, overlap = semantic_topology_cost(
                g_m, g_n, q_m, q_n, graph_neighbors
            )
            matching = sinkhorn(
                matching_cost, marginal, marginal,
                epsilon=matching_epsilon, iterations=sinkhorn_iterations,
            )
            pair_loss = -(matching * torch.log(overlap.clamp_min(log_epsilon))).sum()
            pair_loss = pair_loss * (size / batch_size)
            total = total + 2.0 * view_reliability[m] * view_reliability[n] * pair_loss
            matrices.append(matching)
            pair_costs.append(matching_cost)

    return total, matrices, pair_costs


def global_unpaired_matching_loss(posteriors, global_matching_state, sample_indices, batch_size):
    """Minibatch gradient for the full-pool sparse matching matrix.

    The sparse matching P and opposite-view posterior cache are refreshed once
    per epoch over the complete unaligned pool. This function uses live batch
    posteriors on each side in alternating directions, while the opposite-side
    cache and P remain fixed for the epoch. This is a memory-feasible stochastic
    approximation to Eq. (35); it avoids solving a new independent P per batch.
    """
    if len(posteriors) != 2:
        raise ValueError("global matching is implemented for the two-view dataset")
    device = posteriors[0].device
    dtype = posteriors[0].dtype
    lookup = global_matching_state["unaligned_lookup"]
    batch_ids = sample_indices.detach().cpu().numpy().astype(np.int64, copy=False)
    offsets = lookup[batch_ids]
    local_offsets = [(local_pos, int(offset)) for local_pos, offset in enumerate(offsets) if offset >= 0]
    if not local_offsets or global_matching_state["rows"].numel() == 0:
        return posteriors[0].new_zeros(()), [], []

    row_ptr = global_matching_state["row_ptr"]
    col_ptr = global_matching_state["col_ptr"]
    col_order = global_matching_state["col_order"]
    rows = global_matching_state["rows"]
    cols = global_matching_state["cols"]
    row_weights = global_matching_state["row_weights"]
    col_weights = global_matching_state["col_weights"]
    cached_left, cached_right = global_matching_state["global_posteriors"]

    row_edge_ids = []
    row_local_ids = []
    col_edge_ids = []
    col_local_ids = []
    for local_pos, offset in local_offsets:
        row_start, row_end = int(row_ptr[offset]), int(row_ptr[offset + 1])
        if row_end > row_start:
            count = row_end - row_start
            row_edge_ids.extend(range(row_start, row_end))
            row_local_ids.extend([local_pos] * count)
        col_start, col_end = int(col_ptr[offset]), int(col_ptr[offset + 1])
        if col_end > col_start:
            edge_ids = col_order[col_start:col_end].tolist()
            col_edge_ids.extend(edge_ids)
            col_local_ids.extend([local_pos] * len(edge_ids))

    valid_count = len(local_offsets)
    fraction = valid_count / max(int(batch_size), 1)
    view_weights = global_matching_state["view_weights"]
    reliability = 2.0 * view_weights[0] * view_weights[1]

    row_loss = posteriors[0].new_zeros(())
    if row_edge_ids:
        edge_ids = torch.as_tensor(row_edge_ids, dtype=torch.long, device=device)
        local_ids = torch.as_tensor(row_local_ids, dtype=torch.long, device=device)
        overlap = (posteriors[0][local_ids] * cached_right[cols[edge_ids]]).sum(dim=1)
        row_loss = -(row_weights[edge_ids] * overlap.clamp_min(1e-8).log()).sum() / valid_count

    col_loss = posteriors[0].new_zeros(())
    if col_edge_ids:
        edge_ids = torch.as_tensor(col_edge_ids, dtype=torch.long, device=device)
        local_ids = torch.as_tensor(col_local_ids, dtype=torch.long, device=device)
        overlap = (cached_left[rows[edge_ids]] * posteriors[1][local_ids]).sum(dim=1)
        col_loss = -(col_weights[edge_ids] * overlap.clamp_min(1e-8).log()).sum() / valid_count

    loss = reliability * 0.5 * (row_loss + col_loss) * fraction
    return loss, [], []

def total_loss(
    views,
    outputs,
    plans,
    graph_weight=0.05,
    graph_neighbors=10,
    aligned_mask=None,
    matching_epsilon=0.1,
    sinkhorn_iterations=60,
    global_matching_state=None,
    sample_indices=None,
):
    rec = reconstruction_loss(views, outputs["reconstructions"])
    manifold, transport, graph = discrete_transport_loss(
        outputs["costs"], plans, outputs["latents"], graph_weight=graph_weight, graph_neighbors=graph_neighbors
    )
    if global_matching_state is not None:
        if sample_indices is None:
            raise ValueError("sample_indices are required with a global matching state")
        matching, matching_matrices, matching_costs = global_unpaired_matching_loss(
            outputs["posteriors"], global_matching_state, sample_indices, views[0].shape[0]
        )
    else:
        matching, matching_matrices, matching_costs = unpaired_matching_loss(
            outputs["posteriors"],
            outputs["anchor_affinities"],
            outputs["costs"],
            plans,
            aligned_mask=aligned_mask,
            graph_neighbors=graph_neighbors,
            matching_epsilon=matching_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
        )
    combined = rec + manifold + matching
    metrics = {
        "total_loss": combined,
        "reconstruction_loss": rec,
        "manifold_matching_loss": manifold,
        "transport_loss": transport,
        "graph_loss": graph,
        "sample_matching_loss": matching,
    }
    return combined, metrics, matching_matrices, matching_costs
