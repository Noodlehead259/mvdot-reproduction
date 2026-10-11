import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader

from ot.paper_sinkhorn import sinkhorn, sparse_sinkhorn


def clustering_accuracy(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    size = max(int(labels.max()), int(predictions.max())) + 1
    contingency = np.zeros((size, size), dtype=np.int64)
    for prediction, label in zip(predictions, labels):
        contingency[prediction, label] += 1
    rows, cols = linear_sum_assignment(contingency.max() - contingency)
    return contingency[rows, cols].sum() / max(len(labels), 1)


def _topk_pair_edges(left, right, neighbors=10, chunk_size=512):
    """Global top-k cross-view candidate edges, computed without NxN storage."""
    n = left.shape[0]
    if right.shape[0] != n:
        raise ValueError("global cross-view matching expects equal sample counts")
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    k = min(max(1, int(neighbors)), n)
    edge_keys = []
    right_t = right.T.contiguous()
    left_t = left.T.contiguous()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        scores = left[start:end] @ right_t
        indices = scores.topk(k, dim=1, largest=True, sorted=False).indices
        row_ids = torch.arange(start, end, device=left.device).unsqueeze(1).expand_as(indices)
        keys = row_ids * n + indices
        edge_keys.append(keys.reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False))
        del scores, indices, row_ids, keys

        reverse_scores = right[start:end] @ left_t
        reverse_indices = reverse_scores.topk(k, dim=1, largest=True, sorted=False).indices
        reverse_cols = torch.arange(start, end, device=left.device).unsqueeze(1).expand_as(reverse_indices)
        reverse_keys = reverse_indices * n + reverse_cols
        edge_keys.append(reverse_keys.reshape(-1).detach().cpu().numpy().astype(np.int64, copy=False))
        del reverse_scores, reverse_indices, reverse_cols, reverse_keys

    keys = np.unique(np.concatenate(edge_keys))
    return keys // n, keys % n




def _build_global_graph(left_affinity, right_affinity, neighbors=10, chunk_size=512):
    n = left_affinity.shape[0]
    graph_rows, graph_cols = _topk_pair_edges(
        left_affinity, right_affinity, neighbors=neighbors, chunk_size=chunk_size
    )
    values = np.ones(graph_rows.shape[0], dtype=np.uint8)
    bipartite = csr_matrix((values, (graph_rows, graph_cols)), shape=(n, n), dtype=np.uint8)
    adjacency = bmat([[None, bipartite], [bipartite.T, None]], format="csr", dtype=np.uint8)
    _, component_ids = connected_components(adjacency, directed=False, return_labels=True)
    direct_keys = graph_rows.astype(np.int64) * n + graph_cols
    direct_keys.sort()
    return n, graph_rows, graph_cols, direct_keys, component_ids



def _fallback_derangement_edges(n, permutations=4, seed=1729):
    """Low-density support edges that preserve uniform OT feasibility.

    These edges are not treated as matches or given lower costs. They simply
    keep sparse support feasible when cluster-stratified candidates are blocked
    by collapsed/imbalanced predictions. Their actual cost is still computed
    from semantic overlap and the graph topology term.
    """
    if n < 2 or permutations < 1:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed + n)
    base = np.arange(n, dtype=np.int64)
    all_rows = []
    all_cols = []
    for _ in range(permutations):
        candidate = rng.permutation(n)
        attempts = 0
        while np.any(candidate == base) and attempts < 100:
            candidate = rng.permutation(n)
            attempts += 1
        if np.any(candidate == base):
            candidate = np.roll(base, 1)
        all_rows.append(base.copy())
        all_cols.append(candidate.astype(np.int64, copy=False))
    return np.concatenate(all_rows), np.concatenate(all_cols)

def global_sparse_matching(
    posterior_left,
    posterior_right,
    affinity_left,
    affinity_right,
    graph_neighbors=10,
    matching_epsilon=0.1,
    sinkhorn_iterations=200,
    chunk_size=512,
):
    """Global sparse approximation of Eq. (34) over all unaligned samples.

    Candidate support includes sparse cross-view anchor-affinity graph edges
    and a few random derangement edges to keep the sparse support feasible for
    uniform marginals. Those fallback edges receive no cost advantage and are not
    interpreted as known correspondences. No same-index correspondence is
    injected because that would create a false paired prior when views are
    unaligned. The paper
    leaves graph sparsification unspecified; we use top-k anchor-affinity edges.
    For tractability, non-direct candidates in the same graph component receive
    a 3-hop proxy rather than exact all-pairs shortest paths. This limitation is
    reported explicitly by this implementation.
    """
    n = posterior_left.shape[0]
    if posterior_right.shape[0] != n or n == 0:
        raise ValueError("posterior sample dimensions must match and be non-empty")

    # The paper's inter-view candidate graph is constructed from anchor
    # affinities G^m (G^n)^T. Use its sparse top-k support directly. Do not
    # add posterior-overlap all-pairs search: that would add a different,
    # expensive candidate mechanism not specified by the paper.
    _, graph_rows, graph_cols, graph_keys, component_ids = _build_global_graph(
        affinity_left, affinity_right, neighbors=graph_neighbors, chunk_size=chunk_size
    )
    # Add a few derangement permutations only to keep sparse uniform-marginal
    # OT feasible. They receive the same semantic/topology costs as other edges
    # and are not labelled as known matches. No same-index edge is forced.
    fallback_rows, fallback_cols = _fallback_derangement_edges(n)
    candidate_keys = np.unique(np.concatenate([
        graph_rows.astype(np.int64) * n + graph_cols,
        fallback_rows * n + fallback_cols,
    ]))
    rows_np, cols_np = candidate_keys // n, candidate_keys % n
    positions = np.searchsorted(graph_keys, candidate_keys)
    direct = positions < graph_keys.size
    safe_positions = np.minimum(positions, max(graph_keys.size - 1, 0))
    if graph_keys.size:
        direct &= graph_keys[safe_positions] == candidate_keys
    else:
        direct[:] = False
    same_component = component_ids[rows_np] == component_ids[n + cols_np]
    hop_proxy = np.full(candidate_keys.shape, np.inf, dtype=np.float32)
    hop_proxy[direct] = 1.0
    hop_proxy[(~direct) & same_component] = 3.0
    topology = np.ones_like(hop_proxy, dtype=np.float32)
    finite = np.isfinite(hop_proxy)
    topology[finite] = 1.0 / (1.0 + np.exp(-hop_proxy[finite]))

    device = posterior_left.device
    rows = torch.as_tensor(rows_np, dtype=torch.long, device=device)
    cols = torch.as_tensor(cols_np, dtype=torch.long, device=device)
    topology_t = torch.as_tensor(topology, dtype=posterior_left.dtype, device=device)
    overlap_parts = []
    for start in range(0, rows.numel(), 131072):
        end = min(start + 131072, rows.numel())
        overlap_parts.append(
            (posterior_left[rows[start:end]] * posterior_right[cols[start:end]]).sum(dim=1)
        )
    overlap = torch.cat(overlap_parts).clamp(0.0, 1.0)
    edge_cost = (1.0 - overlap) * topology_t
    marginal = torch.full((n,), 1.0 / n, dtype=edge_cost.dtype, device=device)
    plan = sparse_sinkhorn(
        edge_cost,
        rows,
        cols,
        n,
        n,
        source_mass=marginal,
        target_mass=marginal,
        epsilon=matching_epsilon,
        iterations=sinkhorn_iterations,
    )

    row_mass = torch.zeros(n, dtype=plan.dtype, device=device).scatter_add_(0, rows, plan)
    col_mass = torch.zeros(n, dtype=plan.dtype, device=device).scatter_add_(0, cols, plan)
    row_weights = plan / row_mass[rows].clamp_min(1e-12)
    col_weights = plan / col_mass[cols].clamp_min(1e-12)
    row_error = (row_mass - marginal).abs().max().item()
    col_error = (col_mass - marginal).abs().max().item()
    return rows, cols, row_weights, col_weights, {
        "matching_edges": int(rows.numel()),
        "matching_row_marginal_error": row_error,
        "matching_col_marginal_error": col_error,
        "matching_direct_edge_fraction": float(direct.mean()) if direct.size else 0.0,
        "matching_same_component_fraction": float(same_component.mean()) if same_component.size else 0.0,
    }



_global_sparse_matching = global_sparse_matching

def evaluate_paper_model(
    model,
    dataset,
    batch_size,
    device,
    epsilon=0.1,
    graph_neighbors=10,
    matching_epsilon=0.1,
    sinkhorn_iterations=200,
    workers=0,
):
    """Evaluate using full-dataset transport and globally supported sparse matching."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
    common_batches = [[] for _ in range(model.num_views)]
    labels1, labels2, aligned_masks = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            v1, v2, y1, y2, aligned_mask, _ = batch
            views = [v1.to(device=device, dtype=torch.float32), v2.to(device=device, dtype=torch.float32)]
            _, _, common = model.encode_views(views)
            for view_index, h in enumerate(common):
                common_batches[view_index].append(h.detach())
            labels1.append(y1.numpy())
            labels2.append(y2.numpy())
            aligned_masks.append(aligned_mask.numpy())

        common = [torch.cat(parts, dim=0) for parts in common_batches]
        costs = model.transport_costs(common)
        posteriors, affinities, _ = model.posteriors(common, costs)
        n_total = common[0].shape[0]
        source_mass = torch.full((n_total,), 1.0 / n_total, dtype=common[0].dtype, device=device)
        plans = [
            sinkhorn(cost, source_mass, model.masses, epsilon=epsilon, iterations=sinkhorn_iterations)
            for cost in costs
        ]
        mean_transport_cost = [(plan * cost).sum().item() for plan, cost in zip(plans, costs)]
        reference = int(np.argmin(mean_transport_cost))
        inverse = (torch.as_tensor(mean_transport_cost, dtype=common[0].dtype, device=device) + 1e-8).reciprocal()
        weights = inverse / inverse.sum().clamp_min(1e-12)

        aligned = torch.as_tensor(np.concatenate(aligned_masks), dtype=torch.bool, device=device)
        y_views = [np.concatenate(labels1), np.concatenate(labels2)]
        fused = torch.zeros_like(common[reference])
        aligned_indices = torch.nonzero(aligned, as_tuple=False).flatten()
        if aligned_indices.numel():
            for view_index in range(model.num_views):
                fused[aligned_indices] += weights[view_index] * common[view_index][aligned_indices]

        unaligned_indices = torch.nonzero(~aligned, as_tuple=False).flatten()
        matching_info = {
            "matching_edges": 0,
            "matching_row_marginal_error": 0.0,
            "matching_col_marginal_error": 0.0,
            "matching_direct_edge_fraction": 0.0,
            "matching_same_component_fraction": 0.0,
        }
        if unaligned_indices.numel():
            q0 = posteriors[0].index_select(0, unaligned_indices)
            q1 = posteriors[1].index_select(0, unaligned_indices)
            g0 = affinities[0].index_select(0, unaligned_indices)
            g1 = affinities[1].index_select(0, unaligned_indices)
            rows, cols, row_weights, col_weights, matching_info = global_sparse_matching(
                q0,
                q1,
                g0,
                g1,
                graph_neighbors=graph_neighbors,
                matching_epsilon=matching_epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
            )
            if reference == 0:
                realigned_other = torch.zeros_like(common[1][unaligned_indices])
                realigned_other.index_add_(
                    0, rows, row_weights.unsqueeze(1) * common[1][unaligned_indices[cols]]
                )
                fused[unaligned_indices] = (
                    weights[0] * common[0][unaligned_indices] + weights[1] * realigned_other
                )
            else:
                realigned_other = torch.zeros_like(common[0][unaligned_indices])
                realigned_other.index_add_(
                    0, cols, col_weights.unsqueeze(1) * common[0][unaligned_indices[rows]]
                )
                fused[unaligned_indices] = (
                    weights[0] * realigned_other + weights[1] * common[1][unaligned_indices]
                )

        predictions = torch.cdist(fused, model.centers).argmin(dim=1).cpu().numpy()

    labels = y_views[reference]
    return {
        "acc": clustering_accuracy(labels, predictions),
        "nmi": normalized_mutual_info_score(labels, predictions),
        "ari": adjusted_rand_score(labels, predictions),
        "reference_view": reference,
        "mean_transport_cost": mean_transport_cost,
        "samples": len(labels),
        **matching_info,
    }
