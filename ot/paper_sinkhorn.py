import torch


def sinkhorn(cost, source_mass=None, target_mass=None, epsilon=0.1, iterations=80, tolerance=None):
    if cost.ndim != 2:
        raise ValueError("cost must be a two-dimensional tensor")
    if not torch.is_floating_point(cost):
        raise TypeError("cost must be floating point")
    n, k = cost.shape
    if n < 1 or k < 1:
        raise ValueError("cost must have non-empty dimensions")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    dtype = cost.dtype
    device = cost.device
    if source_mass is None:
        source_mass = torch.full((n,), 1.0 / n, dtype=dtype, device=device)
    else:
        source_mass = source_mass.to(device=device, dtype=dtype)
    if target_mass is None:
        target_mass = torch.full((k,), 1.0 / k, dtype=dtype, device=device)
    else:
        target_mass = target_mass.to(device=device, dtype=dtype)

    if source_mass.shape != (n,) or target_mass.shape != (k,):
        raise ValueError("marginal shapes do not match cost matrix")
    if torch.any(source_mass <= 0) or torch.any(target_mass <= 0):
        raise ValueError("all transport masses must be positive")
    if not torch.isclose(source_mass.sum(), target_mass.sum(), atol=1e-6, rtol=1e-6):
        raise ValueError("source and target marginals must have equal total mass")

    log_a = source_mass.log()
    log_b = target_mass.log()
    log_kernel = -cost / epsilon
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(iterations):
        previous_u = log_u
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
        if tolerance is not None and torch.max(torch.abs(log_u - previous_u)).detach().item() < tolerance:
            break

    plan = torch.exp(log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0))
    return plan


def _segment_logsumexp(values, segment_ids, num_segments):
    if values.ndim != 1 or segment_ids.ndim != 1 or values.shape != segment_ids.shape:
        raise ValueError("values and segment_ids must be one-dimensional and have equal shape")
    maxima = torch.full(
        (num_segments,), -torch.inf, dtype=values.dtype, device=values.device
    )
    maxima.scatter_reduce_(0, segment_ids, values, reduce="amax", include_self=True)
    shifted = values - maxima.index_select(0, segment_ids)
    sums = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, segment_ids, shifted.exp())
    return maxima + sums.clamp_min(torch.finfo(values.dtype).tiny).log()


def sparse_sinkhorn(
    edge_cost,
    row_indices,
    col_indices,
    num_rows,
    num_cols,
    source_mass=None,
    target_mass=None,
    epsilon=0.1,
    iterations=80,
):
    """Entropy-regularized OT on a sparse edge support.

    Edges outside the supplied support have zero transport mass. This is useful
    for global matching when a dense N-by-N coupling would be too large.
    """
    if edge_cost.ndim != 1 or row_indices.ndim != 1 or col_indices.ndim != 1:
        raise ValueError("edge_cost, row_indices and col_indices must be 1D")
    if not (edge_cost.numel() == row_indices.numel() == col_indices.numel()):
        raise ValueError("edge arrays must have equal lengths")
    if edge_cost.numel() == 0:
        raise ValueError("at least one supported edge is required")
    if num_rows < 1 or num_cols < 1 or epsilon <= 0 or iterations < 1:
        raise ValueError("invalid matrix size, epsilon, or iteration count")
    if row_indices.dtype != torch.long or col_indices.dtype != torch.long:
        raise TypeError("row_indices and col_indices must be torch.long")
    if row_indices.min() < 0 or row_indices.max() >= num_rows:
        raise ValueError("row index out of bounds")
    if col_indices.min() < 0 or col_indices.max() >= num_cols:
        raise ValueError("column index out of bounds")

    dtype, device = edge_cost.dtype, edge_cost.device
    row_indices = row_indices.to(device=device)
    col_indices = col_indices.to(device=device)
    if source_mass is None:
        source_mass = torch.full((num_rows,), 1.0 / num_rows, dtype=dtype, device=device)
    else:
        source_mass = source_mass.to(device=device, dtype=dtype)
    if target_mass is None:
        target_mass = torch.full((num_cols,), 1.0 / num_cols, dtype=dtype, device=device)
    else:
        target_mass = target_mass.to(device=device, dtype=dtype)
    if source_mass.shape != (num_rows,) or target_mass.shape != (num_cols,):
        raise ValueError("marginal shapes do not match sparse support")
    if torch.any(source_mass <= 0) or torch.any(target_mass <= 0):
        raise ValueError("all transport masses must be positive")
    if not torch.isclose(source_mass.sum(), target_mass.sum(), atol=1e-6, rtol=1e-6):
        raise ValueError("source and target marginals must have equal total mass")

    row_degree = torch.bincount(row_indices, minlength=num_rows)
    col_degree = torch.bincount(col_indices, minlength=num_cols)
    if torch.any(row_degree == 0) or torch.any(col_degree == 0):
        raise ValueError("every row and column must have at least one supported edge")

    log_kernel = -edge_cost / epsilon
    log_a = source_mass.log()
    log_b = target_mass.log()
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(iterations):
        log_u = log_a - _segment_logsumexp(
            log_kernel + log_v.index_select(0, col_indices), row_indices, num_rows
        )
        log_v = log_b - _segment_logsumexp(
            log_kernel + log_u.index_select(0, row_indices), col_indices, num_cols
        )

    return torch.exp(
        log_kernel
        + log_u.index_select(0, row_indices)
        + log_v.index_select(0, col_indices)
    )
