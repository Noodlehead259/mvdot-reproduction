import torch


def sinkhorn(cost, a=None, b=None, epsilon=0.05, max_iter=100, tol=1e-6):
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D tensor")

    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    n, m = cost.shape

    if n == 0 or m == 0:
        raise ValueError("cost must have non-empty dimensions")

    if a is None:
        a = torch.full((n,), 1.0 / n, device=cost.device, dtype=cost.dtype)

    if b is None:
        b = torch.full((m,), 1.0 / m, device=cost.device, dtype=cost.dtype)

    a = a.to(device=cost.device, dtype=cost.dtype)
    b = b.to(device=cost.device, dtype=cost.dtype)

    if a.shape != (n,) or b.shape != (m,):
        raise ValueError("marginals must match cost dimensions")

    if not torch.isfinite(cost).all():
        raise ValueError("cost must contain only finite values")

    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("marginals must contain only finite values")

    if torch.any(a <= 0) or torch.any(b <= 0):
        raise ValueError("marginals must be strictly positive")

    if not torch.isclose(a.sum(), b.sum(), atol=tol, rtol=tol):
        raise ValueError("marginals must have equal total mass")

    log_a = torch.log(a)
    log_b = torch.log(b)
    log_kernel = -cost / epsilon

    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)

    for _ in range(max_iter):
        previous_u = log_u

        log_u = log_a - torch.logsumexp(
            log_kernel + log_v.unsqueeze(0), dim=1
        )

        log_v = log_b - torch.logsumexp(
            log_kernel + log_u.unsqueeze(1), dim=0
        )

        if torch.max(torch.abs(log_u - previous_u)) < tol:
            break

    transport = torch.exp(
        log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)
    )

    return transport