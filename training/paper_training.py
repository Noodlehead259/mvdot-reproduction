import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.paper_evaluation import global_sparse_matching
from losses.paper_objectives import total_loss
from ot.paper_sinkhorn import sinkhorn


def _batch_views(batch, device):
    view1, view2, _, _, aligned_mask, indices = batch
    views = [
        view1.to(device=device, dtype=torch.float32),
        view2.to(device=device, dtype=torch.float32),
    ]
    return views, aligned_mask.to(device=device, dtype=torch.bool), indices.to(device=device, dtype=torch.long)


def warmup_autoencoders(model, loader, optimizer, device, epoch_count, max_batches=0):
    history = []
    for epoch in range(epoch_count):
        model.train()
        total = 0.0
        samples = 0
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            views, _, _ = _batch_views(batch, device)
            optimizer.zero_grad(set_to_none=True)
            _, reconstructions, _ = model.encode_views(views)
            loss = sum(
                torch.nn.functional.mse_loss(reconstruction, view)
                for reconstruction, view in zip(reconstructions, views)
            )
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * views[0].shape[0]
            samples += views[0].shape[0]
        row = {"warmup_epoch": epoch + 1, "reconstruction_loss": total / max(samples, 1)}
        history.append(row)
        print(f"Warm-up {epoch + 1}/{epoch_count} | reconstruction_loss={row['reconstruction_loss']:.6f}")
    return history


def initialize_barycenter(model, dataset, batch_size, device, seed=42, workers=0, max_init_samples=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
    latent_views = [[] for _ in range(model.num_views)]
    was_training = model.training
    model.eval()
    with torch.no_grad():
        observed = 0
        for batch in loader:
            views, _, _ = _batch_views(batch, device)
            if max_init_samples and observed >= max_init_samples:
                break
            if max_init_samples and observed + views[0].shape[0] > max_init_samples:
                keep = max_init_samples - observed
                views = [view[:keep] for view in views]
            latents, _, _ = model.encode_views(views)
            observed += views[0].shape[0]
            for index, latent in enumerate(latents):
                latent_views[index].append(latent.detach())
    model.train(was_training)
    if not latent_views or not latent_views[0]:
        raise RuntimeError("No samples were available for barycenter initialization")
    latent_views = [torch.cat(parts, dim=0) for parts in latent_views]
    model.initialize_spherical_kmeans(latent_views, seed=seed)
    model.project_centers_()


def build_joint_optimizer(model, learning_rate):
    """Adam optimizes encoders, decoders and F_m; centers use Eq. (32)."""
    parameters = [parameter for name, parameter in model.named_parameters() if name != "centers"]
    return torch.optim.Adam(parameters, lr=learning_rate)


def transport_center_gradient(centers, costs, plans):
    """Differentiate transport cost w.r.t. centers while holding plans fixed.

    This uses the explicit left-hand derivative in Eq. (31):
    2 * sum_{v,i} T[v,i,k] * (c[k] - h[v,i]). The printed shorthand on the
    equation's right-hand side is inconsistent with the paper's definition
    of s_k, so the explicit cost derivative is used.
    """
    if len(costs) != len(plans):
        raise ValueError("costs and plans must have the same length")
    objective = sum((plan.detach() * cost).sum() for plan, cost in zip(plans, costs))
    gradient = torch.autograd.grad(objective, centers, retain_graph=True, create_graph=False)[0]
    return gradient.detach()


def build_global_matching_state(
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
    """Refresh full-pool unpaired matching once per epoch.

    The model is evaluated over all samples to create a global posterior/affinity
    cache. Sparse matching is then solved over the complete unaligned pool. The
    resulting P is held fixed within the epoch while minibatches update the
    networks; minibatch posteriors remain differentiable on the live side of
    each loss. This avoids pretending that each minibatch is an independent
    sample-matching problem, while keeping the computation memory-feasible.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, drop_last=False)
    common_parts = [[] for _ in range(model.num_views)]
    aligned_parts = []
    index_parts = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            views, aligned_mask, indices = _batch_views(batch, device)
            _, _, common = model.encode_views(views)
            for view_index, representation in enumerate(common):
                common_parts[view_index].append(representation.detach())
            aligned_parts.append(aligned_mask.detach())
            index_parts.append(indices.detach())
    model.train(was_training)

    common = [torch.cat(parts, dim=0) for parts in common_parts]
    aligned = torch.cat(aligned_parts, dim=0)
    sample_indices = torch.cat(index_parts, dim=0)
    if sample_indices.numel() != len(dataset):
        raise RuntimeError("Global cache did not process the full dataset")
    if not torch.equal(sample_indices, torch.arange(len(dataset), device=device)):
        order = torch.argsort(sample_indices)
        common = [part.index_select(0, order) for part in common]
        aligned = aligned.index_select(0, order)
        sample_indices = sample_indices.index_select(0, order)

    with torch.no_grad():
        costs = model.transport_costs(common)
        posteriors, affinities, sigma2 = model.posteriors(common, costs)
        n_total = common[0].shape[0]
        source_mass = torch.full((n_total,), 1.0 / n_total, dtype=common[0].dtype, device=device)
        plans = [
            sinkhorn(cost, source_mass, model.masses, epsilon=epsilon, iterations=sinkhorn_iterations)
            for cost in costs
        ]
        transport_distances = torch.stack([
            (plan * cost).sum().detach() for plan, cost in zip(plans, costs)
        ])
        inverse_distances = (transport_distances + 1e-8).reciprocal()
        view_weights = inverse_distances / inverse_distances.sum().clamp_min(1e-12)

        unaligned_indices = torch.nonzero(~aligned, as_tuple=False).flatten()
        pool_size = int(unaligned_indices.numel())
        offset_lookup = np.full(n_total, -1, dtype=np.int64)
        offset_lookup[unaligned_indices.detach().cpu().numpy()] = np.arange(pool_size, dtype=np.int64)

        if pool_size >= 2:
            q_left = posteriors[0].index_select(0, unaligned_indices).detach()
            q_right = posteriors[1].index_select(0, unaligned_indices).detach()
            g_left = affinities[0].index_select(0, unaligned_indices).detach()
            g_right = affinities[1].index_select(0, unaligned_indices).detach()
            rows, cols, row_weights, col_weights, matching_info = global_sparse_matching(
                q_left,
                q_right,
                g_left,
                g_right,
                graph_neighbors=graph_neighbors,
                matching_epsilon=matching_epsilon,
                sinkhorn_iterations=sinkhorn_iterations,
            )
            rows_cpu = rows.detach().cpu().numpy()
            cols_cpu = cols.detach().cpu().numpy()
            row_counts = np.bincount(rows_cpu, minlength=pool_size)
            col_counts = np.bincount(cols_cpu, minlength=pool_size)
            row_ptr = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int64)
            col_order = torch.argsort(cols, stable=True).detach().cpu().numpy().astype(np.int64, copy=False)
            col_ptr = np.concatenate(([0], np.cumsum(col_counts))).astype(np.int64)
        else:
            q_left = posteriors[0].new_empty((0, model.num_clusters))
            q_right = posteriors[1].new_empty((0, model.num_clusters))
            rows = torch.empty(0, dtype=torch.long, device=device)
            cols = torch.empty(0, dtype=torch.long, device=device)
            row_weights = torch.empty(0, dtype=common[0].dtype, device=device)
            col_weights = torch.empty(0, dtype=common[0].dtype, device=device)
            row_ptr = np.zeros(pool_size + 1, dtype=np.int64)
            col_order = np.empty(0, dtype=np.int64)
            col_ptr = np.zeros(pool_size + 1, dtype=np.int64)
            matching_info = {
                "matching_edges": 0,
                "matching_row_marginal_error": 0.0,
                "matching_col_marginal_error": 0.0,
                "matching_direct_edge_fraction": 0.0,
                "matching_same_component_fraction": 0.0,
            }

    return {
        "sigma2": sigma2.detach(),
        "global_posteriors": [q_left, q_right],
        "view_weights": view_weights.detach(),
        "unaligned_lookup": offset_lookup,
        "unaligned_indices": unaligned_indices.detach(),
        "rows": rows,
        "cols": cols,
        "row_weights": row_weights.detach(),
        "col_weights": col_weights.detach(),
        "row_ptr": row_ptr,
        "col_order": col_order,
        "col_ptr": col_ptr,
        "transport_distances": transport_distances.detach(),
        **matching_info,
    }


def train_joint_epoch(
    model,
    loader,
    optimizer,
    device,
    epsilon=0.1,
    graph_weight=0.05,
    graph_neighbors=10,
    momentum=0.98,
    sinkhorn_iterations=200,
    matching_epsilon=0.1,
    max_batches=0,
    center_learning_rate=None,
    global_matching_state=None,
):
    model.train()
    totals = {}
    batches = 0
    if center_learning_rate is None:
        center_learning_rate = optimizer.param_groups[0]["lr"]

    for batch_index, batch in enumerate(loader):
        if max_batches and batch_index >= max_batches:
            break
        views, aligned_mask, sample_indices = _batch_views(batch, device)
        if global_matching_state is None:
            outputs = model(views)
        else:
            outputs = model(views, posterior_sigma2=global_matching_state["sigma2"])
        source_mass = torch.full(
            (views[0].shape[0],), 1.0 / views[0].shape[0], dtype=views[0].dtype, device=device
        )
        plans = [
            sinkhorn(cost, source_mass, model.masses, epsilon=epsilon, iterations=sinkhorn_iterations)
            for cost in outputs["costs"]
        ]
        loss, metrics, _, _ = total_loss(
            views,
            outputs,
            plans,
            graph_weight=graph_weight,
            graph_neighbors=graph_neighbors,
            aligned_mask=aligned_mask,
            matching_epsilon=matching_epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            global_matching_state=global_matching_state,
            sample_indices=sample_indices,
        )
        center_gradient = transport_center_gradient(model.centers, outputs["costs"], plans)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.centers.grad = None
        optimizer.step()
        with torch.no_grad():
            model.centers.add_(center_gradient, alpha=-center_learning_rate)
            model.project_centers_()
        previous_masses = model.masses.clone()
        model.update_masses_from_plans_(plans, momentum=momentum)

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value.detach().item()
        totals["mass_update_delta"] = totals.get("mass_update_delta", 0.0) + torch.max(
            torch.abs(model.masses - previous_masses)
        ).detach().item()
        totals["center_gradient_norm"] = totals.get("center_gradient_norm", 0.0) + center_gradient.norm().item()
        totals["center_step_norm"] = totals.get("center_step_norm", 0.0) + (center_learning_rate * center_gradient.norm()).item()
        batches += 1

    return {key: value / max(batches, 1) for key, value in totals.items()}
