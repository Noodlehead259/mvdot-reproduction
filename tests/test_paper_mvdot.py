import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from datasets.paper_alignment import PaperAlignedRateDataset
from evaluation.paper_evaluation import evaluate_paper_model, _global_sparse_matching
from losses.paper_objectives import graph_laplacian_loss, total_loss
from models.paper_mvdot import PaperMvDOT
from ot.paper_sinkhorn import sinkhorn, sparse_sinkhorn
from training.paper_training import build_global_matching_state, build_joint_optimizer, train_joint_epoch, transport_center_gradient


class PaperMvDOTTests(unittest.TestCase):
    def test_sinkhorn_respects_marginals(self):
        torch.manual_seed(1)
        cost = torch.rand(7, 3, dtype=torch.float64)
        source = torch.full((7,), 1.0 / 7, dtype=torch.float64)
        target = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        plan = sinkhorn(cost, source, target, epsilon=0.2, iterations=200, tolerance=None)
        self.assertTrue(torch.allclose(plan.sum(dim=1), source, atol=1e-5))
        self.assertTrue(torch.allclose(plan.sum(dim=0), target, atol=1e-5))

    def test_sparse_sinkhorn_respects_marginals(self):
        torch.manual_seed(5)
        n = 8
        rows = []
        cols = []
        for i in range(n):
            rows.extend([i, (i + 1) % n, (i + 2) % n])
            cols.extend([i, (i + 1) % n, (i + 2) % n])
        rows = torch.tensor(rows, dtype=torch.long)
        cols = torch.tensor(cols, dtype=torch.long)
        costs = torch.rand(rows.numel())
        mass = torch.full((n,), 1.0 / n)
        plan = sparse_sinkhorn(costs, rows, cols, n, n, mass, mass, epsilon=0.2, iterations=200)
        row_mass = torch.zeros(n).scatter_add_(0, rows, plan)
        col_mass = torch.zeros(n).scatter_add_(0, cols, plan)
        self.assertTrue(torch.allclose(row_mass, mass, atol=1e-4))
        self.assertTrue(torch.allclose(col_mass, mass, atol=1e-4))

    def test_alignment_permutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.pt"
            view1 = torch.rand(20, 16)
            view2 = torch.rand(20, 16)
            labels = torch.arange(20) % 4
            torch.save({"view1": view1, "view2": view2, "labels": labels}, path)
            dataset = PaperAlignedRateDataset(path, aligned_rate=0.5, seed=7)
            self.assertEqual(dataset.aligned_count, 10)
            self.assertEqual(sorted(dataset.view2_indices.tolist()), list(range(20)))
            self.assertTrue(torch.equal(dataset.view2_indices[:10], torch.arange(10)))
            self.assertTrue(torch.all(dataset.view2_indices[10:] != torch.arange(10, 20)))

    def test_laplacian_loss_has_gradient(self):
        z = torch.randn(12, 5, requires_grad=True)
        loss = graph_laplacian_loss(z, neighbors=3)
        self.assertGreaterEqual(loss.detach().item(), -1e-6)
        loss.backward()
        self.assertIsNotNone(z.grad)
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_joint_objectives_backpropagate(self):
        torch.manual_seed(2)
        model = PaperMvDOT([16, 16], num_clusters=4, latent_dim=5, hidden_dims=(12, 10, 8))
        views = [torch.rand(10, 16), torch.rand(10, 16)]
        aligned_mask = torch.tensor([True] * 4 + [False] * 6)
        latents = [autoencoder.encode(view) for autoencoder, view in zip(model.autoencoders, views)]
        model.initialize_spherical_kmeans(latents, seed=9, iterations=5)
        output = model(views)
        source = torch.full((10,), 0.1)
        plans = [sinkhorn(cost, source, model.masses, epsilon=0.2, iterations=30) for cost in output["costs"]]
        loss, metrics, matching, matching_costs = total_loss(
            views,
            output,
            plans,
            graph_weight=0.05,
            graph_neighbors=3,
            aligned_mask=aligned_mask,
            matching_epsilon=0.2,
            sinkhorn_iterations=30,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(matching), 1)
        self.assertEqual(len(matching_costs), 1)
        loss.backward()
        self.assertIsNotNone(model.centers.grad)
        self.assertTrue(torch.isfinite(model.centers.grad).all())
        self.assertTrue(torch.allclose(model.centers.norm(dim=1), torch.ones(4), atol=1e-4))

    def test_sample_matching_loss_uses_only_unaligned_samples(self):
        torch.manual_seed(3)
        model = PaperMvDOT([16, 16], num_clusters=4, latent_dim=5, hidden_dims=(12, 10, 8))
        views = [torch.rand(8, 16), torch.rand(8, 16)]
        outputs = model(views)
        source = torch.full((8,), 1.0 / 8)
        plans = [sinkhorn(cost, source, model.masses, epsilon=0.2, iterations=25) for cost in outputs["costs"]]
        aligned = torch.ones(8, dtype=torch.bool)
        _, metrics, matrices, pair_costs = total_loss(
            views, outputs, plans, graph_weight=0.05, graph_neighbors=3,
            aligned_mask=aligned, matching_epsilon=0.2, sinkhorn_iterations=25,
        )
        self.assertEqual(metrics["sample_matching_loss"].detach().item(), 0.0)
        self.assertEqual(matrices, [])
        self.assertEqual(pair_costs, [])

    def test_global_matching_does_not_force_same_index_edges(self):
        n = 6
        left = torch.eye(n, dtype=torch.float32)
        permutation = torch.tensor([1, 2, 3, 4, 5, 0])
        right = left[permutation]
        rows, cols, row_weights, _, info = _global_sparse_matching(
            left, right, left, right, graph_neighbors=1,
            matching_epsilon=0.2, sinkhorn_iterations=500,
        )
        expected = torch.argsort(permutation)
        expected_weight = row_weights[(cols.cpu() == expected[rows.cpu()])].sum()
        diagonal_weight = row_weights[rows == cols].sum()
        self.assertGreater(expected_weight.item(), diagonal_weight.item())
        self.assertGreater(info["matching_edges"], 0)
        self.assertLess(info["matching_row_marginal_error"], 1e-4)
        self.assertLess(info["matching_col_marginal_error"], 1e-4)

    def test_global_evaluation_uses_sparse_matching_across_unaligned_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            torch.manual_seed(11)
            torch.save({
                "view1": torch.rand(32, 12),
                "view2": torch.rand(32, 12),
                "labels": torch.arange(32) % 4,
            }, path)
            dataset = PaperAlignedRateDataset(path, aligned_rate=0.25, seed=13)
            model = PaperMvDOT([12, 12], num_clusters=4, latent_dim=5, hidden_dims=(10, 8, 6))
            result = evaluate_paper_model(
                model, dataset, batch_size=7, device=torch.device("cpu"),
                graph_neighbors=3,
                matching_epsilon=0.2, sinkhorn_iterations=200,
            )
            self.assertEqual(result["samples"], 32)
            self.assertGreater(result["matching_edges"], 0)
            self.assertLess(result["matching_row_marginal_error"], 1e-4)
            self.assertLess(result["matching_col_marginal_error"], 1e-4)
            self.assertTrue(np.isfinite(result["acc"]))

    def test_joint_training_uses_full_unaligned_pool_matching_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.pt"
            torch.manual_seed(29)
            torch.save({
                "view1": torch.rand(24, 12),
                "view2": torch.rand(24, 12),
                "labels": torch.arange(24) % 3,
            }, path)
            dataset = PaperAlignedRateDataset(path, aligned_rate=0.25, seed=31)
            model = PaperMvDOT([12, 12], num_clusters=3, latent_dim=5, hidden_dims=(10, 8, 6))
            state = build_global_matching_state(
                model, dataset, batch_size=8, device=torch.device("cpu"),
                epsilon=0.2, graph_neighbors=3, matching_epsilon=0.2,
                sinkhorn_iterations=250,
            )
            self.assertEqual(state["global_posteriors"][0].shape[0], 18)
            self.assertGreater(state["matching_edges"], 0)
            self.assertLess(state["matching_row_marginal_error"], 1e-4)
            self.assertLess(state["matching_col_marginal_error"], 1e-4)
            loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)
            optimizer = build_joint_optimizer(model, learning_rate=1e-3)
            metrics = train_joint_epoch(
                model, loader, optimizer, torch.device("cpu"), epsilon=0.2,
                graph_weight=0.05, graph_neighbors=3, sinkhorn_iterations=40,
                global_matching_state=state, max_batches=2,
            )
            self.assertTrue(np.isfinite(metrics["sample_matching_loss"]))
            self.assertTrue(np.isfinite(metrics["total_loss"]))
            self.assertGreater(metrics["sample_matching_loss"], 0.0)
            self.assertTrue(torch.allclose(model.centers.norm(dim=1), torch.ones(3), atol=1e-4))

    def test_joint_optimizer_excludes_barycenter_centers(self):
        model = PaperMvDOT([8, 8], num_clusters=3, latent_dim=4, hidden_dims=(6, 6, 8))
        optimizer = build_joint_optimizer(model, learning_rate=1e-3)
        optimizer_parameter_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertNotIn(id(model.centers), optimizer_parameter_ids)
        self.assertIn(id(model.mapping.weight), optimizer_parameter_ids)
        self.assertTrue(all(id(parameter) not in optimizer_parameter_ids for parameter in [model.centers]))

    def test_center_gradient_matches_fixed_plan_transport_gradient(self):
        centers = torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        h1 = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        h2 = torch.tensor([[0.5, 0.5], [0.9, 0.1]])
        plan1 = torch.tensor([[0.4, 0.1], [0.1, 0.4]])
        plan2 = torch.tensor([[0.2, 0.3], [0.3, 0.2]])
        costs = [
            (h.unsqueeze(1) - centers.unsqueeze(0)).square().sum(dim=2)
            for h in (h1, h2)
        ]
        plans = [plan1, plan2]
        gradient = transport_center_gradient(centers, costs, plans)
        expected = torch.zeros_like(centers)
        for h, plan in zip((h1, h2), plans):
            for i in range(h.shape[0]):
                for k in range(centers.shape[0]):
                    expected[k] += 2.0 * plan[i, k] * (centers.detach()[k] - h[i])
        self.assertTrue(torch.allclose(gradient, expected, atol=1e-6))

    def test_mass_update_is_fixed_under_exact_marginals(self):
        model = PaperMvDOT([8, 8], num_clusters=3, latent_dim=4, hidden_dims=(6, 6, 8))
        source = torch.full((5,), 0.2)
        costs = [torch.rand(5, 3), torch.rand(5, 3)]
        plans = [sinkhorn(cost, source, model.masses, epsilon=0.2, iterations=200, tolerance=None) for cost in costs]
        before = model.masses.clone()
        empirical = model.update_masses_from_plans_(plans, momentum=0.95)
        self.assertTrue(torch.allclose(empirical, before, atol=1e-5))
        self.assertTrue(torch.allclose(model.masses, before, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
