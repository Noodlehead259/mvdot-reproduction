import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.paper_alignment import PaperAlignedRateDataset
from evaluation.paper_evaluation import evaluate_paper_model
from models.paper_mvdot import PaperMvDOT
from training.paper_training import build_global_matching_state, build_joint_optimizer, initialize_barycenter, train_joint_epoch, warmup_autoencoders


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("datasets/noisymnist.pt"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/paper_mvdot_noisymnist"))
    parser.add_argument("--aligned-rate", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, nargs=3, default=(500, 500, 2000))
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--graph-weight", type=float, default=0.05)
    parser.add_argument("--graph-neighbors", type=int, default=10)
    parser.add_argument("--matching-epsilon", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.98)
    parser.add_argument("--sinkhorn-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-init-samples", type=int, default=0)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    if args.warmup_epochs < 0 or args.epochs < 1:
        raise ValueError("warmup-epochs must be nonnegative and epochs must be positive")
    if args.batch_size < 2:
        raise ValueError("batch-size must be at least 2")
    if not 0.0 <= args.aligned_rate <= 1.0:
        raise ValueError("aligned-rate must be between 0 and 1")
    if args.experiment_dir.exists() and any(args.experiment_dir.iterdir()):
        raise FileExistsError(f"Experiment directory is not empty: {args.experiment_dir}")
    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    train_dataset = PaperAlignedRateDataset(args.data, aligned_rate=args.aligned_rate, seed=args.seed)
    eval_dataset = PaperAlignedRateDataset(args.data, aligned_rate=args.aligned_rate, seed=args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    eval_batch_size = args.batch_size

    model = PaperMvDOT(
        input_dims=[train_dataset.view1.shape[1], train_dataset.view2.shape[1]],
        num_clusters=int(torch.unique(train_dataset.labels).numel()),
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Samples: {len(train_dataset)}")
    print(f"Aligned rate: {args.aligned_rate:.2f}")
    print(f"Warm-up epochs: {args.warmup_epochs}")
    print(f"Joint-training epochs: {args.epochs}")
    print(f"Experiment directory: {args.experiment_dir.resolve()}")

    if args.warmup_epochs:
        warmup_optimizer = torch.optim.Adam(
            [parameter for autoencoder in model.autoencoders for parameter in autoencoder.parameters()],
            lr=args.learning_rate,
        )
        warmup_autoencoders(model, train_loader, warmup_optimizer, device, args.warmup_epochs, max_batches=args.max_train_batches)

    initialize_barycenter(model, train_dataset, args.batch_size, device, seed=args.seed, workers=args.workers, max_init_samples=args.max_init_samples)
    joint_optimizer = build_joint_optimizer(model, args.learning_rate)
    csv_path = args.experiment_dir / "metrics.csv"
    fields = [
        "epoch", "total_loss", "reconstruction_loss", "manifold_matching_loss",
        "transport_loss", "graph_loss", "sample_matching_loss", "mass_update_delta",
        "center_gradient_norm", "center_step_norm",
        "acc", "nmi", "ari", "reference_view", "mean_transport_cost",
        "matching_edges", "matching_row_marginal_error", "matching_col_marginal_error",
        "matching_direct_edge_fraction", "matching_same_component_fraction"
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            matching_state = build_global_matching_state(
                model,
                train_dataset,
                batch_size=args.batch_size,
                device=device,
                epsilon=args.epsilon,
                graph_neighbors=args.graph_neighbors,
                matching_epsilon=args.matching_epsilon,
                sinkhorn_iterations=args.sinkhorn_iterations,
                workers=args.workers,
            )
            print(
                f"Epoch {epoch}/{args.epochs} global matching refreshed | "
                f"edges={matching_state['matching_edges']} "
                f"marginal_err=({matching_state['matching_row_marginal_error']:.2e},"
                f"{matching_state['matching_col_marginal_error']:.2e})"
            )
            train_metrics = train_joint_epoch(
                model,
                train_loader,
                joint_optimizer,
                device,
                epsilon=args.epsilon,
                graph_weight=args.graph_weight,
                graph_neighbors=args.graph_neighbors,
                momentum=args.momentum,
                sinkhorn_iterations=args.sinkhorn_iterations,
                matching_epsilon=args.matching_epsilon,
                max_batches=args.max_train_batches,
                center_learning_rate=args.learning_rate,
                global_matching_state=matching_state,
            )
            row = {"epoch": epoch, **train_metrics}
            if not args.skip_evaluation:
                evaluation = evaluate_paper_model(
                    model,
                    eval_dataset,
                    batch_size=eval_batch_size,
                    device=device,
                    epsilon=args.epsilon,
                    graph_neighbors=args.graph_neighbors,
                    matching_epsilon=args.matching_epsilon,
                    sinkhorn_iterations=args.sinkhorn_iterations,
                    workers=args.workers,
                )
                row.update({
                    "acc": evaluation["acc"],
                    "nmi": evaluation["nmi"],
                    "ari": evaluation["ari"],
                    "reference_view": evaluation["reference_view"],
                    "mean_transport_cost": ";".join(f"{x:.8f}" for x in evaluation["mean_transport_cost"]),
                    "matching_edges": evaluation["matching_edges"],
                    "matching_row_marginal_error": evaluation["matching_row_marginal_error"],
                    "matching_col_marginal_error": evaluation["matching_col_marginal_error"],
                    "matching_direct_edge_fraction": evaluation["matching_direct_edge_fraction"],
                    "matching_same_component_fraction": evaluation["matching_same_component_fraction"],
                })
                print(
                    f"Epoch {epoch}/{args.epochs} | loss={row['total_loss']:.4f} "
                    f"Lrec={row['reconstruction_loss']:.4f} Lmm={row['manifold_matching_loss']:.4f} "
                    f"Lsm={row['sample_matching_loss']:.4f} "
                    f"center_grad={row['center_gradient_norm']:.3e} | "
                    f"ACC={row['acc']:.4f} NMI={row['nmi']:.4f} ARI={row['ari']:.4f} "
                    f"global_match_edges={row['matching_edges']} "
                    f"marginal_err=({row['matching_row_marginal_error']:.2e},"
                    f"{row['matching_col_marginal_error']:.2e})"
                )
            else:
                print(
                    f"Epoch {epoch}/{args.epochs} | loss={row['total_loss']:.4f} "
                    f"Lrec={row['reconstruction_loss']:.4f} Lmm={row['manifold_matching_loss']:.4f} "
                    f"Lsm={row['sample_matching_loss']:.4f} "
                    f"center_grad={row['center_gradient_norm']:.3e}"
                )
            writer.writerow({key: row.get(key, "") for key in fields})
            file.flush()
            checkpoint_path = checkpoint_dir / f"paper_mvdot_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": joint_optimizer.state_dict(),
                "config": vars(args) | {"data": str(args.data), "experiment_dir": str(args.experiment_dir)},
                "seed": args.seed,
                "train_metrics": train_metrics,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    print(f"Metrics saved: {csv_path}")


if __name__ == "__main__":
    main()
