# MvDOT paper-grounded implementation

This additive overlay retains the original project files and supplies a separate pipeline under `paper_*` filenames. It covers reconstruction warm-up, spherical k-means initialization, common-space mapping, semi-discrete OT, a graph-Laplacian manifold loss, an epoch-global sparse matching state for unaligned samples, the separate center update, and paper-style inference.

## Verify

```powershell
python -m unittest discover -s tests -v
python scripts\train_paper_mvdot.py --help
```

## Small integration test

```powershell
python scripts\train_paper_mvdot.py --warmup-epochs 1 --epochs 1 --batch-size 128 --max-train-batches 1 --max-init-samples 2048 --experiment-dir experiments\paper_final_smoke
```

The smoke run still builds the global matching state over the supplied dataset, but limits model updates to one minibatch. Use a new experiment directory; the CLI intentionally refuses to use a nonempty experiment directory.

## Full run

```powershell
python scripts\train_paper_mvdot.py --aligned-rate 0.0 --experiment-dir experiments\paper_final_rate0
```

The default run uses 10 reconstruction warm-up epochs followed by 80 joint-training epochs, evaluates each epoch, saves a checkpoint every epoch, and writes `metrics.csv`. It refreshes sparse global unaligned matching at each joint epoch and again for evaluation; a full dataset run will take substantially longer than the smoke test.

Read `PAPER_IMPLEMENTATION_NOTES.md` before interpreting results. The paper omits exact graph discretization and semantic discrepancy details, contains inconsistencies in its mass-update and matching-loss equations, and does not provide the authors' source implementation. The implementation documents the selected choices and does not claim bit-for-bit reproduction.
