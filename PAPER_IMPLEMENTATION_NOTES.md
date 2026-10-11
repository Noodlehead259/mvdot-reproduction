# MvDOT paper-grounded implementation notes — final bundle

This package is an implementation of the method described in the supplied 18-page manuscript, with explicit choices where the manuscript is under-specified and explicit corrections where printed equations are internally inconsistent. It is **not** the authors' unpublished source code, and exact numerical reproduction cannot be guaranteed from the paper alone.

## Paper components implemented

- Independent four-linear-layer encoder/decoder pairs per view; reconstruction-only warm-up using Eq. (27).
- Spherical k-means initialization from concatenated view latents, with normalized cluster centers and uniform initial barycenter masses (Eq. (28)).
- One shared fully connected mapping into the common space.
- Squared-Euclidean sample-to-center costs and entropy-regularized Sinkhorn transport to a shared barycenter with per-view source and common target marginals (Eqs. (29)–(30)).
- Latent-space graph Laplacian regularization in the manifold loss (Eq. (30)).
- A separate center update derived from the explicit fixed-plan transport-cost derivative, followed by unit-norm projection (Eqs. (31)–(32)).
- Barycenter mass update as printed in Eq. (33); it is a fixed point if the exact target marginals in Eq. (29) are imposed.
- Gaussian anchor affinities, a sparse bipartite cross-view graph, graph-hop topological penalties, sparse entropy-regularized sample matching, and negative log cluster-overlap semantic guidance (Eqs. (34)–(37), with the sign correction noted below).
- Full-un­aligned-pool matching state refreshed at the start of every joint-training epoch. The global sparse matching matrix is held fixed within that epoch; the live side's posterior receives gradients in minibatch updates, while the opposite-view posterior cache is detached. This is an explicit memory-feasible alternating approximation, not fully differentiable global end-to-end Sinkhorn.
- Full-dataset transport and reference-view selection for inference, sparse global matching of unaligned samples, soft realignment, reliability-weighted view fusion, and nearest-center clustering (Algorithm 1).

## Deliberate choices and limitations

1. **Graph discretization is not specified in enough detail.** The manuscript describes BFS on a discretized cross-view affinity graph but does not provide its edge-selection rule. This implementation uses the union of top-k anchor-affinity edges in both directions.
2. **The exact semantic discrepancy matrix is not explicitly defined.** The implementation normalizes Gaussian anchor affinities into posteriors, defines agreement by their dot product, and sets semantic discrepancy to one minus this overlap.
3. **Exact graph hop distances are infeasible over all pairs at this scale.** Candidate graph edges have their direct one-hop distance. For a non-direct support edge, this implementation uses a three-hop proxy when endpoints share a connected component, and the maximum penalty otherwise. This is documented approximation, not an exact BFS distance for every candidate pair.
4. **Global matching in training is alternating.** The full-pool sparse `P` and cached posteriors are refreshed once per epoch, then held fixed as the encoder/mapping weights are updated in batches. The live view posterior is differentiable, but the full-pool Sinkhorn solve is not differentiated through. This avoids dense full-dataset autograd memory use. Do not describe this as fully end-to-end differentiation through global `P`.
5. **The mass-update equations conflict.** Eq. (29) fixes each plan's column sums to `pi`. Aggregating these constrained column masses and normalizing returns `pi`, so Eq. (33) cannot adapt the masses in exact arithmetic. The code does not silently substitute unbalanced OT or invent a different estimator; it implements the stated update and records its fixed-point behavior.
6. **Equation (35)'s printed sign conflicts with the prose and total minimization.** Minimizing positive `log(overlap)` would encourage lower overlap. The code uses negative log overlap, so minimizing the loss maximizes cluster-level agreement, consistent with the described intent.
7. **Equation (31)'s shorthand is inconsistent.** The right-hand shorthand as printed does not follow from its defined `s_k`. The code computes the gradient from the explicit transport-cost expression: `2 * sum_i T_ik * (c_k - h_i)` (summed over views), then projects centers to unit norm.
8. **Mini-batch reductions and graph construction require choices.** Reconstruction is per-view mean squared error; the graph Laplacian is a weighted k-NN graph computed within the current training batch. The paper writes the reconstruction objective as squared Frobenius norms but does not fully spell out stochastic reduction details, and the graph's exact construction is not fully specified.
9. **Initialization/bandwidth details are implementation choices.** The shared linear map uses PyTorch's default linear initialization. The Gaussian anchor-affinity bandwidth is estimated from the current full-pool mean transport costs per center and held fixed within each training epoch.
10. **Dataset is an explicit approximation.** `datasets/noisymnist.pt` is the project's constructed MNIST/noisy-view variant. This package supports a configurable aligned rate and permutes the unaligned portion. It does not claim the generated view exactly matches the paper authors' NoisyMNIST preprocessing.
11. **Reproducibility settings.** The CLI defaults are project defaults selected from or around the paper's reported value ranges, not a claim that one set of values is the authors' exact NoisyMNIST configuration. Keep seeds, aligned rate, CLI config, code revision, and dataset-generation parameters with every result.

## Checks performed for this bundle

- `python -m py_compile` on the main model, objectives, transport solver, trainer, evaluator and CLI.
- `python -m unittest discover -s tests -v`: 12 tests passed.
- One end-to-end CLI run on a 48-sample synthetic two-view dataset: warm-up, global matching refresh, two minibatch updates, evaluation, metrics CSV and checkpoint all completed. Matching marginal errors were below `1e-7` in that run.

The build environment did not have the user's actual Windows project or RTX 5050 available. The full 70,000-sample/80-epoch run was not executed here. Run the small integration test in your project before committing to a full training job.
