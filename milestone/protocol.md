# Serial FFN development protocol — 2026-11-19

Previous development results were available for original NSC and a separate cross-layer balance pilot. The rules below were written before computing the serial candidates' development rankings. This is an exploratory milestone evaluation.

Use the linear reference P = B A, A of size f by d and B of size d by f, with independent iid Gaussian entries of SD 0.02 in each projection, as in the starter's reference initialization. Replace the activation by the identity. This isolates serial composition at initialization. The FFN branch excludes residual addition and normalization in both candidate scores. Attention, head grouping, and summation across layers follow the starter.

For G = PᵀP let M1 = E tr G = d² f σ⁴ and M2/M1 = σ⁴[(d+1)(f+d+1)+fd+2]. To obtain M2, condition on A: the d rows of P have covariance σ² AᵀA. The diagonal and cross-row terms in the squared Gram trace give d(d+1) tr(C²) + d(tr C)² for C = σ² AᵀA. Taking Gaussian moments of A yields the expression above.

Evaluate exactly two deterministic specification-only candidates:
- Rank bound: r log(1 + M1/r), r = min(d,f). Concavity of log and the rank bound give an upper bound on E log det(I+G).
- Moment bound: M1 log(1+q)/q, q = M2/M1. Weight the ensemble eigenvalues by their magnitude. The weighted mean is q and log(1+x)/x = integral from 0 to 1 of 1/(1+tx) dt is convex. Jensen gives a lower bound on E log det(I+G).

Replace only the two FFN terms with the chosen bound. Restoring both original FFN terms is the ablation. The candidates measure the linear branch under the stated initialization. Neither bound asserts capacity of the nonlinear Transformer or equality of scale with additive NSC. No fitted rescaling.

Numerical checks use small rectangular and square matrices, multiple signal scales, and seeded Monte Carlo with SVD as an independent reference. Compare both bounds, trace moments, finite values, and exact restoration of NSC. Benchmark checks use the supplied evaluator on development only. Compare both candidates with original NSC, position example, parameters, FLOPs, ablation, and random selection. Count changed pair orders and designs involved, including same-width near-parameter pairs and same-width/depth/total-FFN-width pairs. Report exact starter tie policy and four budgets with k=1 and 3. Also measure agreement with a first-moment-only FFN control of the same scale.

For any later final evaluation, prefer the candidate with lower mean selection regret over the eight development settings, then higher size-controlled tau-b, then the moment bound on exact ties. Any choice to advance a candidate requires reporting failure cases and how much its choices differ from size controls. Keep both candidates' development outputs regardless of results. Final labels remain held out at the milestone.
