# Milestone experiments for serial FFN scoring

## Reproduce the results

Complete the [setup](../README.md#setup) and run these commands from the
repository root, with the same Python environment active throughout. The
last command reads the two evaluation outputs, so run the commands in order.

```sh
python3 starter/check_starter.py
PYTHONPATH=starter:milestone python3 milestone/check_serial.py
PYTHONPATH=milestone python3 starter/evaluate.py --split development --student milestone/serial_score.py --output milestone/results/moment.json
PYTHONPATH=milestone python3 starter/evaluate.py --split development --student milestone/rank_score.py --output milestone/results/rank.json
PYTHONPATH=starter:milestone python3 milestone/audit_development.py
```

The commands write four JSON files in [results/](results/), replacing the
saved copies if run in place. The starter check verifies data integrity.
The scoring runs use the development split's validation perplexities.
Scoring times in the JSON vary by run. On the tested Apple Silicon Mac,
the checks and evaluations took a few seconds after setup.

## Development results

See the [experiment protocol](protocol.md) for the assumptions, derivation,
and candidate selection rule.

| Score | Spearman rho | Kendall tau-b | Same-width near-parameter tau-b |
|---|---:|---:|---:|
| Original NSC and ablation | 0.969745 | 0.861414 | 0.622222 |
| Position example | 0.970069 | 0.865051 | 0.622222 |
| Moment bound and rank bound | 0.975626 | 0.887273 | 0.655556 |
| Reported parameters | 0.727873 | 0.575758 | 0.133333 |
| Reference FLOPs | 0.967948 | 0.858471 | 0.622222 |

The original metrics reproduce the starter's expected results and the previous baseline. The ablation exactly recovers original NSC on all 100 development architectures. Sixteen dimension/scale checks passed using 4,096 matrix draws per shape. The SVD tool reuses the same draws across scales for each shape.

Relative to original NSC, 208 of 4,950 pair orders changed, involving 78 architectures. Among the 180 same-width near-parameter pairs, seven orders changed across 13 architectures. Five changes agreed with the validation PPL order and two did not. Examples are in the audit JSON, including config_13_j5 versus config_2_j8 (corrected) and config_11_j1 versus config_19_j6 (worsened). The archive provided zero pairs matching hidden width, depth, and total FFN width simultaneously. Pair counts are descriptive because pairs share architectures.

The serial candidates reduce regret at 10M/top-1 (1.94079 to zero) and 20M/top-3 (0.56504 to zero) relative to original NSC. The other six settings are unchanged. Mean regret over the eight settings decreases from 0.96857 to 0.65535. At 30M/top-3, reference FLOPs reaches zero regret while the serial scores leave 0.08459. The full budget tables include uniform random selection and tie variability.

Both bounds and the first-moment-only control produce identical pair ordering across all 4,950 development pairs and identical budget outcomes. The data therefore give no evidence of an incremental benefit from the spectral terms in these bounds. The product's first moment scales as d² times FFN width at fixed initialization. Keeping raw scales also changes the relative influence of attention and FFN. Both effects limit mechanism attribution. The predefined tie-break prefers the moment bound if carried forward. Optional training and final evaluation remain outside this run.
