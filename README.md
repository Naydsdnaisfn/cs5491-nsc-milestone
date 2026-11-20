# Serial FFN scoring for Transformer architecture selection

CS5491 milestone code and development results. The two candidate scores
replace the original NSC feed-forward network (FFN) term with bounds on
the spectral score of the composed linear projections.

## Setup

1. Download the course-provided NSC starter from Canvas under **CS5491 →
   Files → project → neural-spectral-capacity**. Extract its contents into
   `starter/` at the repository root. Check that `starter/evaluate.py`,
   `starter/check_starter.py`, and `starter/data/provenance.json` are present
   directly under the paths shown.
2. From the repository root, set up Python 3.11 or later and install the
   dependency for the singular-value decomposition (SVD) check. The full
   workflow was tested on macOS with Python 3.12 and NumPy 2.3.5.

   ```sh
   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -r requirements.txt
   ```

3. Follow the [reproduction commands](milestone/README.md#reproduce-the-results).
   Scoring and archive evaluation use Python's standard library.

The course starter and its data are supplied separately through Canvas.
Git ignores the local `starter/` directory.

## Code and results

| Path | Purpose |
|---|---|
| [milestone/serial_score.py](milestone/serial_score.py) | Implements the projection moments, the two bounds, the first-moment control, and the original NSC ablation. The evaluator entry point uses the moment bound. |
| [milestone/rank_score.py](milestone/rank_score.py) | Uses the rank bound through the same evaluator interface. |
| [milestone/check_serial.py](milestone/check_serial.py) | Checks the analytic moments and bounds against seeded SVD calculations. |
| [milestone/audit_development.py](milestone/audit_development.py) | Verifies the baseline and ablation, counts pair-order changes, and evaluates the first-moment control. |
| [milestone/results/](milestone/results/) | Stores the numerical checks, both candidate evaluations, and the pair-order audit. |

See the [experiment protocol](milestone/protocol.md) for the derivation and
the [experiment record](milestone/README.md) for a summary of results.
