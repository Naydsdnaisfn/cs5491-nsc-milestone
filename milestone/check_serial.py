import json
from pathlib import Path

import numpy as np
from nsc import original_nsc
from serial_score import ablation, product_moments, serial_capacity


def main():
    rng = np.random.default_rng(20261119)
    draws = 4096
    rows = []
    for d, f in [(4, 4), (4, 12), (12, 4), (8, 16)]:
        a = rng.standard_normal((draws, f, d))
        b = rng.standard_normal((draws, d, f))
        squared = np.linalg.svd(b @ a, compute_uv=False) ** 2
        for sigma in [0.0, 0.02, 0.2, 0.5]:
            eigenvalues = squared * sigma ** 4
            lower = serial_capacity(d, f, 'moment', sigma)
            upper = serial_capacity(d, f, 'rank', sigma)
            trace, ratio = product_moments(d, f, sigma)
            assert np.isfinite([lower, upper]).all() and 0 <= lower <= upper
            for samples, expected in [(eigenvalues.sum(axis=1), trace),
                                      ((eigenvalues ** 2).sum(axis=1), trace * ratio)]:
                se = samples.std(ddof=1) / np.sqrt(draws)
                assert abs(samples.mean() - expected) <= 5 * se
            samples = np.log1p(eigenvalues).sum(axis=1)
            mean = float(samples.mean())
            se = float(samples.std(ddof=1) / np.sqrt(draws))
            assert lower - 5 * se <= mean <= upper + 5 * se
            rows.append(dict(d=d, f=f, sigma=sigma, lower=lower, upper=upper,
                             svd_mean=mean, standard_error=se))
    cfg = dict(d_model=64, n_layer=2, n_head=[4, 8], d_inner=[64, 192])
    assert ablation(cfg) == original_nsc(cfg)
    return dict(seed=20261119, draws=draws, cases=rows)


if __name__ == '__main__':
    results = main()
    path = Path(__file__).resolve().parent / 'results' / 'numerical.json'
    path.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Passed {len(results["cases"])} dimension/scale checks and exact ablation.')
