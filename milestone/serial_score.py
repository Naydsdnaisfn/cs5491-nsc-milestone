import math

from nsc import layer_capacities, layers, psi_mp


def product_moments(d, f, sigma=0.02):
    variance_squared = sigma ** 4
    return (d * d * f * variance_squared,
            variance_squared * ((d + 1) * (f + d + 1) + f * d + 2))


def serial_capacity(d, f, method, sigma=0.02):
    trace, ratio = product_moments(d, f, sigma)
    if method == 'trace':
        return trace
    if method == 'rank':
        rank = min(d, f)
        return rank * math.log1p(trace / rank)
    if method == 'moment':
        return trace * (math.log1p(ratio) / ratio) if ratio else 0.0
    raise ValueError(f'Unknown method: {method}')


def architecture_score(config, method):
    d, heads, inner = layers(config)
    return math.fsum(3 * h * psi_mp(d, d // h) + psi_mp(d, d)
                     + serial_capacity(d, f, method)
                     for h, f in zip(heads, inner))


def score(config):
    return architecture_score(config, 'moment')


def ablation(config):
    return math.fsum(layer_capacities(config))
