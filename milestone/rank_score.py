from serial_score import ablation, architecture_score


def score(config):
    return architecture_score(config, 'rank')
