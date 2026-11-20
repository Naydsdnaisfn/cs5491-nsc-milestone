import itertools
import json
import math
from pathlib import Path
import statistics

from evaluate import BUDGETS, K_VALUES, canonical, pair_summary, select_trials
from nsc import layers, original_nsc, resource_estimate
from serial_score import ablation, architecture_score

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'starter' / 'data'


def read(path):
    return json.loads(path.read_text())


def order(values, i, j):
    return (values[i] > values[j]) - (values[i] < values[j])


def main():
    keys = read(DATA / 'splits.json')['development']
    all_configs = read(DATA / 'architectures.json')
    resources = read(DATA / 'resources.json')
    labels = read(DATA / 'development_labels.json')
    configs = [all_configs[key] for key in keys]
    totals = [resources[key]['total'] for key in keys]
    ppl = [labels[key]['valid_ppl'] for key in keys]
    quality = [-p for p in ppl]
    dimensions = [layers(cfg) for cfg in configs]
    signatures = [(d, len(heads), sum(inner)) for d, heads, inner in dimensions]
    assert all(ablation(cfg) == original_nsc(cfg) for cfg in configs)
    scores = {method: [canonical(architecture_score(c, method)) for c in configs]
              for method in ('rank', 'moment', 'trace')}
    scores['original'] = [canonical(original_nsc(c)) for c in configs]
    scores['flops'] = [resource_estimate(c)['reference_flops_per_token'] for c in configs]
    scores['parameters'] = totals
    pairs = list(itertools.combinations(range(len(keys)), 2))
    groups = {
        'all': pairs,
        'same_width_near_parameters': [(i, j) for i, j in pairs
            if signatures[i][0] == signatures[j][0] and min(totals[i], totals[j]) / max(totals[i], totals[j]) >= .9],
        'same_width_depth_total_ffn_width': [(i, j) for i, j in pairs if signatures[i] == signatures[j]],
    }
    comparisons = {}
    for method in ('rank', 'moment'):
        comparisons[method] = {}
        for label, group in groups.items():
            comparisons[method][label] = {}
            for baseline in ('original', 'trace', 'flops', 'parameters', 'rank' if method == 'moment' else 'moment'):
                changed = [(i, j) for i, j in group if order(scores[method], i, j) != order(scores[baseline], i, j)]
                wins = [(i, j) for i, j in changed if order(scores[method], i, j) == order(quality, i, j)]
                losses = [(i, j) for i, j in changed if order(scores[baseline], i, j) == order(quality, i, j)]

                def example(subset):
                    if not subset:
                        return None
                    i, j = max(subset, key=lambda pair: abs(ppl[pair[0]] - ppl[pair[1]]))
                    return [dict(id=keys[k], ppl=ppl[k], score=scores[method][k],
                                 reference_score=scores[baseline][k], parameters=totals[k],
                                 hidden_width=signatures[k][0]) for k in (i, j)]

                comparisons[method][label][baseline] = dict(
                    changed_pairs=len(changed), architectures=len({k for pair in changed for k in pair}),
                    candidate_correct=len(wins), reference_correct=len(losses),
                    example_win=example(wins), example_loss=example(losses))

    reference = read(HERE.parent / 'starter' / 'expected_development.json')
    selection = []
    for method in ('moment', 'rank'):
        current = read(HERE / 'results' / f'{method}.json')
        for baseline in ('original_nsc', 'position_example', 'reported_parameters', 'reference_flops'):
            assert current['ranking'][baseline] == reference['ranking'][baseline]
        assert current['ranking']['ablation'] == current['ranking']['original_nsc']
        assert len(current['selection']) == len(reference['selection'])
        for got, expected in zip(current['selection'], reference['selection']):
            fields = ('parameter_cap', 'k', 'eligible', 'oracle_ppl')
            assert all(got[field] == expected[field] for field in fields)
            assert got['methods']['ablation'] == got['methods']['original_nsc']
            for baseline in ('original_nsc', 'position_example', 'reported_parameters', 'reference_flops', 'random'):
                for metric, value in expected['methods'][baseline].items():
                    assert math.isclose(got['methods'][baseline][metric], value, abs_tol=1e-12)
        selection.append(dict(
            method=method,
            mean_regret=statistics.mean(row['methods']['student']['mean_regret'] for row in current['selection']),
            controlled_tau=current['ranking']['student']['size_controlled_pairs']['tau_b']))

    trace_results = []
    for budget in BUDGETS:
        pool = [i for i, size in enumerate(totals) if size <= budget]
        for k in K_VALUES:
            if len(pool) < k:
                continue
            trace_results.append(dict(budget=budget, k=k, **select_trials(pool, scores['trace'], k, ppl,
                                                                       202627 + budget + k)))
    result = dict(
        architectures=len(keys), group_sizes={name: len(group) for name, group in groups.items()},
        baseline_reproduced=True, ablation_exact=True, comparisons=comparisons,
        trace_ranking={name: pair_summary(scores['trace'], quality, group) for name, group in groups.items()},
        trace_selection=trace_results, selection_criterion=selection)
    (HERE / 'results' / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(result['group_sizes'])
    for method, groups_result in comparisons.items():
        for group, results in groups_result.items():
            print(method, group, {k: v['changed_pairs'] for k, v in results.items()})
    print(selection)


if __name__ == '__main__':
    main()
