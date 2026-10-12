"""Frozen supplementary readers for existing scientific CSV/JSON only.

No activations, inputs, predictions, model inference, training, significance
tests, metric fitting, or outcome-selected subsets. --fixture uses synthetic
tables in a temporary directory. Missing GraphWorld results remain pending.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PERM = HERE.parent
GRAPH = PERM.parent / 'graphworld_pilot_20261010'
FIELDS = tuple(f'{phase}_{role}' for phase in ('initial', 'trained')
               for role in ('correct', 'wrong', 'identity')) + ('B0', 'Bt', 'dc', 'dw', 'di', 'Delta')
KEYS = ('architecture', 'encoding', 'relation', 'control', 'branch', 'layer', 'metric')
METRICS = ('linear_cka', 'knn_overlap', 'procrustes')
CKAS = ('linear_cka', 'debiased_cka')
COUPLED = 'Descriptive coupled cells; one world and shared initial weights; not independent repetitions'


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def read(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write(path, records):
    require(bool(records), 'An empty result table must not masquerade as completed')
    keys = list(dict.fromkeys(key for row in records for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def number(value):
    return float(value) if value not in ('', None, 'None', 'nan', 'NaN') else float('nan')


def sign(value):
    value = number(value)
    return '?' if not math.isfinite(value) else '+' if value > 0 else '-' if value < 0 else '0'


def flip(first, second):
    a, b = sign(first), sign(second)
    return 'undefined' if '?' in (a, b) else 'yes' if {a, b} == {'+', '-'} else 'no'


def fixed_mean(records, field, lengths, length_key='length'):
    require(sorted(int(row[length_key]) for row in records) == lengths,
            'A fixed-length aggregate is missing or duplicates a cell')
    values = [number(row.get(field)) for row in records]
    return math.fsum(values) / len(lengths) if all(map(math.isfinite, values)) else float('nan')


def equal(first, second):
    a, b = number(first), number(second)
    require((math.isnan(a) and math.isnan(b)) or
            (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-10),
            f'Existing-table arithmetic differs: {first} / {second}')


def component_coverage(source, summaries, keys, lengths, length_key='length'):
    groups = defaultdict(list)
    for row in source:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, records in groups.items():
        require(key in summaries, 'Missing corresponding scientific aggregate')
        identity = dict(zip(keys, key))
        for field in FIELDS:
            missing = sorted(int(row[length_key]) for row in records if not math.isfinite(number(row.get(field))))
            expected = fixed_mean(records, field, lengths, length_key)
            actual = summaries[key].get(field)
            equal(actual, expected)
            result.append(identity | dict(component=field, undefined_lengths=json.dumps(missing),
                undefined_length_count=len(missing), fixed_length_count=len(lengths),
                aggregate=actual, defined_aggregate=math.isfinite(number(actual)),
                fixed_weights='1/' + str(len(lengths)), missing_weights_reallocated=False,
                cell_dependence=COUPLED,
                static_reference=summaries[key].get('static_reference', False)))
    return result


def add_values(row, prefix, scientific):
    for field in FIELDS:
        row[prefix + field] = scientific.get(field)


def direction_fields(row, get, controls):
    for control in controls:
        for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
            signs = {}
            for metric in METRICS:
                value = get(control, metric).get(field)
                row[control + '_' + metric + '_' + field] = value
                signs[metric] = sign(value)
            row[control + '_' + field + '_directions'] = json.dumps(signs, sort_keys=True)
            row[control + '_' + field + '_agreement'] = len(set(signs.values())) == 1 and '?' not in signs.values()


def permworld(diag, output, fixture=False):
    protocol = js(diag / 'protocol.json')
    lengths = protocol['lengths']
    require(lengths == list(range(10, 31)), 'PermWorld21 fixed lengths changed')
    names = ['per_length_results.csv', 'contrast_summary.csv', 'source_learning.csv', 'source_learning_summary.csv']
    if not fixture:
        audited = js(diag / 'metric_review/final_reader_audit.json')
        require(audited['status'] == 'passed',
                'Run independent scientific-table audit before supplementary interpretation')
        require(js(diag / 'report_audit.json')['status'] == 'passed', 'Frozen reader report has not completed')
        for name in names:
            require(sha(diag / name) == audited['input_sha256'][name], 'Table differs from independent audit: ' + name)
    source, summary, learning, learning_summary = (read(diag / name) for name in names)
    require(len(source) == 24192 and len(summary) == 2912 and len(learning) == 16128 and
            len(learning_summary) == 768, 'Partial PermWorld output cannot be interpreted as completed')
    conditions = {tuple(row[key] for key in KEYS): row for row in summary if row['contrast_type'] == 'condition_summary'}
    require(len(conditions) == 1152, 'Expected all1152 coupled scientific condition groups')
    coverage = component_coverage(source, conditions, KEYS, lengths)
    require(len(coverage) == 13824, 'Twelve components of every condition must remain visible')
    write(output / 'supplementary_component_coverage.csv', coverage)
    primary, directions, layers, encodings = [], [], [], []
    for relation in protocol['relations']:
        r = relation['pair_id']
        for architecture in protocol['architectures']:
            for encoding in protocol['encodings']:
                identity = dict(relation=r, architecture=architecture, encoding=encoding,
                    right_label_offset=relation['right_label_offset'], transform=relation['input_transform'],
                    cell_dependence=COUPLED)
                get = lambda control, metric, layer='final_norm', branch='hidden': conditions[
                    architecture, encoding, r, control, branch, layer, metric]
                row = dict(identity)
                add_values(row, '', get('answer_strata', 'linear_cka'))
                for control in protocol['controls']:
                    for metric in CKAS:
                        add_values(row, f'{control}_{metric}_', get(control, metric))
                        add_values(row, f'output_{control}_{metric}_', get(control, metric, 'output', 'output_prob'))
                row['interpretation'] = 'S first, then Bt, then B0/dc/dw/Delta; hidden minus output Delta is not a mechanism estimate'
                primary.append(row)
                row = dict(identity)
                direction_fields(row, get, protocol['controls'])
                directions.append(row)
                order = [f'block{i}' if architecture == 'transformer' else f'layer{i}' for i in range(1, 5)]
                order += ['final_norm', 'embedding']
                reference = order[3]
                for control in protocol['controls']:
                    for metric in CKAS:
                        for layer in order:
                            current = get(control, metric, layer)
                            anchor = get(control, metric, reference)
                            row = identity | dict(control=control, metric=metric, layer=layer,
                                fixed_layer_reference=reference, layer_number_or_norm=layer,
                                embedding_status='structural_constant_undefined' if layer == 'embedding' else 'fixed_endpoint',
                                Bt_direction=sign(current['Bt']), Delta_direction=sign(current['Delta']),
                                Bt_flip_vs_layer4=flip(anchor['Bt'], current['Bt']),
                                Delta_flip_vs_layer4=flip(anchor['Delta'], current['Delta']))
                            add_values(row, '', current)
                            for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
                                row[field + '_minus_layer4'] = number(current[field]) - number(anchor[field])
                            line = conditions[architecture, 'one_line', r, control, 'hidden', layer, metric]
                            cycle = conditions[architecture, 'canonical_cycle', r, control, 'hidden', layer, metric]
                            row['Bt_encoding_flip'] = flip(line['Bt'], cycle['Bt'])
                            row['Delta_encoding_flip'] = flip(line['Delta'], cycle['Delta'])
                            row['norm_vs_layer4_only_changes_normalization'] = layer == 'final_norm'
                            layers.append(row)
            row = dict(relation=r, architecture=architecture, cell_dependence=COUPLED,
                       orientation='canonical_cycle minus one_line')
            for control in protocol['controls']:
                for metric in CKAS:
                    line = conditions[architecture, 'one_line', r, control, 'hidden', 'final_norm', metric]
                    cycle = conditions[architecture, 'canonical_cycle', r, control, 'hidden', 'final_norm', metric]
                    contrast = next(value for value in summary if value['contrast_type'] == 'encoding_difference'
                        and value['relation'] == r and value['architecture'] == architecture and value['control'] == control
                        and value['branch'] == 'hidden' and value['layer'] == 'final_norm' and value['metric'] == metric)
                    prefix = f'{control}_{metric}_'
                    for field in ('B0', 'Bt', 'dc', 'dw', 'Delta'):
                        row[prefix + 'one_line_' + field] = line[field]
                        row[prefix + 'cycle_' + field] = cycle[field]
                    for field in ('Gamma0', 'Gammat', 'Gammac', 'Gammaw', 'GammaDelta'):
                        row[prefix + field] = contrast[field]
                    row[prefix + 'Bt_flip'] = flip(line['Bt'], cycle['Bt'])
                    row[prefix + 'Delta_flip'] = flip(line['Delta'], cycle['Delta'])
                    row[prefix + 'Delta_flip_without_Bt_flip'] = (row[prefix + 'Delta_flip'] == 'yes' and row[prefix + 'Bt_flip'] == 'no')
            encodings.append(row)
    require((len(primary), len(directions), len(layers), len(encodings)) == (32, 32, 768, 16), 'Fixed readout scope incomplete')
    write(output / 'main32_final_cka_readout.csv', primary)
    write(output / 'three_metric_directions32.csv', directions)
    write(output / 'all_layer_directions.csv', layers)
    write(output / 'encoding16_decomposition.csv', encodings)
    lkeys = ('architecture', 'encoding', 'task', 'condition', 'split', 'action')
    lindex = {tuple(row[key] for key in lkeys): row for row in learning_summary}
    require(len(lindex) == 768, 'Duplicated source-learning summaries')
    lgroups = defaultdict(list)
    for row in learning:
        lgroups[tuple(row[key] for key in lkeys)].append(row)
    sources, curves = [], []
    for job in protocol['checkpoint_inventory']:
        key = job['architecture'], job['encoding'], job['task'], 'trained', 'test', 'identity'
        original = lindex[key]
        detail = lgroups[key]
        gain = number(original['accuracy_minus_length_mode'])
        row = {key: job[key] for key in ('run_id', 'architecture', 'encoding', 'task', 'parameters')}
        for field in ('answer_accuracy', 'answer_ce', 'minimum_length_accuracy', 'length_majority_accuracy', 'accuracy_minus_length_mode'):
            row[field] = original[field]
        for field in ('answer_accuracy', 'answer_ce', 'length_majority_accuracy', 'accuracy_minus_length_mode'):
            equal(original[field], fixed_mean(detail, field, lengths))
        row.update(gain_over_train_defined_length_mode_pp=100 * gain,
            baseline_comparison='undefined' if not math.isfinite(gain) else
                'at_or_below_train_defined_length_mode' if gain <= 0 else 'above_train_defined_length_mode',
            descriptive_not_learning_gate=True, independent_worlds=1,
            learning_difference_prevents_equal_mastery_mechanism_claim=True,
            length_gain_directions=''.join(sign(value['accuracy_minus_length_mode'])
                for value in sorted(detail, key=lambda value: int(value['length']))), cell_dependence=COUPLED)
        completion = js(job['completed_path'])
        require(int(completion['step']) == 20000, 'Reader must use final20k sources only')
        row['curve_completion_path'] = job['completed_path']
        row['curve_completion_sha256'] = sha(job['completed_path'])
        for value in completion['curve']:
            curves.append({key: job[key] for key in ('run_id', 'architecture', 'encoding', 'task')} | dict(value,
                fixed_endpoint=20000, curve_not_for_checkpoint_selection=True, cell_dependence=COUPLED))
        sources.append(row)
    require(len(sources) == 64, 'All64 source models, including weak learners, must be retained')
    write(output / 'source64_learning_and_baseline.csv', sources)
    write(output / 'source_learning_curves.csv', curves)
    return dict(status='completed_readout', scientific_metrics_recomputed=0, training_updates=0,
        source_models=64, relation_instances=8, transform_types=2, independent_worlds=1,
        counts_are_coupled_not_independent_repetitions=True, component_coverage_rows=len(coverage),
        primary_condition_rows=32, metric_direction_rows=32, all_layer_rows=768, encoding_rows=16,
        source_learning_rows=64, curve_rows=len(curves), missing_graph_not_imputed=True,
        input_sha256={name: sha(diag / name) for name in names}, fixture_only=fixture)


def graphworld(diag, output, fixture=False):
    names = ['geometry_per_n.csv', 'geometry_summary.csv', 'source_learning.csv', 'evaluation_complete.json']
    missing = [name for name in names if not (diag / name).is_file()]
    if missing:
        return dict(status='pending', missing_artifacts=missing, scientific_metrics_recomputed=0,
                    training_updates=0, no_graph_results_fabricated=True)
    protocol = js(diag / 'protocol.json')
    require(js(diag / 'evaluation_complete.json')['status'] == 'completed', 'Graph evaluation not complete')
    require(protocol['lengths'] == [6, 7, 8], 'Graph fixed3 sizes changed')
    source, summary, learning = (read(diag / name) for name in names[:3])
    require((len(source), len(summary), len(learning)) == (720, 240, 240), 'Partial GraphWorld grid')
    keys = ('relation', 'control', 'branch', 'metric')
    conditions = {tuple(row[key] for key in keys): row for row in summary}
    require(len(conditions) == 240, 'Duplicated graph summary cells')
    coverage = component_coverage(source, conditions, keys, [6, 7, 8], 'n')
    write(output / 'graph_component_coverage.csv', coverage)
    controls = protocol['evaluation']['controls']
    require(controls == ['raw_within_n', 'answer_strata', 'answer_density_strata'], 'Graph control grid changed')
    primary, directions = [], []
    for relation in protocol['relations']:
        r = relation['relation']
        get = lambda control, metric, branch='prefix_final': conditions[r, control, branch, metric]
        row = dict(relation=r, left=relation['left'], right=relation['right'],
            correct_transform=relation['correct'], wrong_transform=relation['wrong'],
            mapped_right=relation['mapping'], cell_dependence=COUPLED)
        add_values(row, '', get('answer_strata', 'linear_cka'))
        for control in controls:
            for metric in CKAS:
                add_values(row, f'{control}_{metric}_', get(control, metric))
                add_values(row, f'output_{control}_{metric}_', get(control, metric, 'output_prob'))
        row['cross_domain_limitation'] = 'Changes graph size/input/task/distribution as well as domain; no PermWorld minus GraphWorld causal subtraction'
        primary.append(row)
        row = dict(relation=r, cell_dependence=COUPLED)
        direction_fields(row, get, controls)
        directions.append(row)
    require(len(primary) == len(directions) == 4, 'All4 coupled graph relations must be retained')
    write(output / 'graph_main4_readout.csv', primary)
    write(output / 'graph_three_metric_directions4.csv', directions)
    sources = []
    for job in protocol['training_manifest']:
        records = [row for row in learning if row['task'] == job['task'] and row['condition'] == 'trained'
                   and row['split'] == 'analysis' and row['action'] == 'identity']
        row = {key: job[key] for key in ('run_id', 'architecture', 'encoding', 'task')}
        for field in ('answer_accuracy', 'answer_ce', 'mode_accuracy', 'edge_mean', 'density_mean'):
            row[field] = fixed_mean(records, field, [6, 7, 8], 'n')
        row.update(gain_over_train_defined_mode_pp=100 * (row['answer_accuracy'] - row['mode_accuracy']),
            minimum_size_accuracy=min(number(value['answer_accuracy']) for value in records),
            EOS_evaluated=False, descriptive_not_learning_gate=True, cell_dependence=COUPLED)
        sources.append(row)
    require(len(sources) == 4, 'All4 graph source tasks must remain visible')
    write(output / 'graph_source4_learning.csv', sources)
    return dict(status='completed_readout', scientific_metrics_recomputed=0, training_updates=0,
        source_models=4, relation_instances=4, transform_types=2, independent_worlds=1,
        counts_are_coupled_not_independent_repetitions=True, primary_relation_rows=4,
        component_coverage_rows=len(coverage), input_sha256={name: sha(diag / name) for name in names}, fixture_only=fixture)


def fixture():
    """Complete synthetic schemas, including identity-only undefined coverage."""
    # Only task/configuration metadata are read from current protocols.
    p = js(PERM / 'protocol.json')
    g = js(GRAPH / 'protocol.json')
    sys.path.insert(0, str(PERM.parents[1]))
    from experiments.permworld_eight_relations_20261010 import analysis_helpers as helpers
    with tempfile.TemporaryDirectory(prefix='supplementary-readout-synthetic-') as directory:
        root = Path(directory); perm = root / 'perm'; graph = root / 'graph'; out = root / 'out'
        perm.mkdir(); graph.mkdir(); out.mkdir()
        data = []
        for a in p['architectures']:
            order = [f'block{i}' if a == 'transformer' else f'layer{i}' for i in range(1, 5)] + ['embedding']
            branches = [('hidden', 'final_norm', METRICS + ('debiased_cka',)), ('output_prob', 'output', METRICS + ('debiased_cka',))]
            branches += [('hidden', layer, CKAS) for layer in order]
            for e in p['encodings']:
                for rel in p['relations']:
                    for control in p['controls']:
                        for branch, layer, metrics in branches:
                            for metric in metrics:
                                for n in p['lengths']:
                                    values = {phase: {role: (float('nan') if layer == 'embedding' else
                                        -.4 if metric == 'procrustes' else .4) for role in ('correct', 'wrong', 'identity')}
                                        for phase in ('initial', 'trained')}
                                    if layer != 'embedding':
                                        base = -.9 if metric == 'procrustes' else 0.
                                        values['initial']['correct'] = base + (.9 if e == 'canonical_cycle' else .4)
                                        values['initial']['wrong'] = base + (.1 if e == 'canonical_cycle' else .4)
                                        values['trained']['correct'] = base + (.55 if e == 'canonical_cycle' else .5)
                                        values['trained']['wrong'] = base + (.5 if e == 'canonical_cycle' else .4)
                                        if layer in ('block1', 'layer1'):
                                            values['trained']['wrong'] = base + .7
                                    # This case demonstrates that Delta can be
                                    # defined while an identity cell is not.
                                    if layer == 'final_norm' and metric == 'knn_overlap' and n == 10:
                                        values['initial']['identity'] = float('nan')
                                    fields = helpers.component_values(values)
                                    errors = {key: -fields[key.removeprefix('error_')] if metric == 'procrustes' else float('nan')
                                              for key in helpers.ERROR_FIELDS}
                                    data.append(dict(architecture=a, encoding=e, relation=rel['pair_id'], control=control,
                                        branch=branch, layer=layer, metric=metric, length=n, **fields, **errors))
        summaries, _ = helpers.summarize(data, p['lengths'])
        write(perm / 'per_length_results.csv', data); write(perm / 'contrast_summary.csv', summaries)
        learning, aggregates = [], []
        for job in p['checkpoint_inventory']:
            for phase in ('initial', 'trained'):
                for split in ('validation', 'test'):
                    for action in p['actions']:
                        identity = {key: job[key] for key in ('architecture', 'encoding', 'task')}
                        identity.update(condition=phase, split=split, action=action)
                        records = [dict(identity, length=n, answer_accuracy=.6, answer_ce=.8,
                            length_majority_accuracy=.4, accuracy_minus_length_mode=.2) for n in p['lengths']]
                        learning.extend(records)
                        aggregates.append(dict(identity, answer_accuracy=.6, answer_ce=.8, minimum_length_accuracy=.6,
                            length_majority_accuracy=.4, accuracy_minus_length_mode=.2))
            cp = perm / (job['run_id'] + '.json')
            json_write(cp, dict(step=20000, curve=[dict(step=step, answer_accuracy=.6)
                for step in range(1000, 20001, 1000)]))
            job['completed_path'] = str(cp)
        json_write(perm / 'protocol.json', p)
        write(perm / 'source_learning.csv', learning); write(perm / 'source_learning_summary.csv', aggregates)
        first = permworld(perm, out, fixture=True)
        coverage = read(out / 'supplementary_component_coverage.csv')
        require(any(row['component'] == 'initial_identity' and row['metric'] == 'knn_overlap' and
            row['undefined_lengths'] == '[10]' and row['defined_aggregate'] == 'False' for row in coverage),
            'Identity-only undefined component was hidden by defined Delta')
        require(any(row['component'] == 'Delta' and row['metric'] == 'knn_overlap' and
            row['defined_aggregate'] == 'True' for row in coverage), 'Fixture should retain Delta in that case')
        encoding = read(out / 'encoding16_decomposition.csv')
        require(all(row['answer_strata_linear_cka_Delta_flip'] == 'yes' and
            row['answer_strata_linear_cka_Bt_flip'] == 'no' and
            row['answer_strata_linear_cka_Delta_flip_without_Bt_flip'] == 'True' for row in encoding),
            'Delta flip was incorrectly labeled as trained-preference flip')
        layers = read(out / 'all_layer_directions.csv')
        require(any(row['layer'] in ('block1', 'layer1') and row['Bt_flip_vs_layer4'] == 'yes' for row in layers),
            'Fixed first-layer versus fourth-layer flip was not retained')
        require(all(row['Bt_flip_vs_layer4'] == 'undefined' for row in layers if row['layer'] == 'embedding'),
            'Undefined constant layer manufactured a numeric flip')
        json_write(graph / 'protocol.json', g)
        pending = graphworld(graph, out, fixture=True)
        require(pending['status'] == 'pending' and not (out / 'graph_main4_readout.csv').exists(), 'Pending graph fabricated')
        per_n, summaries = [], []
        for rel in g['relations']:
            for control in g['evaluation']['controls']:
                for branch in ('prefix_final', 'output_prob', 'gold_onehot', 'gold_scalar', 'input_onehot'):
                    static = branch.startswith('gold') or branch == 'input_onehot'
                    for metric in METRICS + ('debiased_cka',):
                        fields = {key: float('nan') if static and (key.startswith('initial') or key in ('B0', 'dc', 'dw', 'di', 'Delta'))
                                  else -.4 if metric == 'procrustes' and '_' in key else .4 if '_' in key else 0. for key in FIELDS}
                        identity = dict(relation=rel['relation'], control=control, branch=branch, metric=metric)
                        per_n.extend(dict(identity, n=n, **fields) for n in (6, 7, 8))
                        summaries.append(dict(identity, static_reference=static, **fields))
        learning = [dict(task=job['task'], condition=phase, split=split, action=action, n=n,
            answer_accuracy=.6, answer_ce=.8, mode_accuracy=.4, edge_mean=10., density_mean=.4)
            for job in g['training_manifest'] for phase in ('initial', 'trained')
            for split in ('validation', 'analysis') for action in g['actions'] for n in (6, 7, 8)]
        write(graph / 'geometry_per_n.csv', per_n); write(graph / 'geometry_summary.csv', summaries)
        write(graph / 'source_learning.csv', learning); json_write(graph / 'evaluation_complete.json', dict(status='completed'))
        second = graphworld(graph, out, fixture=True)
    result = dict(status='passed', synthetic_only=True, no_real_geometry_read=True,
        no_metrics_recomputed=True, no_training=True, permworld=first, graphworld=second,
        graph_pending_case_passed=True, identity_only_undefined_case_passed=True,
        Delta_flip_without_Bt_flip_case_passed=True, all_layers_and_undefined_flip_case_passed=True,
        source_sha256=sha(__file__))
    json_write(HERE / 'synthetic_readout_fixture.json', result)
    return result


def verify_reader(stage):
    frozen = js(HERE / 'readout_freeze.json')
    require(frozen['source_sha256'] == sha(__file__), 'Supplementary reader bytes differ from freeze')
    diag = PERM if stage == 'permworld' else GRAPH
    expected = frozen['protocol_sha256'][stage]
    require(sha(diag / 'protocol.json') == expected, 'Scientific protocol differs from supplementary reader binding')
    return diag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['permworld', 'graphworld'])
    parser.add_argument('--fixture', action='store_true')
    args = parser.parse_args()
    if args.fixture:
        require(args.stage is None, 'Fixture and real stages are distinct')
        result = fixture()
    else:
        require(args.stage is not None, 'Specify --stage permworld or --stage graphworld')
        diag = verify_reader(args.stage)
        output = HERE / args.stage; output.mkdir(exist_ok=True)
        result = (permworld if args.stage == 'permworld' else graphworld)(diag, output)
        result.update(reader_sha256=sha(__file__), readout_freeze_sha256=sha(HERE / 'readout_freeze.json'))
        json_write(output / 'readout_status.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
