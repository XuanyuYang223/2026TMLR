"""Read-only coverage/version/arithmetic audit of completed diagnostic outputs.

This never starts training or writes any inherited/protected artifact.  A CSV
cell arithmetic check is not an independent activation or metric recomputation.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
OLD = WORKSPACE / 'diagnostics/permworld_scope_inventory_20261009'
DEV = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
LENGTHS = set(range(10, 31))
UNIT = ('domain', 'cohort', 'relation', 'family', 'kind', 'branch',
        'control', 'metric', 'source_unit', 'seed', 'world')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def js(path):
    return json.loads(Path(path).read_text())


def read(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def finite(value):
    if value in ('', None):
        return None
    result = float(value)
    assert math.isfinite(result), 'Nonfinite value must be explicitly undefined'
    return result


def arithmetic(rows, delta, correct, wrong):
    counts = dict(preference_identities=0, delta_identities=0, change_identities=0)
    for row in rows:
        for value in (row.get(k) for k in ('B0', 'Bt', delta, correct, wrong)):
            finite(value)
        for condition, preference in [('initial', 'B0'), ('trained', 'Bt')]:
            a = finite(row.get(condition + '_correct'))
            b = finite(row.get(condition + '_wrong'))
            p = finite(row.get(preference))
            if a is not None and b is not None and p is not None:
                assert abs(p - (a - b)) < 2e-9
                counts['preference_identities'] += 1
        a, b, d = (finite(row.get(k)) for k in ('Bt', 'B0', delta))
        if all(v is not None for v in (a, b, d)):
            assert abs(d - (a - b)) < 2e-9
            counts['delta_identities'] += 1
        a, b = (finite(row.get(k)) for k in (correct, wrong))
        if all(v is not None for v in (a, b, d)):
            assert abs(d - (a - b)) < 2e-9
            counts['change_identities'] += 1
    return counts


def coverage(rows, keys, expected_rows):
    assert len(rows) == expected_rows, (len(rows), expected_rows)
    cells = {}
    for row in rows:
        key = tuple(row[k] for k in keys) + (int(row['length']),)
        assert key not in cells, ('Duplicate cell', key)
        cells[key] = row
    groups = {}
    for key in cells:
        groups.setdefault(key[:-1], set()).add(key[-1])
    assert all(lengths == LENGTHS for lengths in groups.values())
    return cells, dict(rows=len(rows), unique_cells=len(cells),
                       complete_length_groups=len(groups), lengths=sorted(LENGTHS))


def existing():
    frozen = js(ROOT / 'freeze.json')
    assert sha(ROOT / 'protocol.json') == frozen['protocol_sha256']
    for path, digest in frozen['files'].items():
        assert sha(path) == digest, ('Frozen dependency changed', path)
    rows = read(ROOT / 'layer_metric_results.csv')
    cells, result = coverage(rows, UNIT, 187488)
    assert all(row['initialization_grade'] == 'C' and
               row['original_initial_weight_identity_verified'] == 'False' for row in rows)
    result['decomposition_checks'] = arithmetic(rows, 'delta', 'delta_correct', 'delta_wrong')
    inherited = read(OLD / 'permutation_metric_results.csv')
    assert len(inherited) == 52080
    values = 0
    for old in inherited:
        branch = 'final_norm_end' if old['branch'] == 'prefix_final_norm' else old['branch']
        key = tuple(branch if field == 'branch' else old[field] for field in UNIT) + (int(old['length']),)
        new = cells[key]
        for field, value in old.items():
            expected = branch if field == 'branch' else value
            assert new[field] == expected, ('Inherited field changed', key, field)
            values += 1
        assert new['provenance'] == 'reused_frozen_metric_row'
    result['inherited_rows_exact'] = len(inherited)
    result['inherited_string_fields_exact'] = values
    result['frozen_dependencies'] = len(frozen['files'])
    result['original_initial_weight_identity_verified'] = False
    result['initialization_grade'] = 'C'
    result['new_existing_source_updates'] = 0
    return result


def development():
    config = js(DEV / 'protocol.json')
    frozen = js(DEV / 'freeze_manifest.json')
    assert sha(DEV / 'protocol.json') == frozen['protocol_sha256']
    for relative, digest in frozen['source_sha256'].items():
        assert sha(WORKSPACE / relative) == digest
    manifest = config['training_manifest']
    assert len(manifest) == 16
    assert len({(job['architecture'], job['encoding'], job['task']) for job in manifest}) == 16
    initializations = {}
    for architecture in ('transformer', 'gru'):
        folder = DEV / 'initializations' / architecture
        metadata = js(folder / 'step0.json')
        assert metadata['actual_initial_identity_verified'] is True
        assert metadata['protocol_sha256'] == frozen['protocol_sha256']
        assert metadata['checkpoint_sha256'] == sha(folder / 'step0.pt')
        initializations[architecture] = metadata
    completed = []
    for job in manifest:
        folder = DEV / 'models' / job['run_id']
        result = js(folder / 'completed.json')
        assert result['step'] == result['finite_gradient_updates'] == job['updates'] == 20000
        assert result['checkpoint_sha256'] == sha(folder / 'checkpoint.pt')
        assert result['protocol_sha256'] == frozen['protocol_sha256']
        assert result['actual_initial_identity_verified'] is True
        for field in ('run_id', 'architecture', 'encoding', 'task',
                      'model_seed', 'data_seed', 'batch_seed'):
            assert result[field] == job[field], ('Manifest identity differs', field)
        assert result['initial_parameter_sha256'] == initializations[job['architecture']]['parameter_sha256']
        assert result['exposures'] == 20000 * 64
        assert result['supervised_tokens'] == 2 * 20000 * 64
        assert [row['step'] for row in result['curve']] == list(range(1000, 20001, 1000))
        completed.append(result)
    data_ready = js(DEV / 'data_ready.json')
    assert len({row['data_sha256'] for row in completed}) == 1
    assert all(row['data_sha256'] == data_ready['data_sha256'] for row in completed)
    assert len({row['execution_source_version_sha256'] for row in completed}) == 1
    assert data_ready['within_world_orbit_disjoint'] is True
    assert data_ready['shared_underlying_objects_and_batches'] is True
    keys = ('architecture', 'encoding', 'relation', 'control', 'branch', 'metric')
    rows = read(DEV / 'geometry_length.csv')
    _, details = coverage(rows, keys, 2688)
    details['decomposition_checks'] = arithmetic(rows, 'Delta', 'change_correct', 'change_wrong')
    summary = read(DEV / 'geometry_summary.csv')
    interactions = read(DEV / 'encoding_interactions.csv')
    assert len(summary) == 128 and len(interactions) == 64
    indexed = {tuple(row[k] for k in keys): row for row in summary}
    for row in interactions:
        a = indexed[(row['architecture'], 'one_line', row['relation'], row['control'], row['branch'], row['metric'])]
        b = indexed[(row['architecture'], 'canonical_cycle', row['relation'], row['control'], row['branch'], row['metric'])]
        for field, gamma in [('B0', 'Gamma0'), ('Bt', 'Gammat'),
                             ('change_correct', 'Gamma_correct'), ('change_wrong', 'Gamma_wrong'), ('Delta', 'GammaDelta')]:
            left, right, difference = finite(a[field]), finite(b[field]), finite(row[gamma])
            if left is None or right is None:
                assert difference is None
            else:
                assert difference is not None and abs(difference - (right - left)) < 2e-9
    assert len(read(DEV / 'task_accuracy_length.csv')) == 2016
    assert len(read(DEV / 'gold_baseline_length.csv')) == 8064
    independent = js(DEV / 'independent_evaluation.json')
    assert independent['status'] == 'passed' and independent['activation_vectors_checked'] == 2304
    engineering = js(DEV / 'engineering_complete.json')
    assert engineering['formal_training_started'] is False and engineering['stop_after_report'] is True
    # A failed final source learning gate is a reportable engineering result;
    # this audit checks faithful completion, not a favorable task/geometry effect.
    details.update(models=16, updates_per_model=20000, source_worlds=1,
                   engineering_status=engineering['status'],
                   actual_initial_identity_verified=True, formal_training_started=False,
                   initial_parameters_shared_within_architecture=True,
                   initial_parameters_shared_across_architectures=False,
                   independent_activation_vectors=2304,
                   independent_cka_scope='Fixed first relationship n10, all architectures/encodings/controls/branches/roles; not every CSV cell',
                   data_ready_sha256=sha(DEV / 'data_ready.json'))
    return details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-only', action='store_true')
    args = parser.parse_args()
    protected = js(ROOT / 'protected_f17_before.json')
    for relative, digest in protected.items():
        assert sha(WORKSPACE / relative) == digest, ('Protected F17 artifact changed', relative)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='passed',
                  protected_f17_files_unchanged=len(protected), existing=existing(),
                  audit_scope='Coverage, hashes and arithmetic; independent activation/metric checks have their separately stated scopes')
    if not args.existing_only:
        result['development'] = development()
    path = ROOT / ('existing_final_audit.json' if args.existing_only else 'final_audit.json')
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
