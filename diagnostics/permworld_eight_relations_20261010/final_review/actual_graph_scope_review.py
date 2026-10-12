"""Read saved GraphWorld deliveries; no scientific module import or model execution."""
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

WORK = Path('/home/yangx/ICML')
DIAG = WORK / 'diagnostics/graphworld_pilot_20261010'
ROOT = Path('/mnt/yangx-projects-f/20261010/graphworld_pilot')
PERM = WORK / 'diagnostics/permworld_eight_relations_20261010'
OUT = PERM / 'final_review'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def number(value):
    return None if value in ('', None) else float(value)


def main():
    problems = []
    artifacts = {}

    def check(condition, message):
        if not condition:
            problems.append(message)

    def source(path):
        path = Path(path)
        artifacts[str(path)] = sha(path)
        return read(path)

    c = source(DIAG / 'protocol.json')
    pd = sha(DIAG / 'protocol.json')
    freeze = source(DIAG / 'freeze_manifest.json')
    verification = source(DIAG / 'verification_freeze.json')
    version = verification['main_source_version_sha256']
    source_bytes = dict(freeze['source_sha256'])
    source_bytes.update(verification['source_sha256'])
    for path, expected in source_bytes.items():
        check(sha(path) == expected, 'Frozen source differs: ' + path)
    check(pd == freeze['protocol_sha256'] == verification['protocol_sha256'], 'Protocol/freeze binding')
    check(sha(c['initial_checkpoint_path']) == c['initial_checkpoint_sha256'], 'Actual initial bytes')
    check(sha(c['starting_rng_path']) == c['starting_rng_sha256'], 'Archived starting RNG bytes')
    check(c['actual_initial_identity_verified'] is True, 'Actual step0 not verified')
    check(c['worlds'] == 1 and len(c['training_manifest']) == 4, 'Fixed four-source one-world scope')
    data = source(ROOT / 'world/data_audit.json')
    check(data['data_sha256'] == verification['data_sha256'] == sha(ROOT / 'world/data.npz'), 'Actual data bytes/fixed verification binding')
    training = source(ROOT / 'training_complete.json')
    evaluation = source(DIAG / 'evaluation_complete.json')
    completion = source(DIAG / 'completion.json')
    reader = source(DIAG / 'report_reader_audit.json')
    actual = source(DIAG / 'independent_actual_small_batch.json')
    embedded = source(DIAG / 'independent_metrics.json')
    bridge = source(PERM / 'scope_bridge_state.json')
    orchestration = source(PERM / 'orchestration_state.json')
    readout = source(OUT / 'graphworld/readout_status.json')
    repair_directory = PERM / 'graph_evaluation_environment_repair_01'
    repair = source(repair_directory / 'repair.json')
    packages_before = source(repair_directory / 'packages_before.json')
    packages_after = source(repair_directory / 'packages_after.json')
    preserved = source(repair_directory / 'artifacts_before.json')
    check(repair['status'] == 'completed_fixed_remaining_stages', 'Explicit evaluation environment recovery completion')
    check(repair['pinned_version'] == packages_after.get('threadpoolctl') == '3.6.0', 'Pinned missing dependency')
    check(all(packages_after.get(k) == v for k, v in packages_before.items()) and set(packages_after) - set(packages_before) == {'threadpoolctl'}, 'Existing package versions unchanged')
    check(sha(repair['module_path']) == repair['module_sha256'], 'Installed threadpoolctl module bytes')
    check(not repair['scientific_sources_changed'] and not repair['scientific_protocol_changed'] and repair['training_updates'] == repair['new_source_endpoints'] == repair['remaining_training_updates'] == 0, 'No scientific or training change in recovery')
    check(repair['all_checkpoint_and_cache_bytes_unchanged'] and not repair['duplicate_extraction_forward'], 'Recovery reused preserved extraction')
    check(len(preserved['checkpoints']) == 4 and len(preserved['cache']) == 16, 'Pre-repair preserved artifact scope')
    for group in preserved.values():
        for path, expected in group.items():
            check(sha(path) == expected, 'Pre/post-recovery bytes: ' + path)
    check(repair['failed_phase']['exit_code'] == 1 and repair['failed_phase']['wall_seconds'] > 0, 'Failed attempt retained with cost')
    artifacts[str(repair_directory / 'graph_evaluate.failed.log')] = sha(repair_directory / 'graph_evaluate.failed.log')
    artifacts[str(repair_directory / 'graph_evaluate.failed.resources.txt')] = sha(repair_directory / 'graph_evaluate.failed.resources.txt')
    check(training['status'] == 'completed' and training['source_models'] == 4, 'Actual training completion')
    check(evaluation['status'] == 'completed', 'Actual evaluation completion')
    check(completion['status'] == 'completed_and_stopped' and completion['new_source_models'] == 4, 'Actual stopped completion')
    check(reader['status'] == 'passed' and reader['source_count'] == 4 and not reader['problems'], 'Actual reader audit is not a four-source pass')
    check(reader['protocol_sha256'] == actual['protocol_sha256'] == pd, 'Actual reader/verification protocol binding')
    check(reader['verification_freeze_sha256'] == sha(DIAG / 'verification_freeze.json'), 'Actual reader verification freeze binding')
    check(reader['world_count'] == reader['shared_initial_count'] == 1, 'Reader dependence counts')
    check(actual['status'] == 'passed' and actual['failure_count'] == 0, 'Actual independent metric verification')
    check(embedded['status'] == 'passed', 'Embedded covariance verification')
    check(orchestration['status'] == 'all_fixed_scientific_stages_completed', 'Fixed pipeline has not completed')
    check(bridge['status'] == 'scientific_scope_completed_report_review_pending', 'Bridge terminal scientific state')
    check(readout['status'] == 'completed_readout' and readout['source_models'] == 4 and not readout['fixture_only'], 'Actual readout completion')
    for name, record in [('training', training), ('evaluation', evaluation), ('completion', completion)]:
        check(record['protocol_sha256'] == pd and record['source_version_sha256'] == version, name + ' binding')
    check(actual['verification_freeze_sha256'] == sha(DIAG / 'verification_freeze.json'), 'Independent verification freeze binding')

    source_records = []
    receipts = {}
    check(len(training['records']) == 4 and {r['run_id'] for r in training['records']} == {j['run_id'] for j in c['training_manifest']}, 'Training receipt manifest coverage')
    for job in c['training_manifest']:
        directory = ROOT / 'models' / job['run_id']
        r = source(directory / 'completed.json')
        receipts[job['task']] = r
        check(r['run_id'] == job['run_id'] and r['task'] == job['task'] and r['step'] == 20000, 'Endpoint identity: ' + job['run_id'])
        check(r['protocol_sha256'] == pd and r['source_version_sha256'] == version and r['data_sha256'] == data['data_sha256'], 'Endpoint source binding: ' + job['run_id'])
        check(sha(directory / 'checkpoint.pt') == r['checkpoint_sha256'], 'Endpoint checkpoint bytes: ' + job['run_id'])
        check(r['initial_parameter_sha256'] == c['initial_parameter_sha256'] and r['initial_checkpoint_sha256'] == c['initial_checkpoint_sha256'] and r['starting_rng_hash'] == c['starting_rng_hash'] and r['actual_initial_identity_verified'], 'Endpoint true initial binding: ' + job['run_id'])
        check(r['finite_gradient_updates'] == 20000 and r['maximum_gradient_norm'] > 0 and r['final_parameter_sha256'] != r['initial_parameter_sha256'], 'Updated finite-gradient endpoint: ' + job['run_id'])
        check([x['step'] for x in r['curve']] == list(range(1000, 20001, 1000)), 'Complete validation curve: ' + job['run_id'])
        endpoint = r['curve'][-1]['validation']
        check(endpoint['samples'] == 768, 'Endpoint validation coverage: ' + job['run_id'])
        source_records.append(dict(run_id=job['run_id'], task=job['task'], step=r['step'],
            checkpoint_sha256=r['checkpoint_sha256'], recorded_process_seconds=r['elapsed_seconds'],
            peak_cuda_memory_bytes=r['peak_cuda_memory_bytes'], invocation_start_step=r['invocation_start_step'],
            invocation_updates=r['invocation_updates'], validation_answer_accuracy=endpoint['answer_accuracy'],
            validation_eos_accuracy=endpoint['eos_accuracy'], validation_samples=endpoint['samples']))

    index = source(ROOT / 'evaluation_cache/index.json')
    check(index['status'] == 'completed' and index['logical_cache_count'] == len(index['records']) == 16, 'Sixteen caches')
    check(index['protocol_sha256'] == pd and index['source_version_sha256'] == version and index['data_sha256'] == data['data_sha256'], 'Cache index binding')
    check(index['actual_initial_identity_verified'] and index['initial_hidden_task_independent_bitwise'] and index['source_updates'] == 0, 'Cache initial/no-update audit')
    tasks = [j['task'] for j in c['training_manifest']]
    splits = ('validation', 'analysis')
    conditions = ('initial', 'trained')
    expected_cache = set(itertools.product(tasks, conditions, splits))
    keys = [(r['task'], r['condition'], r['split']) for r in index['records']]
    check(len(set(keys)) == 16 and set(keys) == expected_cache, 'Logical cache Cartesian coverage')
    cache_checks = []
    initial_hidden = {}
    manual_errors = []
    for record in index['records']:
        task, condition, split = record['task'], record['condition'], record['split']
        path = Path(record['path']); meta = read(path.with_suffix('.json')); sig = meta['signature']
        check(meta == record['metadata'] and sha(path) == meta['sha256'], 'Cache bytes/metadata: ' + path.name)
        check(sig == dict(protocol_sha256=pd, data_sha256=data['data_sha256'], source_version_sha256=version,
            checkpoint_sha256=c['initial_checkpoint_sha256'] if condition == 'initial' else receipts[task]['checkpoint_sha256'],
            task=task, condition=condition, split=split), 'Cache signature: ' + path.name)
        check(meta['parameter_sha256_before'] == meta['parameter_sha256_after'] and meta['source_updates'] == 0 and not meta['EOS_evaluated'], 'Frozen parameter/no-EOS cache: ' + path.name)
        check(meta['parameter_sha256_before'] == (c['initial_parameter_sha256'] if condition == 'initial' else receipts[task]['final_parameter_sha256']), 'Cache loaded-parameter identity: ' + path.name)
        N = c['data']['per_n'][split] * 3
        expected_shape = {'prefix_final': (N, 5, 256), 'output_prob': (N, 5, 31), 'prediction': (N, 5), 'answer_ce': (N, 5)}
        with np.load(path, allow_pickle=False) as arrays:
            check(set(arrays.files) == set(expected_shape), 'Cache array keys: ' + path.name)
            for name, shape in expected_shape.items():
                check(arrays[name].shape == shape and tuple(meta['shape'][name]) == shape and np.isfinite(arrays[name]).all(), 'Cache shape/finite: ' + path.name + '/' + name)
            check(arrays['prefix_final'].dtype == arrays['output_prob'].dtype == np.dtype('float32'), 'FP32 hidden/output cache: ' + path.name)
            if condition == 'initial':
                h = arrays['prefix_final']
                if split in initial_hidden:
                    check(np.array_equal(initial_hidden[split], h), 'Initial task-prefix bitwise equality: ' + path.name)
                else:
                    initial_hidden[split] = h.copy()
        checks = meta['manual_extraction_checks']
        if split == 'validation':
            check(len(checks) == 15, 'Extraction check coverage: ' + path.name)
            check({(x['n'], x['action']) for x in checks} == set(itertools.product((6, 7, 8), c['actions'])), 'Extraction n/action coverage: ' + path.name)
        else:
            check(not checks, 'Unexpected analysis extraction checks: ' + path.name)
        for x in checks:
            check(x['same_shape'] and x['vectors'] == 8 and x['max_abs'] <= 2e-5, 'Same-shape extraction tolerance: ' + path.name)
            manual_errors.append(x['max_abs'])
        cache_checks.append(dict(task=task, condition=condition, split=split, sha256=meta['sha256'], shape=meta['shape']))

    tables = {}
    for name in ('geometry_per_n.csv', 'geometry_summary.csv', 'source_learning.csv'):
        artifacts[str(DIAG / name)] = sha(DIAG / name); tables[name] = rows(DIAG / name)
    lengths, summary, learning = (tables[k] for k in ('geometry_per_n.csv', 'geometry_summary.csv', 'source_learning.csv'))
    relations = [r['relation'] for r in c['relations']]
    controls = c['evaluation']['controls']
    branches = ('prefix_final', 'output_prob', 'gold_onehot', 'gold_scalar', 'input_onehot')
    metrics = ('linear_cka', 'knn_overlap', 'procrustes', 'debiased_cka')
    group_fields = ('relation', 'control', 'branch', 'metric')
    expected_groups = set(itertools.product(relations, controls, branches, metrics))
    expected_lengths = {(r, str(n), ctl, b, m) for r, ctl, b, m in expected_groups for n in (6, 7, 8)}
    lk = [tuple(r[k] for k in ('relation', 'n', 'control', 'branch', 'metric')) for r in lengths]
    sk = [tuple(r[k] for k in group_fields) for r in summary]
    check(len(lk) == len(set(lk)) == 720 and set(lk) == expected_lengths, '720 unique scientific cells')
    check(len(sk) == len(set(sk)) == 240 and set(sk) == expected_groups, '240 unique summaries')
    groups = {}
    for r in lengths:
        groups.setdefault(tuple(r[k] for k in group_fields), []).append(r)
        v = {k: number(r[k]) for k in c['evaluation']['fields']}
        check(all(x is None or math.isfinite(x) for x in v.values()), 'Nonfinite numeric cell')
        for target, a, b in [('B0', 'initial_correct', 'initial_wrong'), ('Bt', 'trained_correct', 'trained_wrong'),
            ('dc', 'trained_correct', 'initial_correct'), ('dw', 'trained_wrong', 'initial_wrong'),
            ('di', 'trained_identity', 'initial_identity'), ('Delta', 'Bt', 'B0')]:
            expected = None if v[a] is None or v[b] is None else v[a] - v[b]
            check((v[target] is None) == (expected is None) and (expected is None or abs(v[target] - expected) <= 1e-9), 'Saved decomposition: ' + target)
        if r['static_reference'] == 'True':
            check(all(v[k] is None for k in ('initial_correct', 'initial_wrong', 'initial_identity', 'B0', 'dc', 'dw', 'di', 'Delta')), 'Fabricated static increment')
        if r['metric'] == 'procrustes':
            for condition, role in itertools.product(conditions, ('correct', 'wrong', 'identity')):
                error = number(r[condition + '_' + role + '_error']); value = v[condition + '_' + role]
                check((value is None) == (error is None) and (error is None or abs(value + error) <= 1e-9), 'Negative Procrustes orientation')
    for s in summary:
        rr = groups[tuple(s[k] for k in group_fields)]
        for field in c['evaluation']['fields']:
            vals = [number(r[field]) for r in rr]
            expected = math.fsum(vals) / 3 if all(v is not None for v in vals) else None
            got = number(s[field])
            check((got is None) == (expected is None) and (expected is None or abs(got - expected) <= 1e-9), 'Fixed 1/3 summary: ' + field)
    learning_keys = [tuple(r[k] for k in ('task', 'condition', 'split', 'n', 'action')) for r in learning]
    expected_learning = set(itertools.product(tasks, conditions, splits, ('6', '7', '8'), c['actions']))
    check(len(learning_keys) == len(set(learning_keys)) == 240 and set(learning_keys) == expected_learning, '240 source-learning cells')
    for r in learning:
        check(int(r['samples']) == c['data']['per_n'][r['split']] and r['EOS_evaluated'] == 'False', 'Learning sample/EOS coverage')
    for task, receipt in receipts.items():
        selected = [r for r in learning if r['task'] == task and r['condition'] == 'trained' and r['split'] == 'validation' and r['action'] == 'identity']
        mean = math.fsum(float(r['answer_accuracy']) for r in selected) / 3
        check(abs(mean - receipt['curve'][-1]['validation']['answer_accuracy']) <= 1e-12, 'Saved endpoint/cache learning agreement: ' + task)
    check(evaluation['per_n_rows'] == 720 and evaluation['summary_rows'] == evaluation['source_learning_rows'] == 240 and evaluation['cache_count'] == 16, 'Evaluation receipt coverage')
    statuses = Counter(r['status'] for r in reader['coverage'])
    check(len(reader['coverage']) == 728 and statuses.get('missing', 0) == 0 and statuses.get('coupled') == 4, 'Actual four-state coverage')
    verification_statuses = Counter(r['status'] for r in actual['checks'] + actual['input_kernel_checks'])
    check(verification_statuses.get('failed', 0) == 0 and len(actual['actual_cache_provenance']) == 8, 'Actual verification records/cache count')
    available_controls = [control for control, selector in verification['selectors'].items()
        if all(s['selected_indices'] is not None and len(s['selected_indices']) == 64 for s in selector.values())]
    unavailable_controls = [control for control in verification['selectors'] if control not in available_controls]
    check(len(actual['checks']) == 72 * len(available_controls) + len(unavailable_controls), 'Fixed actual verification check coverage')
    check(len(actual['input_kernel_checks']) == 5 * len(available_controls), 'Fixed actual input-isometry check coverage')
    indexed_hashes = {r['path']: r['metadata']['sha256'] for r in index['records']}
    for r in actual['actual_cache_provenance']:
        check(indexed_hashes.get(r['path']) == r['sha256'], 'Actual verification/cache binding: ' + r['path'])
    check(actual['fixed64_subset_is_numerical_verification_only'] and not actual['new_training'] and not actual['new_forward'], 'Independent check scope')
    metric_verification_coverage = {}
    for metric in metrics:
        selected = [r for r in actual['checks'] if r['check'].endswith('/' + metric)]
        metric_verification_coverage[metric] = dict(checks=len(selected), statuses=dict(Counter(r['status'] for r in selected)),
            non_numeric_checks=[dict(check=r['check'], status=r['status'],
                reason=r.get('reason', 'Both paths returned undefined under the frozen rank/variance rules'))
                for r in selected if r['status'] != 'passed'],
            unavailable_controls=[dict(control=r['check'], reason=r.get('reason')) for r in actual['checks'] if r['status'] == 'unavailable'])
    recorded = math.fsum(r['recorded_process_seconds'] for r in source_records)
    check(abs(recorded - completion['sum_recorded_training_process_seconds']) <= 1e-8, 'Recorded process-cost sum')
    check(completion['peak_cuda_memory_bytes'] == max(r['peak_cuda_memory_bytes'] for r in source_records), 'Maximum source PyTorch allocation peak')

    limitations = [
        'Four trained tasks, two transformations, one graph world, one shared true step0/start RNG; the four relations are coupled and are not four independent replications.',
        'Analysis is the fixed graph analysis split, with n=6,7,8 equally weighted; full specified transformation orbits were excluded across splits, arbitrary vertex isomorphism classes were not.',
        'The independent actual metric check uses only frozen reverse_max/n=6 first64 qualifying anchors in raw/answer strata, both conditions, hidden/output branches and three roles; it verifies numerics, not every720 endpoint independently.',
        'Undefined/rank-deficient/zero-variance cells remain undefined. Complete table coverage is not complete finite-metric coverage, and missing values are never imputed as zero.',
        'Hidden state is final_norm at graph-end before task; answer accuracy uses the full188-vocabulary argmax, output_prob is conditional on numeric0..30. Geometry extraction did not score EOS; saved source validation EOS uses teacher-conditioned answers.',
        'Input position-wise binary onehot Grams are exact isometries under R/C/W; they cannot distinguish correct and wrong in this fixed pilot and have no fabricated initialization increment.',
        'Graph peak_cuda_memory_bytes and Perm peak_cuda_allocated_bytes both derive from torch.cuda.max_memory_allocated(), not total-device usage or reserved-memory peak. They include live PyTorch allocations in the process, not isolated incremental model memory. Graph resets per invocation; Perm retains the maximum across resume invocations. These graph receipts disclose whether training was continuous.',
        'Recorded monotonic process wall includes validation/checkpoint IO and is not GPU-active time. CPU core-time, utilization-integrated GPU time and energy were not measured by this review.',
        'The preparation report_reader_audit had source_count=0 (SHA59947c6e0e8e864abbfe75ef920ab8ebec3230d5f29dd1504338efc7d8c5dae0); final conclusions use the actual four-source audit recorded here.',
        'The first graph evaluation stopped after16 extraction caches on a missing threadpoolctl dependency, before geometry. Root pinned threadpoolctl3.6.0 without changing existing package versions or scientific code/protocol, and resumed only the five fixed remaining stages. All4 checkpoints/16 cache hashes match the pre-repair snapshot; the failed17.262815s attempt and its recovered CPU/resource record are preserved separately rather than counted as success or erased.',
        'This reader verifies frozen code, receipts, cache bytes/shapes and saved arithmetic without new inference/metric calculation or old scientific-artifact edits. Existing actual audit supplies protected-old-file checks; this reader does not rehash the entire protected manifest.',
        'Cross-system results change size, encoding, tasks and distribution together; neither a domain-only causal effect nor a mathematical encoding mechanism follows from CKA alone.',
    ]
    result = dict(status='passed' if not problems else 'failed', reviewed_utc=datetime.now(timezone.utc).isoformat(),
        reader_source_sha256=sha(__file__), scientific_metrics_recomputed=0, new_training_updates=0,
        protocol_sha256=pd, main_source_version_sha256=version, verification_freeze_sha256=sha(DIAG / 'verification_freeze.json'),
        source_count=4, independent_worlds=1, shared_true_initial_count=1, relationship_instances=4, transform_types=2,
        logical_cache_count=16, geometry_length_rows=len(lengths), geometry_summary_rows=len(summary), source_learning_rows=len(learning),
        source_records=source_records, cache_records=cache_checks, actual_metric_failure_count=actual['failure_count'],
        actual_metric_check_status_counts=dict(verification_statuses), reader_coverage_status_counts=dict(statuses),
        reader_metric_coverage_status_counts=dict(Counter(r['status'] for r in reader['coverage'] if r['kind'] == 'metric')),
        defined_length_field_counts={field: sum(r[field] != '' for r in lengths) for field in c['evaluation']['fields']},
        defined_summary_field_counts={field: sum(r[field] != '' for r in summary) for field in c['evaluation']['fields']},
        actual_verification_available_controls=available_controls, actual_verification_unavailable_controls=unavailable_controls,
        actual_metric_verification_coverage=metric_verification_coverage,
        actual_verification_records=actual['checks'], actual_input_kernel_verification_records=actual['input_kernel_checks'],
        frozen_source_files_checked=len(source_bytes),
        same_shape_extraction_checks=len(manual_errors), maximum_manual_extraction_error=max(manual_errors),
        undefined_length_cells=sum(r['trained_correct'] == '' or r['trained_wrong'] == '' for r in lengths),
        recorded_training_process_seconds=recorded, peak_cuda_memory_bytes=completion['peak_cuda_memory_bytes'],
        evaluation_environment_repair=repair,
        preserved_pre_repair_checkpoint_count=len(preserved['checkpoints']), preserved_pre_repair_cache_count=len(preserved['cache']),
        failed_evaluation_attempt_seconds=repair['failed_phase']['wall_seconds'],
        failed_evaluation_attempt_cpu_resource=repair['failed_resource_recovered'],
        old_protected_files_reported_by_actual_reader=reader['old_protected_files'], artifact_sha256=artifacts,
        no_training=True, no_new_forward=True, no_new_scientific_metrics=True, no_frozen_science_or_existing_audit_modified=True,
        limitations=limitations, problems=problems)
    OUT.mkdir(exist_ok=True)
    (OUT / 'actual_graph_scope_review.json').write_text(json.dumps(result, indent=2) + '\n')
    text = ['Actual GraphWorld fixed-scope delivery review', '',
        'Status: ' + result['status'],
        'Actual source_count=4; four true20k checkpoints, one graph world and one shared true initialization.',
        'All16 task/condition/split caches and720 unique n-level/240 summary/240 learning rows checked; source initialization and checkpoint bytes are bound.',
        f'Actual independent metric failures={actual["failure_count"]}; statuses={dict(verification_statuses)}.',
        f'Fixed64 verification controls available={available_controls}; unavailable={unavailable_controls}. Unavailable is not a failed score and is not counted as a completed numerical check.',
        f'Four-state actual reader coverage={dict(statuses)}; undefined n-level cells={result["undefined_length_cells"]}.',
        f'Same-shape extraction checks={len(manual_errors)}, maximum absolute error={max(manual_errors):.12g}.',
        f'Source recorded monotonic process wall={recorded:.9f}s ({recorded/3600:.6f}h); PyTorch allocation peak={completion["peak_cuda_memory_bytes"]} bytes.',
        f'Original dependency-failed evaluation attempt={repair["failed_phase"]["wall_seconds"]:.9f}s, retained separately; recovered CPU/resource record={repair["failed_resource_recovered"]}.',
        '', 'Actual20k validation endpoints (full188 answer argmax; teacher-conditioned EOS):']
    for r in source_records:
        text.append(f'{r["task"]}: answer={r["validation_answer_accuracy"]:.9%}; EOS={r["validation_eos_accuracy"]:.9%}; samples={r["validation_samples"]}; elapsed={r["recorded_process_seconds"]:.6f}s; peak={r["peak_cuda_memory_bytes"]}bytes; invocation_start={r["invocation_start_step"]},updates={r["invocation_updates"]}.')
    text += ['', 'Per fixed metric actual numerical verification (unavailable/undefined are not numeric passes):']
    for metric, coverage in metric_verification_coverage.items():
        text.append(f'{metric}: checks={coverage["checks"]}; statuses={coverage["statuses"]}; unavailable_controls={coverage["unavailable_controls"]}.')
        for r in coverage['non_numeric_checks']:
            text.append(f'  {r["check"]}: {r["status"]}; {r["reason"]}.')
    text += ['', 'Scope and interpretation limits:'] + ['- ' + x for x in limitations]
    text += ['', 'Problems:'] + (['- ' + p for p in problems] if problems else ['None found within this read-only delivery scope.'])
    (OUT / 'actual_graph_scope_review.txt').write_text('\n'.join(text) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'source_count', 'logical_cache_count', 'geometry_length_rows', 'geometry_summary_rows', 'source_learning_rows', 'actual_metric_failure_count', 'reader_coverage_status_counts', 'recorded_training_process_seconds', 'peak_cuda_memory_bytes', 'problems')}))
    return bool(problems)


if __name__ == '__main__':
    raise SystemExit(main())
