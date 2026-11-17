"""Training-specificity controls with one immutable trained peaks readout.

Only two missing target-to-trained-source orthogonal maps are fitted. This
module never fits probes/scalers or constructs a neural model.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import svd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DIAG = ROOT / 'diagnostics/permworld_closing_audit_20261110/alignment'
OUT = Path('/mnt/yangx-projects-f/20261110/permworld_closing_audit/alignment')
PV = ROOT / 'diagnostics/permworld_orthogonal_alignment_20261110'
ED = ROOT / 'diagnostics/permworld_exceedance_alignment_20261110'
LENGTHS = tuple(range(10, 31))
TARGETS = ('trained_valleys', 'actual_saved_step0', 'trained_fixed_points')
CONTROLS = ('means_only', 'fit_identity__test_complement', 'fit_inverse__test_complement',
            *(f'fit_shuffle_{s}__test_complement' for s in range(1026, 1034)))
sys.path.insert(0, str(ROOT / 'experiments/permworld_peaks_probe_20261109/vendor'))
from analyze import read_length, example_auc  # original frozen orbit/label policy


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csv_write(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def loaded(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def source_plan():
    """Metadata/identity audit before outcomes; actual used inputs are bound."""
    prior, edaudit, pvaudit = read(PV / 'protocol.json'), read(ED / 'cache_audit.json'), read(PV / 'cache_audit.json')
    plan, inputs, missing = [], {}, []
    def bind(path, expected=None, role='input'):
        path = Path(path)
        if not path.is_file():
            missing.append(str(path)); return None
        actual = sha(path)
        if expected is not None and expected != actual:
            raise ValueError(('Source SHA mismatch', str(path), expected, actual))
        record = dict(path=str(path), sha256=actual, size_bytes=path.stat().st_size, role=role)
        if str(path) in inputs:
            assert inputs[str(path)]['sha256'] == actual
        else:
            inputs[str(path)] = record
        return record
    for path in [PV/'protocol.json', PV/'cache_audit.json', ED/'cache_audit.json',
                 ROOT/'experiments/permworld_peaks_probe_20261109/vendor/analyze.py',
                 ROOT/'experiments/permworld_orthogonal_alignment_20261110/vendor/alignment.py',
                 Path(__file__), DIAG/'spec.json']:
        bind(path, role='code_or_protocol')
    for old in prior['units']:
        name = old['unit']; eu = next(row for row in edaudit['units'] if row['unit'] == name)
        receipt_path = ED/'results'/name/'pv_score_rebuild.json'
        receipt = read(receipt_path); bind(receipt_path, role='saved_trained_probe_score_receipt')
        score = receipt['score_files']['one_line']
        entry = dict(unit=name, historical=old['historical'], data=bind(old['data_path'], old['data_sha256'], 'data'),
            readouts=bind(score['parameters_path'], score['parameters_sha256'], 'exact_saved_trained_peaks_probe_scaler_and_maps'),
            scores=bind(score['path'], score['sha256'], 'old_trained_valleys_scores_and_controls'), standardized={}, fixed_points={})
        for task in ('peaks', 'valleys', 'initial'):
            source = next(row for row in pvaudit['standardized_features']
                          if row['unit'] == name and row['encoding'] == 'one_line' and row['name'] == task)
            entry['standardized'][task] = dict(**bind(source['path'], source['sha256'], f'raw_FP32_schema_{task}'),
                checkpoint_sha256=source['checkpoint_sha256'], checkpoint_path=source['checkpoint_path'])
            bind(source['checkpoint_path'], source['checkpoint_sha256'], f'actual_{task}_checkpoint')
        additions = Path(prior['output'])/'validation'/name/'one_line-initial.npz'
        adaptation_path = additions.parent/'adaptation.json'; adaptation = read(adaptation_path)
        bind(adaptation_path, role='initial_target_validation_adaptation_receipt')
        adapted = next(row for row in adaptation['files'] if row['output_path'] == str(additions))
        entry['initial_complement_validation'] = bind(additions, adapted['output_sha256'], 'actual_saved_step0_complement_validation')
        assert adapted['checkpoint_sha256'] == entry['standardized']['initial']['checkpoint_sha256']
        assert adapted['data_sha256'] == old['data_sha256']
        for split in ('validation', 'test'):
            source = next(row for row in eu['raw_features'] if row['encoding'] == 'one_line' and
                          row['task'] == 'fixed_points' and row['phase'] == 'trained' and row['split'] == split)
            entry['fixed_points'][split] = {**source, **bind(source['path'], source['sha256'], f'fixed_points_{split}')}
            bind(source['checkpoint_path'], source['checkpoint_sha256'], 'actual_fixed_points_checkpoint')
            if source.get('metadata_path'):
                bind(source['metadata_path'], source.get('metadata_sha256'), 'fixed_points_binding_metadata')
        assert eu['data_sha256'] == old['data_sha256']
        plan.append(entry)
    result = dict(status='passed' if not missing else 'missing_inputs', created_utc=datetime.now(timezone.utc).isoformat(),
        units=plan, inputs=inputs, missing=missing, no_fits=True, no_test_scoring=True,
        source_bug='Old phase=initial fits the initial source probe. That is not the trained-source step0-target control.',
        initialization_identity='True saved shared architecture step0 checkpoint. All tasks share pre-task prefix representation; cache initial source name does not imply use of an initial source probe.')
    write(DIAG/'source_plan.json', result)
    return result


def checked_load(record):
    assert sha(record['path']) == record['sha256'], record['path']
    return loaded(record['path'])


def fixed_array(record):
    assert sha(record['path']) == record['sha256']
    if record.get('field'):
        with np.load(record['path'], allow_pickle=False) as z:
            result = z[record['field']].copy()
    else:
        result = np.load(record['path'], mmap_mode='r', allow_pickle=False)
    assert result.shape == tuple(record['shape']) and result.dtype == np.float32
    return result


def frozen_check():
    freeze = read(DIAG/'freeze.json')
    assert sha(DIAG/'spec.json') == freeze['spec_sha256']
    assert sha(DIAG/'source_plan.json') == freeze['source_plan_sha256']
    assert sha(__file__) == freeze['source_sha256']
    plan = read(DIAG/'source_plan.json')
    assert plan['status'] == 'passed'
    for source in plan['inputs'].values():
        assert sha(source['path']) == source['sha256'], source['path']
    return plan, freeze


def probe_from_saved(parameters, n, values):
    p = f'peaks_n{n}_'
    values = np.asarray(values)
    # Mapped outputs are FP64, as in the original frozen pipeline.
    assert values.dtype == np.float64
    scaled = ((values - parameters[p+'mean']) / parameters[p+'scale']).astype(np.float64)
    return scaled @ parameters[p+'coef'].T + parameters[p+'intercept']


def finite_mean(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    return float(values[finite].mean()) if finite.any() else np.nan


def run_unit(entry):
    unit = entry['unit']; dest = OUT/unit; dest.mkdir(parents=True, exist_ok=True)
    local = DIAG/'results'/unit; local.mkdir(parents=True, exist_ok=True)
    done = local/'completion.json'
    if done.exists():
        previous = read(done)
        assert previous['spec_sha256'] == sha(DIAG/'spec.json')
        for path, expected in previous['outputs'].items():
            assert sha(path) == expected
        return previous
    data = checked_load(entry['data'])
    params = checked_load(entry['readouts']); saved = checked_load(entry['scores'])
    source = checked_load(entry['standardized']['peaks'])
    valleys = checked_load(entry['standardized']['valleys'])
    initial = checked_load(entry['standardized']['initial'])
    init_val = checked_load(entry['initial_complement_validation'])
    fixed = {split: fixed_array(entry['fixed_points'][split]) for split in ('validation', 'test')}
    count = len(data['test_lengths'])
    assert count == 10_752
    for task in [source, valleys, initial]:
        for split in ('validation', 'test'):
            assert np.array_equal(task[split+'_indices'], np.arange(len(data[split+'_lengths'])))
    assert np.array_equal(init_val['validation_indices'], initial['validation_indices'])
    assert str(init_val['checkpoint_sha256']) == str(initial['checkpoint_sha256'])
    assert str(source['checkpoint_sha256']) == str(saved['source_checkpoint_sha256'])
    assert str(params['data_sha256']) == entry['data']['sha256'] == str(saved['data_sha256'])
    assert str(params['source_checkpoint_sha256']) == str(source['checkpoint_sha256'])
    assert str(saved['parameters_sha256']) == entry['readouts']['sha256']
    assert np.array_equal(saved['anchor_indices'], np.arange(count))
    lengths = data['test_lengths']
    assert np.array_equal(saved['lengths'], lengths)
    splits = {n: read_length(data, n) for n in LENGTHS}
    source_means, target_means = {}, {task:{} for task in TARGETS}
    cross = {task:np.zeros((256,256), np.float64) for task in TARGETS[1:]}
    fit_records, checks = [], []
    for n in LENGTHS:
        fit, dev, eligible, matched, _, _, _, _ = splits[n]
        val_ids = np.flatnonzero(data['validation_lengths'] == n)
        test_ids = np.flatnonzero(lengths == n)
        assert np.array_equal(saved['eligible'][test_ids], eligible)
        assert np.array_equal(saved['matched'][test_ids], matched)
        assert np.array_equal(params[f'peaks_n{n}_source_fit_indices'], val_ids[fit])
        assert np.array_equal(params[f'peaks_n{n}_source_tuning_indices'], val_ids[dev])
        x = source['validation_0_hidden'][val_ids]
        mx = x[fit].astype(np.float64).mean(0)
        assert np.array_equal(mx, params[f'n{n}_mu_peaks'])
        source_means[n] = mx
        target_means['trained_valleys'][n] = params[f'n{n}_mu_valleys_action1']
        cx = x[fit] - mx
        for task, array in [('actual_saved_step0', init_val['validation_1_hidden'][val_ids]),
                            ('trained_fixed_points', fixed['validation'][val_ids,1])]:
            my = array[fit].astype(np.float64).mean(0)
            target_means[task][n] = my
            cross[task] += (array[fit]-my).T @ cx / int(fit.sum())
        fit_records.append(dict(unit=unit, length=n, fit_anchors=int(fit.sum()), dev_anchors=int(dev.sum()),
            source_probe_fit_indices_sha256=array_sha(val_ids[fit]),
            all_targets_fit_indices_equal=True, source_probe_scaler_archive_sha256=entry['readouts']['sha256']))
    rotations = {'trained_valleys':params['R_complement']}
    for task in TARGETS[1:]:
        left, _, right = svd(cross[task], full_matrices=False)
        rotations[task] = left @ right
    orthogonality = {task:float(np.max(np.abs(r.T@r-np.eye(256)))) for task,r in rotations.items()}
    assert max(orthogonality.values()) <= 2e-12
    padded = np.full((len(TARGETS), count,28), np.nan, np.float64)
    auc = np.full((len(TARGETS),count), np.nan, np.float64)
    labels = saved['labels'].copy()
    assert labels.shape == (count,28)
    old_method_names = list(saved['method_names'])
    correct_index = old_method_names.index('fit_complement__test_complement')
    max_score_error = 0.
    records, sample_rows = [], []
    target_test = {'trained_valleys':valleys['test_1_hidden'],
                   'actual_saved_step0':initial['test_1_hidden'],
                   'trained_fixed_points':fixed['test'][:,1]}
    for n in LENGTHS:
        ids = np.flatnonzero(lengths == n)
        peak = splits[n][6]
        assert np.array_equal(labels[ids,:n-2], peak)
        for ti, task in enumerate(TARGETS):
            mapped = ((target_test[task][ids] - target_means[task][n]) @ rotations[task]
                      + source_means[n])
            scores = probe_from_saved(params,n,mapped)
            if task == 'trained_valleys':
                old = saved['rawscore'][correct_index,ids,:n-2]
                error = float(np.max(np.abs(scores-old)))
                max_score_error = max(max_score_error,error)
                assert error <= 1e-12, (unit,n,'trained route failed reproduction',error)
                # Retain the original per-example scores byte-for-byte after verification.
                scores = old.copy()
            values = example_auc(scores,peak)
            if task == 'trained_valleys':
                np.testing.assert_allclose(values,saved['per_sample_auroc'][correct_index,ids],atol=1e-12,rtol=0,equal_nan=True)
            padded[ti,ids,:n-2] = scores; auc[ti,ids] = values
            for support in ['full_test','count_matched']:
                keep = saved['eligible' if support=='full_test' else 'matched'][ids]
                selected = values[keep]
                records.append(dict(unit=unit,historical_reference=entry['historical'],target=task,length=n,
                    support=support,n=int(keep.sum()),finite=int(np.isfinite(selected).sum()),
                    undefined=int((~np.isfinite(selected)).sum()),auroc=finite_mean(selected)))
        # Separate exact existing mapping controls, with the same trained source probe.
        for method in CONTROLS:
            mi = old_method_names.index(method)
            values = saved['per_sample_auroc'][mi,ids]
            for support in ['full_test','count_matched']:
                keep = saved['eligible' if support=='full_test' else 'matched'][ids]
                selected = values[keep]
                records.append(dict(unit=unit,historical_reference=entry['historical'],target=method,length=n,
                    support=support,n=int(keep.sum()),finite=int(np.isfinite(selected).sum()),
                    undefined=int((~np.isfinite(selected)).sum()),auroc=finite_mean(selected)))
        for j,global_id in enumerate(ids):
            for ti, task in enumerate(TARGETS):
                sample_rows.append(dict(unit=unit,anchor_index=int(global_id),sample_id=str(saved['sample_ids'][global_id]),
                    length=n,target=task,full_test=bool(saved['eligible'][global_id]),
                    count_matched=bool(saved['matched'][global_id]),defined=bool(np.isfinite(auc[ti,global_id])),
                    undefined_reason='' if np.isfinite(auc[ti,global_id]) else 'no_positive_or_no_negative_position',
                    auroc=float(auc[ti,global_id])))
    parameters_path = dest/'maps_and_means.npz'
    np.savez_compressed(parameters_path,
        target_names=np.asarray(TARGETS), lengths=np.asarray(LENGTHS),
        R=np.stack([rotations[task] for task in TARGETS]),
        source_means=np.stack([source_means[n] for n in LENGTHS]),
        target_means=np.stack([[target_means[task][n] for n in LENGTHS] for task in TARGETS]),
        source_probe_scaler_archive_sha256=np.asarray(entry['readouts']['sha256']),
        source_checkpoint_sha256=source['checkpoint_sha256'],data_sha256=np.asarray(entry['data']['sha256']))
    scores_path = dest/'test_scores.npz'
    np.savez_compressed(scores_path,rawscore=padded,per_sample_auroc=auc,labels=labels,
        target_names=np.asarray(TARGETS),lengths=lengths,anchor_indices=saved['anchor_indices'],
        sample_ids=saved['sample_ids'],eligible=saved['eligible'],matched=saved['matched'],
        source_probe_scaler_archive_sha256=np.asarray(entry['readouts']['sha256']),
        source_checkpoint_sha256=source['checkpoint_sha256'],data_sha256=np.asarray(entry['data']['sha256']))
    csv_write(local/'by_length.csv',records); csv_write(local/'per_sample.csv',sample_rows)
    csv_write(local/'fit_audit.csv',fit_records)
    output_paths=[parameters_path,scores_path,local/'by_length.csv',local/'per_sample.csv',local/'fit_audit.csv']
    result=dict(status='completed',unit=unit,historical_reference=entry['historical'],
        spec_sha256=sha(DIAG/'spec.json'),source_plan_sha256=sha(DIAG/'source_plan.json'),
        source_probe_and_scaler_sha256=entry['readouts']['sha256'],same_trained_probe_all_three_targets=True,
        no_probe_fits=True,no_neural_forward=True,no_base_training=True,
        new_maps_only=list(TARGETS[1:]),reused_trained_valleys_map=True,
        maximum_trained_route_score_error=max_score_error,maximum_orthogonality_error=orthogonality,
        outputs={str(path):sha(path) for path in output_paths},created_utc=datetime.now(timezone.utc).isoformat())
    write(done,result)
    print(json.dumps({key:result[key] for key in ['unit','status','maximum_trained_route_score_error']}),flush=True)
    return result


def summarize(plan):
    by_unit, contrasts, controls = [], [], []
    for entry in plan['units']:
        name=entry['unit']
        with (DIAG/'results'/name/'by_length.csv').open() as handle:
            records=list(csv.DictReader(handle))
        indexed={(r['target'],r['support'],int(r['length'])):r for r in records}
        for support in ['full_test','count_matched']:
            values={}
            for task in (*TARGETS,*CONTROLS):
                rr=[indexed[task,support,n] for n in LENGTHS]
                vv=np.asarray([float(r['auroc']) for r in rr])
                value=float(vv.sum()/21) if np.isfinite(vv).all() else np.nan
                values[task]=value
                row=dict(unit=name,historical_reference=entry['historical'],support=support,target=task,
                    auroc=value,finite_lengths=int(np.isfinite(vv).sum()),
                    anchors=sum(int(r['n']) for r in rr),finite=sum(int(r['finite']) for r in rr),
                    undefined=sum(int(r['undefined']) for r in rr))
                (by_unit if task in TARGETS else controls).append(row)
            best=max(CONTROLS,key=lambda task:values[task]) if all(np.isfinite(values[k]) for k in CONTROLS) else None
            for contrast,baseline in [('valleys_minus_step0','actual_saved_step0'),
                                      ('valleys_minus_fixed_points','trained_fixed_points')]:
                contrasts.append(dict(unit=name,historical_reference=entry['historical'],support=support,
                    contrast=contrast,control_method=baseline,difference=values['trained_valleys']-values[baseline],
                    trained_valleys_auroc=values['trained_valleys'],control_auroc=values[baseline]))
            contrasts.append(dict(unit=name,historical_reference=entry['historical'],support=support,
                contrast='valleys_minus_best_common_mapping_control',
                control_method=best or '',
                difference=values['trained_valleys']-values[best] if best else np.nan,
                trained_valleys_auroc=values['trained_valleys'],control_auroc=values[best] if best else np.nan))
            controls.append(dict(unit=name,historical_reference=entry['historical'],support=support,
                target='best_common_mapping_control',auroc=values[best] if best else np.nan,
                finite_lengths=21 if best else 0,anchors='',finite='',undefined=''))
    csv_write(DIAG/'by_unit.csv',by_unit); csv_write(DIAG/'contrasts_by_unit.csv',contrasts)
    csv_write(DIAG/'common_mapping_controls_by_unit.csv',controls)
    stats=[]
    for name,rows,key,valuefield in [('route',by_unit,'target','auroc'),('contrast',contrasts,'contrast','difference')]:
        for support in ['full_test','count_matched']:
            for condition in sorted(set(r[key] for r in rows)):
                selected=[r for r in rows if not r['historical_reference'] and r['support']==support and r[key]==condition]
                assert len(selected)==4
                values=np.array([r[valuefield] for r in selected],float)
                finite=np.isfinite(values)
                record=dict(kind=name,support=support,condition=condition,registered_units=4,finite_units=int(finite.sum()),
                    mean=float(values.mean()) if finite.all() else np.nan,
                    sample_sd=float(values.std(ddof=1)) if finite.all() else np.nan,
                    minimum=float(values.min()) if finite.all() else np.nan,
                    maximum=float(values.max()) if finite.all() else np.nan,
                    positive_units=int((values>0).sum()),negative_units=int((values<0).sum()),
                    raw_unit_values=json.dumps({r['unit']:float(r[valuefield]) for r in selected}))
                stats.append(record)
    csv_write(DIAG/'four_unit_summary.csv',stats)
    write(DIAG/'completion.json',dict(status='completed',units=[r['unit'] for r in plan['units']],
        formal_units=['unit02','unit03','unit04','unit05'],historical_reference='archdev001',
        spec_sha256=sha(DIAG/'spec.json'),source_plan_sha256=sha(DIAG/'source_plan.json'),
        no_probe_fitting=True,no_base_training=True,no_neural_forward=True,
        data_support_and_probe_unchanged=True,
        output_tables={str(path):sha(path) for path in [DIAG/'by_unit.csv',DIAG/'contrasts_by_unit.csv',
            DIAG/'common_mapping_controls_by_unit.csv',DIAG/'four_unit_summary.csv']}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','freeze','run'])
    args=parser.parse_args()
    if args.stage=='prepare':
        source_plan();return
    if args.stage=='freeze':
        assert read(DIAG/'source_plan.json')['status']=='passed'
        value=dict(created_utc=datetime.now(timezone.utc).isoformat(),spec_sha256=sha(DIAG/'spec.json'),
            source_plan_sha256=sha(DIAG/'source_plan.json'),source_sha256=sha(__file__),before_new_control_scoring=True)
        if (DIAG/'freeze.json').exists():
            old=read(DIAG/'freeze.json')
            assert all(old[k]==value[k] for k in ['spec_sha256','source_plan_sha256','source_sha256'])
        else:write(DIAG/'freeze.json',value)
        return
    plan,_=frozen_check()
    with threadpool_limits(limits=2):
        for entry in plan['units']:
            run_unit(entry)
        summarize(plan)


if __name__=='__main__':main()
