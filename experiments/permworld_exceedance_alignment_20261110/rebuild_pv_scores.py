"""Reproduce prior trained peaks/valleys scores with their saved rotations.

The old source-only ridge fitter and orbit split are reused unchanged. No
orthogonal map is fitted here. Every trained vendor result, including the
native-input auxiliary rows, must reproduce before per-position scores are
exported to this experiment's new output directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from .shared import (ROOT, DIAG, OUT, PV, ENCODINGS, LENGTHS, SHUFFLE_SEEDS,
                     array_sha, read, read_csv, sha, vendor, write, write_csv)

METHODS = ('direct', 'means_only', 'within_peaks', 'within_valleys',
           'oracle_count_position_prior',
           *(f'fit_{action}__test_complement' for action in ('identity', 'complement', 'inverse')),
           *(f'fit_shuffle_{seed}__test_complement' for seed in SHUFFLE_SEEDS))
TOLERANCE = 1e-12


def frozen_config():
    """Guard actual data access/fitting behind the parent's frozen protocol."""
    config = read(DIAG / 'protocol.json')
    freeze = read(DIAG / 'freeze.json')
    assert sha(DIAG / 'protocol.json') == freeze['protocol_sha256']
    own_path = str(Path(__file__).resolve().relative_to(ROOT))
    assert own_path in freeze['source_sha256'], 'PV exporter must be included before protocol freeze'
    for path, digest in freeze['source_sha256'].items():
        assert sha(ROOT / path) == digest, ('Frozen source changed', path)
    assert Path(config['output']) == OUT
    return config, freeze


def load_bound(path, digest, bindings, *, role):
    path = Path(path)
    actual = sha(path)
    assert actual == digest, ('Source hash mismatch', path)
    bindings[str(path)] = dict(sha256=actual, role=role)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez(temporary, **arrays)
    temporary.replace(path)


def probe_parameters(probe, *, task, n, alpha, fit_indices, dev_indices, parameters):
    """Serialize the unchanged fitter's selected model and scaler, not a refit."""
    closure = inspect.getclosurevars(probe).nonlocals
    selected, scaler = closure['best_model'], closure['scaling']
    prefix = f'{task}_n{n}_'
    values = dict(coef=selected.coef_, intercept=selected.intercept_,
                  mean=scaler.mean_, scale=scaler.scale_, alpha=np.asarray(alpha),
                  source_fit_indices=fit_indices, source_tuning_indices=dev_indices)
    hashes = {}
    for name, value in values.items():
        value = np.asarray(value).copy()
        parameters[prefix + name] = value
        hashes[name] = array_sha(value)
    return hashes


def _pack_length(scores, labels, ids, n, all_scores, all_labels, all_auc, example_auc):
    """Only padding/adaptation; position scores remain raw FP64 outputs."""
    assert labels.shape == (len(ids), n - 2)
    all_labels[ids, :n - 2] = labels
    for mi, name in enumerate(METHODS):
        values, truth = scores[name]
        assert values.shape == labels.shape and np.array_equal(truth, labels)
        all_scores[mi, ids, :n - 2] = values
        all_auc[mi, ids] = example_auc(values, truth)


def _scalar_equal(actual, expected):
    actual, expected = float(actual), float(expected)
    if np.isnan(actual) or np.isnan(expected):
        return bool(np.isnan(actual) and np.isnan(expected)), 0.
    difference = abs(actual - expected)
    return difference <= TOLERANCE, difference


def run_unit(unit_audit):
    """Rebuild one unit's trained PV scores; returns paths and strict checks."""
    config, freeze = frozen_config()
    unit = unit_audit['unit']
    registered = next(r for r in config['units'] if r['unit'] == unit)
    assert registered['data_sha256'] == unit_audit['data_sha256']
    assert registered['data_path'] == unit_audit['data_path']
    started = time.monotonic()
    destination = OUT / 'pv_scores' / unit
    receipt_path = DIAG / 'results' / unit / 'pv_score_rebuild.json'
    if receipt_path.exists():
        previous = read(receipt_path)
        assert previous['status'] == 'completed' and previous['protocol_sha256'] == freeze['protocol_sha256']
        assert previous['data_sha256'] == unit_audit['data_sha256']
        for desc in previous['score_files'].values():
            assert sha(desc['path']) == desc['sha256']
            assert sha(desc['parameters_path']) == desc['parameters_sha256']
        assert sha(previous['reproduced_by_length_path']) == previous['reproduced_by_length_sha256']
        for path, desc in previous['source_bindings'].items():
            assert sha(path) == desc['sha256']
        return previous

    fit_probe, read_length, example_auc, average, position_prior, _, _ = vendor()
    # This function only evaluates the original expressions. It does not call
    # inputs_by_length(), rotation(), or any alignment fit.
    from alignment import scores_for_length
    bindings = {}
    assert sha(unit_audit['data_path']) == unit_audit['data_sha256']
    bindings[unit_audit['data_path']] = dict(sha256=unit_audit['data_sha256'], role='unit_data')
    with np.load(unit_audit['data_path'], allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files if key.endswith(('_lengths', '_permutations'))}
    lengths = data['test_lengths']
    count = len(lengths)
    assert count == 10752 and set(lengths) == set(LENGTHS)
    splits = {n: read_length(data, n) for n in LENGTHS}
    prior_audit_path = PV / 'cache_audit.json'
    prior_protocol_path = PV / 'protocol.json'
    prior_audit, prior_protocol = read(prior_audit_path), read(prior_protocol_path)
    for path, role in [(prior_audit_path, 'prior_PV_cache_audit'), (prior_protocol_path, 'prior_PV_protocol')]:
        bindings[str(path)] = dict(sha256=sha(path), role=role)
    old_unit = next(r for r in prior_protocol['units'] if r['unit'] == unit)
    assert old_unit['data_sha256'] == unit_audit['data_sha256'] and old_unit['data_path'] == unit_audit['data_path']
    previous_table = PV / 'results' / unit / 'by_length.csv'
    bindings[str(previous_table)] = dict(sha256=sha(previous_table), role='previous_trained_PV_AUROC_table')
    previous_rows = [row for row in read_csv(previous_table) if row['phase'] == 'trained']
    previous = {(row['encoding'], int(row['length']), row['method'], row['support']): row for row in previous_rows}
    assert len(previous) == len(previous_rows) == 1848

    map_prefix = Path(prior_protocol['output']) / 'map_audits' / unit / 'alignment'
    map_metadata_path, map_archive_path = map_prefix.with_suffix('.json'), map_prefix.with_suffix('.npz')
    metadata = read(map_metadata_path)
    assert metadata['status'] == 'completed' and metadata['unit'] == unit
    bindings[str(map_metadata_path)] = dict(sha256=sha(map_metadata_path), role='saved_PV_map_descriptors')
    maps_archive = load_bound(map_archive_path, metadata['parameter_archive_sha256'], bindings, role='saved_PV_rotations_no_refit')
    validation_folder = Path(prior_protocol['output']) / 'validation' / unit
    adaptation_path = validation_folder / 'adaptation.json'
    adaptation = read(adaptation_path)
    bindings[str(adaptation_path)] = dict(sha256=sha(adaptation_path), role='prior_PV_validation_adaptation')
    detailed, checks, score_files, parameter_records = [], [], {}, []
    pending_exports = []
    maximum_error = 0.

    for encoding in ENCODINGS:
        selected = {}
        for task in ('peaks', 'valleys'):
            desc = next(row for row in prior_audit['standardized_features']
                        if row['unit'] == unit and row['encoding'] == encoding and row['name'] == task)
            selected[task] = load_bound(desc['path'], desc['sha256'], bindings, role='prior_PV_raw_coordinate_schema_export')
            assert str(selected[task]['checkpoint_sha256']) == desc['checkpoint_sha256']
            # Check the named actual source weights without loading a neural module.
            assert sha(desc['checkpoint_path']) == desc['checkpoint_sha256']
            bindings[desc['checkpoint_path']] = dict(sha256=desc['checkpoint_sha256'], role=f'trained_{task}_checkpoint')
            for split in ('validation', 'test'):
                assert np.array_equal(selected[task][split + '_indices'], np.arange(len(data[split + '_lengths'])))
        source, target = selected['peaks'], selected['valleys']
        additions_path = validation_folder / f'{encoding}-trained.npz'
        adapted = next(row for row in adaptation['files'] if row['output_path'] == str(additions_path))
        additions = load_bound(additions_path, adapted['output_sha256'], bindings, role='prior_PV_complement_inverse_validation')
        assert str(additions['checkpoint_sha256']) == str(target['checkpoint_sha256'])
        assert np.array_equal(additions['validation_indices'], source['validation_indices'])
        assert adapted['data_sha256'] == unit_audit['data_sha256']
        assert adapted['checkpoint_sha256'] == str(target['checkpoint_sha256'])

        rotation_records = [row for row in metadata['maps']
                            if row.get('encoding') == encoding and row.get('phase') == 'trained']
        expected_conditions = ['identity', 'complement', 'inverse', *(f'shuffle_{seed}' for seed in SHUFFLE_SEEDS)]
        assert [row['fit_condition'] for row in rotation_records] == expected_conditions
        transforms, parameters = {}, {}
        for row in rotation_records:
            key = f'R_{row["call"]:03d}'
            value = maps_archive[key]
            assert value.shape == (256, 256) and value.dtype == np.float64
            assert array_sha(value) == row['map_sha256']
            assert np.max(np.abs(value.T @ value - np.eye(256))) <= 2e-12
            transforms[row['fit_condition']] = value
            parameters['R_' + row['fit_condition']] = value.copy()
        maps = [transforms[name] for name in expected_conditions[:3]]
        shuffles = [transforms[f'shuffle_{seed}'] for seed in SHUFFLE_SEEDS]
        scores_all = np.full((len(METHODS), count, 28), np.nan, np.float64)
        auc_all = np.full((len(METHODS), count), np.nan, np.float64)
        labels_all = np.zeros((count, 28), bool)
        eligible_all, matched_all = np.zeros(count, bool), np.zeros(count, bool)

        for n in LENGTHS:
            fit, dev, eligible, matched, peak_fit, valley_fit, peak, valley_masks = splits[n]
            val_ids = np.flatnonzero(data['validation_lengths'] == n)
            ids = np.flatnonzero(lengths == n)
            assert len(val_ids) == 256 and len(ids) == 512
            x = source['validation_0_hidden'][val_ids]
            y = [target['validation_0_hidden'][val_ids],
                 additions['validation_1_hidden'][val_ids], additions['validation_2_hidden'][val_ids]]
            probe, alpha = fit_probe(x, peak_fit, fit, dev)
            own, own_alpha = fit_probe(y[0], valley_fit, fit, dev)
            source_hashes = probe_parameters(probe, task='peaks', n=n, alpha=alpha,
                fit_indices=val_ids[fit], dev_indices=val_ids[dev], parameters=parameters)
            target_hashes = probe_parameters(own, task='valleys', n=n, alpha=own_alpha,
                fit_indices=val_ids[fit], dev_indices=val_ids[dev], parameters=parameters)
            mu_x = x[fit].astype(np.float64).mean(axis=0)
            mu_y = [value[fit].astype(np.float64).mean(axis=0) for value in y]
            parameters[f'n{n}_mu_peaks'] = mu_x
            for action, mean in enumerate(mu_y):
                parameters[f'n{n}_mu_valleys_action{action}'] = mean
            parameter_records.append(dict(encoding=encoding, length=n,
                source_alpha=float(alpha), target_alpha=float(own_alpha),
                source_probe_parameter_sha256=source_hashes, target_probe_parameter_sha256=target_hashes,
                fit_anchors=int(fit.sum()), tuning_anchors=int(dev.sum()),
                validation_fit_indices_sha256=array_sha(val_ids[fit]), validation_tuning_indices_sha256=array_sha(val_ids[dev]),
                mean_policy='Original probe-fit rows, per-length float64 means; all complement test mappings use complement target mean.'))
            row = dict(length=n, eligible=eligible, matched=matched, peak=peak,
                       valley_masks=valley_masks, probe=probe, own=own,
                       mu_x=mu_x, mu_y=mu_y, peak_fit=peak_fit[fit],
                       xt=source['test_0_hidden'][ids],
                       yt=[target[f'test_{action}_hidden'][ids] for action in range(3)])
            results = scores_for_length(row, maps, shuffles)
            assert len(results) == 22
            for method, (values, truth) in results.items():
                auc = example_auc(values, truth)
                for support, subset in [('eligible', eligible), ('matched', matched)]:
                    old_row = previous[encoding, n, method, support]
                    actual = average(auc[subset])
                    good, difference = _scalar_equal(actual, old_row['auc'])
                    maximum_error = max(maximum_error, difference)
                    finite = int(np.isfinite(auc[subset]).sum())
                    assert int(old_row['n']) == int(subset.sum()) and int(old_row['finite']) == finite, ('Prior support differs', unit, encoding, n, method, support)
                    assert good, ('Prior PV AUROC did not reproduce; no score export', unit, encoding, n, method, support, actual, old_row['auc'], difference)
                    checks.append(dict(encoding=encoding, length=n, method=method, support=support,
                                       previous=float(old_row['auc']), actual=float(actual), absolute_error=difference))
                    detailed.append(dict(unit=unit, encoding=encoding, phase='trained', length=n,
                        support=support, method=method, n=int(subset.sum()), finite=finite, auc=float(actual)))
            # The same common peak labels apply to all exported reference/routes.
            assert np.array_equal(peak, valley_masks[1])
            _pack_length(results, peak, ids, n, scores_all, labels_all, auc_all, example_auc)
            eligible_all[ids], matched_all[ids] = eligible, matched

        assert np.all(matched_all <= eligible_all)
        for n in LENGTHS:
            ids = np.flatnonzero(lengths == n)
            assert np.isnan(scores_all[:, ids, n-2:]).all()
            assert np.isfinite(scores_all[:, ids, :n-2]).all()
        path = destination / f'{encoding}-scores.npz'
        param_path = destination / f'{encoding}-readouts.npz'
        parameters['data_sha256'] = np.asarray(unit_audit['data_sha256'])
        parameters['source_checkpoint_sha256'] = source['checkpoint_sha256']
        parameters['target_checkpoint_sha256'] = target['checkpoint_sha256']
        parameters['map_archive_sha256'] = np.asarray(metadata['parameter_archive_sha256'])
        parameters['probe_transform_policy'] = np.asarray('StandardScaler.transform preserves input dtype, then cast tofloat64, selected Ridge.predict. Original FP32 inputs differ from FP64 mapped inputs; serialized mean/scale must be applied with that same dtype convention.')
        score_arrays = dict(scores=scores_all, rawscore=scores_all,
                 per_sample_auroc=auc_all, labels=labels_all, lengths=lengths,
                 anchor_indices=np.arange(count, dtype=np.int64),
                 sample_ids=np.asarray([f'{unit}:test:{i}' for i in range(count)]),
                 eligible=eligible_all, matched=matched_all,
                 method_names=np.asarray(METHODS), unit=np.asarray(unit), encoding=np.asarray(encoding),
                 phase=np.asarray('trained'), pair=np.asarray('peaks_valleys'),
                 position_layout=np.asarray('Interior positions i=2..n-1 stored at0..n-3; NaN score padding to28.'),
                 data_sha256=np.asarray(unit_audit['data_sha256']),
                 source_checkpoint_sha256=source['checkpoint_sha256'], target_checkpoint_sha256=target['checkpoint_sha256'],
                 map_archive_sha256=np.asarray(metadata['parameter_archive_sha256']),
                 source_test_action=np.asarray(0), common_target_test_action=np.asarray(1),
                 target_labels_equal_source=np.asarray(True),
                 test_permutations_sha256=np.asarray(array_sha(data['test_permutations'])))
        pending_exports.append((encoding, path, param_path, parameters, score_arrays))

    # Do not export either encoding before every trained vendor cell for this
    # complete unit has passed. Partial technical files are not success receipts.
    assert len(checks) == len(previous) == 1848
    for encoding, path, param_path, parameters, score_arrays in pending_exports:
        save_npz(param_path, **parameters)
        score_arrays['parameters_sha256'] = np.asarray(sha(param_path))
        save_npz(path, **score_arrays)
        score_files[encoding] = dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size,
            parameters_path=str(param_path), parameters_sha256=sha(param_path),
            shape=list(score_arrays['scores'].shape), methods=len(METHODS), position_label_layout='interior_n_minus2_padded28')

    reproduced_path = DIAG / 'results' / unit / 'pv_score_rebuild_by_length.csv'
    write_csv(reproduced_path, detailed)
    result = dict(status='completed', unit=unit, protocol_sha256=freeze['protocol_sha256'],
        data_sha256=unit_audit['data_sha256'], score_files=score_files,
        checks=checks, check_count=len(checks), maximum_auroc_error=maximum_error,
        reproduction_tolerance=TOLERANCE, reproduction_scope='All22 trained vendor methods,2encodings,21lengths,2supports; includes native-input auxiliary checks not exported as common-input methods.',
        source_bindings=bindings, parameter_records=parameter_records,
        reproduced_by_length_path=str(reproduced_path), reproduced_by_length_sha256=sha(reproduced_path),
        no_new_mapping_fit=True, saved_rotations_reused_byte_for_byte=True,
        ridge_probes_rebuilt_with_unchanged_vendor=True, no_base_training=True, no_neural_forward=True,
        exploratory=True, old_results_used_only_for_fixed_numerical_reproduction=True,
        completed_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
    write(receipt_path, result)
    return result


def synthetic_checks():
    """Format/serialization checks only; no real data, fit, or test scoring."""
    class ProbeModel:
        def __init__(self):
            self.coef_ = np.arange(6, dtype=float).reshape(2, 3)
            self.intercept_ = np.array([.2, -.3])
    def source_closure():
        best_model = ProbeModel()
        scaling = SimpleNamespace(mean_=np.arange(3.), scale_=np.arange(1., 4.))
        return lambda x: (best_model, scaling, x)
    params = {}
    hashes = probe_parameters(source_closure(), task='synthetic', n=4, alpha=10,
                             fit_indices=np.array([0, 2]), dev_indices=np.array([1]), parameters=params)
    assert len(hashes) == 7 and np.array_equal(params['synthetic_n4_coef'], np.arange(6.).reshape(2, 3))
    count = 4
    scores = {method: (np.full((2, 2), j / 100.), np.array([[True, False], [False, True]]))
              for j, method in enumerate(METHODS)}
    packed = np.full((len(METHODS), count, 28), np.nan)
    labels = np.zeros((count, 28), bool)
    auc = np.full((len(METHODS), count), np.nan)
    _pack_length(scores, scores[METHODS[0]][1], np.array([1, 3]), 4, packed, labels, auc,
                 lambda s, t: np.full(len(s), .5))
    assert np.isnan(packed[:, [0, 2]]).all() and np.isnan(packed[:, :, 2:]).all()
    assert np.array_equal(auc[:, [1, 3]], np.full((len(METHODS), 2), .5))
    assert labels[1, 0] and labels[3, 1] and not labels[:, 2:].any()
    assert _scalar_equal(.5, .5) == (True, 0.)
    assert _scalar_equal(np.nan, np.nan) == (True, 0.)
    assert not _scalar_equal(.5, .50001)[0]
    assert len(METHODS) == 16 and len(set(METHODS)) == 16
    # Evaluate the exact vendor scorer on synthetic known-orientation data.
    # Importing the definition does not invoke fit_probe or rotation.
    vendor()
    from alignment import scores_for_length
    x = np.arange(12, dtype=float).reshape(4, 3)
    mx, my = np.array([10., 12., -3.]), np.array([-5., 8., 2.])
    q = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    target = (x-mx) @ q + my
    truth = np.array([[True, False], [False, True], [True, False], [False, True]])
    probe = lambda value: value[:, :2]
    row = dict(peak=truth, valley_masks=(truth, truth, truth), probe=probe, own=probe,
               xt=x, yt=[target.copy() for _ in range(3)], mu_x=mx,
               mu_y=[my.copy() for _ in range(3)], peak_fit=truth)
    scores = scores_for_length(row, [np.eye(3), q.T, np.eye(3)], [np.eye(3) for _ in SHUFFLE_SEEDS])
    np.testing.assert_allclose(scores['fit_complement__test_complement'][0], probe(x), atol=1e-14, rtol=0)
    assert not np.allclose(scores['fit_identity__test_complement'][0], probe(x))
    assert len(scores) == 22 and set(METHODS).issubset(scores)
    assert all(np.array_equal(scores[name][1], truth) for name in METHODS)
    return dict(status='passed', synthetic_only=True, no_real_data_access=True,
                no_probe_or_mapping_fits=True, no_neural_forward=True,
                checks=['closure parameter serialization', 'interior-score NaN padding',
                        'original anchor order', 'common-label invariance',
                        'strict finite/NaN scalar comparison', 'complete16methodinventory',
                        'vendor target-to-source rotation orientation and centering',
                        'all22vendor expressions and shared16methodlabels'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit', choices=('archdev001', 'unit02', 'unit03', 'unit04', 'unit05'))
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    if args.synthetic:
        print(json.dumps(synthetic_checks()), flush=True)
        return
    if args.unit is None:
        parser.error('--unit is required outside synthetic checks')
    frozen_config()
    audit = read(DIAG / 'cache_audit.json')
    assert audit['status'] == 'passed'
    unit = next(row for row in audit['units'] if row['unit'] == args.unit)
    with threadpool_limits(limits=2):
        receipt = run_unit(unit)
    print(json.dumps({key: receipt[key] for key in ('status', 'unit', 'check_count', 'maximum_auroc_error', 'elapsed_seconds')}), flush=True)


if __name__ == '__main__':
    main()
