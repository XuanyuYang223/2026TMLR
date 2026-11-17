"""Bounded post-result geometry/gradient audit on exact saved local edits."""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .gradient_graph import early_logits, margin
from experiments.permworld_local_causal_20261110.recurrent import load_model, parameter_hash
from experiments.permworld_local_causal_20261110.shared import families, catalog, ids_for

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'diagnostics/permworld_closing_audit_20261110/gradient'
TIMING = ROOT / 'diagnostics/permworld_local_timing_20261110'
OLD = Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')
OUT = Path('/mnt/yangx-projects-f/20261110/permworld_closing_audit/gradient')
SEEDS = tuple(range(1026, 1034))
EPS = 1e-12


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze():
    path = HERE / 'run_freeze.json'
    if path.exists():
        return verify()
    files = [Path(__file__), Path(__file__).with_name('gradient_graph.py'), HERE / 'spec.json',
             ROOT / 'tests/test_permworld_closing_gradient.py']
    value = dict(created_utc=datetime.now(timezone.utc).isoformat(), before_gradient_outcomes=True,
                 sources={str(p): sha(p) for p in files})
    write(path, value)
    return value


def verify():
    frozen = read(HERE / 'run_freeze.json')
    for path, digest in frozen['sources'].items():
        assert sha(path) == digest, ('Frozen gradient source changed', path)
    return frozen


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True


def scalar(array):
    value = float(array)
    return value if np.isfinite(value) else ''


def normalized_dot(g, delta):
    norm_g = np.linalg.norm(g, axis=1)
    norm_d = np.linalg.norm(delta, axis=1)
    dot = np.einsum('bd,bd->b', g, delta)
    cosine = np.full(len(g), np.nan)
    valid = (norm_g > EPS) & (norm_d > EPS)
    cosine[valid] = np.clip(dot[valid] / norm_g[valid] / norm_d[valid], -1, 1)
    angle = np.degrees(np.arccos(cosine))
    return norm_g, norm_d, dot, cosine, angle, valid


def finite_difference(model64, previous, h, suffix, start, counts, signs, local, random):
    """Pre-fixed two-direction FP64 shadow check; never tune step/tolerance."""
    device = next(model64.parameters()).device
    prev = torch.as_tensor(previous, device=device, dtype=torch.float64)
    h64 = torch.tensor(h, device=device, dtype=torch.float64, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=False):
        baseline = margin(early_logits(model64, prev, h64, suffix, start), counts, signs)
        gradient, = torch.autograd.grad(baseline.sum(), h64)
        results = []
        for name, value in [('local', local), ('random_1026', random)]:
            norm = np.linalg.norm(value.astype(np.float64), axis=1)
            if np.any(norm <= EPS):
                raise ValueError('Fixed FD direction near zero; keep failure, do not select another')
            unit = torch.as_tensor(value / norm[:, None], device=device, dtype=torch.float64)
            analytic = (gradient * unit).sum(1).detach().cpu().numpy()
            with torch.no_grad():
                plus = margin(early_logits(model64, prev, h64 + 1e-4 * unit, suffix, start), counts, signs)
                minus = margin(early_logits(model64, prev, h64 - 1e-4 * unit, suffix, start), counts, signs)
                numerical = ((plus - minus) / 2e-4).cpu().numpy()
            error = np.abs(numerical - analytic)
            threshold = 1e-7 + 1e-3 * np.abs(analytic)
            for index in range(len(h)):
                results.append(dict(row=index, direction=name, analytic=float(analytic[index]),
                                    finite_difference=float(numerical[index]), absolute_error=float(error[index]),
                                    allowed_error=float(threshold[index]), passed=bool(error[index] <= threshold[index])))
        return results


def execute(unit, spec):
    name = unit['unit']; dest = HERE / 'results' / name; dest.mkdir(parents=True, exist_ok=True)
    finished = dest / 'completion.json'
    if finished.exists():
        receipt = read(finished)
        assert receipt['spec_sha256'] == sha(HERE / 'spec.json')
        for file, digest in receipt['files'].items():
            assert sha(file) == digest, file
        print('GRADIENT REUSE', name, flush=True); return receipt
    started = time.monotonic()
    selected = [family for family in families(name) if family['split'] == 'test']
    records = {record['pair_id']: record for family in selected for record in family['pairs']
               if record['kind'] in ('simple', 'count_matched')}
    timing_receipt = read(TIMING / 'results' / name / 'completion.json')
    timing_cells = {(task['task'], cell['n'], cell['i']): cell for task in timing_receipt['task_reports'] for cell in task['cells']}
    old_audit = read(TIMING / 'input_audit.json')
    old_unit = next(item for item in old_audit['units'] if item['unit'] == name)
    old_cells = {(task, cell['n'], cell['i']): cell for task, cells in old_unit['cells'].items() for cell in cells}
    source_inputs = {}
    for path in [TIMING / 'protocol.json', TIMING / 'input_audit.json',
                 TIMING / 'results' / name / 'completion.json', OLD / name / 'selected_families.json.gz']:
        source_inputs[str(path)] = sha(path)
    physical = OUT / name; physical.mkdir(parents=True, exist_ok=True)
    pair_path = physical / 'gradient_per_pair.csv.gz'
    random_path = physical / 'random_directions.csv.gz'
    pair_stream = gzip.open(str(pair_path) + '.partial', 'wt', newline='')
    random_stream = gzip.open(str(random_path) + '.partial', 'wt', newline='')
    pair_writer = random_writer = None
    all_rows = []; fd_rows = []; artifacts = []; task_reports = []; pair_count = 0
    try:
        for task in ['peaks', 'valleys']:
            model = load_model(unit['models'][task], 'cuda' if torch.cuda.is_available() else 'cpu', spec['model_config'])
            before = parameter_hash(model)
            source_inputs[unit['models'][task]['checkpoint_path']] = unit['models'][task]['checkpoint_sha256']
            # Exact frozen weight values in higher precision for derivative wiring only.
            model64 = copy.deepcopy(model).to(dtype=torch.float64).eval().requires_grad_(False)
            nodes, _ = catalog(selected, task, {'test'})
            node_by_id = {row['node_id']: row for row in nodes}
            probe_path = OLD / name / 'probes' / task / 'layer1_point1.npz'
            assert sha(probe_path) == old_unit['probes'][task]['sha256']
            source_inputs[str(probe_path)] = old_unit['probes'][task]['sha256']
            with np.load(probe_path, allow_pickle=False) as z:
                saved_v = z['direction'].copy()
                raw_weight = z['edit_ridge_coef'].copy() / z['edit_scaler_scale']
                assert np.allclose(raw_weight, z['edit_weight'], atol=1e-12, rtol=0)
                assert np.allclose(raw_weight / np.linalg.norm(raw_weight), saved_v, atol=1e-12, rtol=0)
            maxima = dict(baseline_logit_error=0., edited_logit_error=0., projection_coordinate_error=0.,
                          ideal_vs_actual_l2=0., random_norm_error=0.)
            task_pairs = 0
            for n, i in sorted((n, i) for t, n, i in timing_cells if t == task):
                desc = timing_cells[task, n, i]; old_desc = old_cells[task, n, i]
                assert sha(desc['path']) == desc['sha256']
                assert sha(old_desc['path']) == old_desc['sha256']
                source_inputs[desc['path']] = desc['sha256']
                source_inputs[old_desc['path']] = old_desc['sha256']
                with np.load(desc['path'], allow_pickle=False) as z:
                    new = {key: z[key].copy() for key in z.files}
                with np.load(old_desc['path'], allow_pickle=False) as z:
                    old = {key: z[key].copy() for key in z.files}
                assert np.array_equal(new['pair_ids'], old['pair_ids'])
                assert np.array_equal(saved_v, old['direction'])
                assert str(new['source_cell_sha256'].item()) == old_desc['sha256']
                h = old['recipient_states'][:, 0]
                donor_difference = old['donor_states'][:, 0].astype(np.float64) - h.astype(np.float64)
                injected_names = list(map(str, new['injected_conditions']))
                actual_h = new['injected_layer1_states'][injected_names.index('local_projection')]
                delta = actual_h.astype(np.float64) - h.astype(np.float64)
                random_delta = np.stack([
                    new['injected_layer1_states'][injected_names.index(f'random_direction_{seed}')].astype(np.float64) - h
                    for seed in SEEDS])
                coordinate = donor_difference @ saved_v
                ideal = coordinate[:, None] * saved_v
                assert np.max(np.abs(coordinate - old['coordinate'])) <= 1e-12
                assert np.array_equal((h.astype(np.float64) + ideal).astype(np.float32), actual_h)
                maxima['projection_coordinate_error'] = max(maxima['projection_coordinate_error'], float(np.max(np.abs(coordinate - old['coordinate']))))
                maxima['ideal_vs_actual_l2'] = max(maxima['ideal_vs_actual_l2'], float(np.linalg.norm(delta - ideal, axis=1).max()))
                ids = [str(k) for k in old['pair_ids']]
                meta = [records[k] for k in ids]
                counts = np.array([item['recipient_total_count'] for item in meta])
                signs = np.array([item['direction'] for item in meta])
                nodes_here = [node_by_id[str(k)] for k in old['recipient_node_ids']]
                assert all(int(row['total_count']) == int(count) for row, count in zip(nodes_here, counts))
                token_ids = ids_for([row['permutation'] for row in nodes_here], task)
                start = 5 + 2 * i; suffix = token_ids[:, start:]
                previous = new['recipient_previous_states']
                times = list(map(str, new['timings'])); conditions = list(map(str, new['conditions']))
                baseline_index = next(k for k, (a, b) in enumerate(zip(times, conditions)) if (a, b) == ('early', 'unedited'))
                edited_index = next(k for k, (a, b) in enumerate(zip(times, conditions)) if (a, b) == ('early', 'local_projection'))
                baseline_logits = new['logits'][baseline_index]
                edited_logits = new['logits'][edited_index]
                parameter = next(model.parameters())
                h_tensor = torch.tensor(h, device=parameter.device, dtype=torch.float32, requires_grad=True)
                with torch.backends.cudnn.flags(enabled=False):
                    logits = early_logits(model, previous, h_tensor, suffix, start)
                    m = margin(logits, counts, signs)
                    gradient, = torch.autograd.grad(m.sum(), h_tensor)
                    assert gradient is not None and bool(torch.isfinite(gradient).all())
                    g = gradient.detach().cpu().numpy().astype(np.float64)
                    reconstructed = logits.detach().cpu().numpy()
                    with torch.no_grad():
                        changed = early_logits(model, previous, torch.tensor(actual_h, device=parameter.device), suffix, start).cpu().numpy()
                baseline_error = float(np.abs(reconstructed - baseline_logits).max())
                edited_error = float(np.abs(changed - edited_logits).max())
                maxima['baseline_logit_error'] = max(maxima['baseline_logit_error'], baseline_error)
                maxima['edited_logit_error'] = max(maxima['edited_logit_error'], edited_error)
                assert baseline_error <= 1e-4, ('Frozen baseline mismatch', name, task, n, i, baseline_error)
                assert edited_error <= 1e-4, ('Frozen edit mismatch', name, task, n, i, edited_error)
                gn, dn, dot, cosine, angle, defined = normalized_dot(g, delta)
                donor_norm = np.linalg.norm(donor_difference, axis=1)
                energy = np.full(len(h), np.nan)
                energy[donor_norm > EPS] = coordinate[donor_norm > EPS] ** 2 / donor_norm[donor_norm > EPS] ** 2
                row_indices = np.arange(len(h))
                baseline_m = baseline_logits[row_indices, counts + signs].astype(np.float64) - baseline_logits[row_indices, counts]
                edited_m = edited_logits[row_indices, counts + signs].astype(np.float64) - edited_logits[row_indices, counts]
                random_dots = []; random_equal = []; random_actual_changes = []
                for random_index, seed in enumerate(SEEDS):
                    _, random_norm, random_dot, random_cos, random_angle, random_defined = normalized_dot(g, random_delta[random_index])
                    error = np.abs(random_norm - dn)
                    maxima['random_norm_error'] = max(maxima['random_norm_error'], float(error.max()))
                    assert np.all(error <= 1e-5 + 1e-5 * dn)
                    equal = np.full(len(h), np.nan)
                    valid = random_norm > EPS
                    equal[valid] = random_dot[valid] * dn[valid] / random_norm[valid]
                    random_logits = new['logits'][next(k for k, pair in enumerate(zip(times, conditions)) if pair == ('early', f'random_direction_{seed}'))]
                    random_m = random_logits[row_indices, counts + signs].astype(np.float64) - random_logits[row_indices, counts]
                    random_dots.append(random_dot); random_equal.append(equal); random_actual_changes.append(random_m - baseline_m)
                    for j in range(len(h)):
                        row = dict(unit=name, task=task, pair_id=ids[j], group='main' if meta[j]['kind'] == 'simple' else 'matched',
                                   sign=int(signs[j]), length=n, position=i, random_seed=seed, displacement_norm=float(random_norm[j]),
                                   matched_local_norm=float(dn[j]), norm_error=float(error[j]), g_dot_delta=float(random_dot[j]),
                                   equal_local_norm_g_dot_direction=scalar(equal[j]), cosine=scalar(random_cos[j]), angle_degrees=scalar(random_angle[j]),
                                   angle_defined=bool(random_defined[j]), actual_logit_margin_change=float(random_m[j] - baseline_m[j]))
                        if random_writer is None:
                            random_writer = csv.DictWriter(random_stream, fieldnames=list(row)); random_writer.writeheader()
                        random_writer.writerow(row)
                if n in [10, 20, 30] and i == 6:
                    fd_indices = []
                    for group in ['simple', 'count_matched']:
                        for sign in [-1, 1]:
                            fd_indices.append(next(j for j, row in enumerate(meta) if row['kind'] == group and row['direction'] == sign))
                    fd = finite_difference(model64, previous[fd_indices], h[fd_indices], suffix[fd_indices], start,
                                           counts[fd_indices], signs[fd_indices], delta[fd_indices], random_delta[0, fd_indices])
                    for row in fd:
                        index = fd_indices[row.pop('row')]
                        row.update(unit=name, task=task, length=n, position=i, pair_id=ids[index],
                                   group='main' if meta[index]['kind'] == 'simple' else 'matched', sign=int(signs[index]))
                        fd_rows.append(row)
                    assert all(row['passed'] for row in fd), ('Finite difference check failed', name, task, n)
                cell_path = physical / 'cells' / task / f'n{n}_i{i}.npz'; cell_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cell_path, pair_ids=old['pair_ids'], recipient_node_ids=old['recipient_node_ids'],
                    donor_node_ids=old['donor_node_ids'], raw_gradient=g, donor_difference=donor_difference,
                    actual_displacement=delta, ideal_orthogonal_projection=ideal, raw_unit_axis=saved_v,
                    random_displacements=random_delta, random_seeds=np.array(SEEDS),
                    true_recipient_count=counts, local_sign=signs, cached_baseline_margin=baseline_m,
                    cached_edited_margin=edited_m, source_timing_cell_sha256=np.asarray(desc['sha256']),
                    source_old_cell_sha256=np.asarray(old_desc['sha256']), checkpoint_sha256=np.asarray(unit['models'][task]['checkpoint_sha256']),
                    data_sha256=np.asarray(unit['data_sha256']), spec_sha256=np.asarray(sha(HERE / 'spec.json')))
                artifacts.append(cell_path)
                for j in range(len(h)):
                    row = dict(unit=name, historical_reference=unit['historical'], task=task,
                               group='main' if meta[j]['kind'] == 'simple' else 'matched', sign=int(signs[j]),
                               pair_id=ids[j], family_id=meta[j]['family_id'], length=n, position=i,
                               true_recipient_count=int(counts[j]), desired_local_count=int(counts[j] + signs[j]),
                               donor_final_count=int(meta[j]['donor_total_count']), donor_difference_norm=float(donor_norm[j]),
                               actual_displacement_norm=float(dn[j]), ideal_projection_norm=float(abs(coordinate[j])),
                               projection_energy_fraction=scalar(energy[j]), actual_vs_ideal_norm=float(np.linalg.norm(delta[j] - ideal[j])),
                               gradient_norm=float(gn[j]), g_dot_delta=float(dot[j]), cosine=scalar(cosine[j]), angle_degrees=scalar(angle[j]),
                               angle_defined=bool(defined[j]), gradient_exact_zero=bool(gn[j] == 0), gradient_near_zero=bool(gn[j] <= EPS),
                               displacement_exact_zero=bool(dn[j] == 0), displacement_near_zero=bool(dn[j] <= EPS),
                               donor_difference_near_zero=bool(donor_norm[j] <= EPS), baseline_logit_margin=float(baseline_m[j]),
                               edited_logit_margin=float(edited_m[j]), actual_logit_margin_change=float(edited_m[j] - baseline_m[j]),
                               linear_prediction_error=float((edited_m[j] - baseline_m[j]) - dot[j]),
                               random_mean_g_dot_delta=float(np.mean(random_dots, axis=0)[j]),
                               random_equal_norm_mean_g_dot_delta=scalar(np.mean(random_equal, axis=0)[j]),
                               random_mean_actual_margin_change=float(np.mean(random_actual_changes, axis=0)[j]),
                               local_minus_random_equal_norm_g_dot_delta=scalar(dot[j] - np.mean(random_equal, axis=0)[j]),
                               baseline_argmax_token=int(baseline_logits[j].argmax()), edited_argmax_token=int(edited_logits[j].argmax()))
                    if pair_writer is None:
                        pair_writer = csv.DictWriter(pair_stream, fieldnames=list(row)); pair_writer.writeheader()
                    pair_writer.writerow(row); all_rows.append(row)
                pair_count += len(h); task_pairs += len(h)
                if i == n - 2:
                    print('GRADIENT', name, task, 'n', n, 'cases', task_pairs, 'elapsed', round(time.monotonic() - started, 1), flush=True)
            after = parameter_hash(model)
            assert before == after == unit['models'][task]['parameter_sha256']
            task_reports.append(dict(task=task, pairs=task_pairs, maxima=maxima,
                                     parameter_sha256_before=before, parameter_sha256_after=after,
                                     no_fitted_objects=True, autograd_inputs_only=True, all_modules_eval=True))
            del model64, model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    finally:
        pair_stream.close(); random_stream.close()
    Path(str(pair_path) + '.partial').replace(pair_path)
    Path(str(random_path) + '.partial').replace(random_path)
    for label, source in [('gradient_per_pair.csv.gz', pair_path), ('random_directions.csv.gz', random_path)]:
        alias = dest / label
        if not alias.exists(): alias.symlink_to(source)
        assert alias.resolve() == source
    write_csv(dest / 'finite_difference.csv', fd_rows)
    summary = aggregate(all_rows)
    write_csv(dest / 'by_length.csv', summary['by_length'])
    write_csv(dest / 'by_unit.csv', summary['by_unit'])
    artifacts += [pair_path, random_path, dest / 'finite_difference.csv', dest / 'by_length.csv', dest / 'by_unit.csv']
    input_rows = []
    for path, expected in sorted(source_inputs.items()):
        actual = sha(path)
        input_rows.append(dict(path=path, expected_sha256=expected, end_sha256=actual, unchanged=actual == expected))
    assert all(row['unchanged'] for row in input_rows)
    write(dest / 'input_bindings.json', dict(status='passed', scope='Explicit raw cell/checkpoint/probe/family and source-receipt inputs used by this unit at recorded endpoints, not continuous monitoring.', files=input_rows))
    artifacts.append(dest / 'input_bindings.json')
    receipt = dict(status='completed', unit=name, historical_reference=unit['historical'], pairs=pair_count,
        spec_sha256=sha(HERE / 'spec.json'), source_freeze_sha256=sha(HERE / 'run_freeze.json'),
        elapsed_seconds=time.monotonic() - started, task_reports=task_reports, files={str(path): sha(path) for path in artifacts},
        finite_difference_rows=len(fd_rows), finite_difference_all_passed=all(row['passed'] for row in fd_rows),
        no_base_training=True, no_probe_fitting=True, no_direction_selection=True, no_new_conditions=True)
    write(finished, receipt); return receipt


METRICS = ['donor_difference_norm', 'actual_displacement_norm', 'ideal_projection_norm', 'projection_energy_fraction',
           'actual_vs_ideal_norm', 'gradient_norm', 'g_dot_delta', 'cosine', 'angle_degrees', 'angle_defined',
           'gradient_exact_zero', 'gradient_near_zero', 'displacement_exact_zero', 'displacement_near_zero',
           'donor_difference_near_zero', 'baseline_logit_margin', 'edited_logit_margin', 'actual_logit_margin_change',
           'linear_prediction_error', 'random_mean_g_dot_delta', 'random_equal_norm_mean_g_dot_delta',
           'random_mean_actual_margin_change', 'local_minus_random_equal_norm_g_dot_delta']


def write_csv(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    assert values
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)


def finite_mean(values):
    finite = np.asarray([float(value) for value in values if value != '' and np.isfinite(float(value))])
    return float(finite.mean()) if len(finite) else float('nan')


def aggregate(rows):
    centers = defaultdict(list)
    for row in rows:
        centers[row['unit'], row['task'], row['group'], row['sign'], row['length'], row['position']].append(row)
    lengths = defaultdict(list)
    for key, group in centers.items():
        for metric in METRICS:
            values = [row[metric] for row in group]
            finite_count = sum(value != '' and np.isfinite(float(value)) for value in values)
            lengths[key[:5] + (metric,)].append(dict(center=key[-1], mean=finite_mean(values), n=len(values), finite=finite_count))
    by_length = []
    units = defaultdict(list)
    for key, group in sorted(lengths.items()):
        unit, task, kind, sign, n, metric = key
        expected = n - (3 if kind == 'main' else 7)
        assert len(group) == expected
        value = float(np.mean([row['mean'] for row in group]))
        row = dict(unit=unit, task=task, group=kind, sign=sign, length=n, metric=metric,
                   mean=scalar(value), registered_centers=expected, finite_centers=sum(np.isfinite(row['mean']) for row in group),
                   samples=sum(row['n'] for row in group), finite_samples=sum(row['finite'] for row in group),
                   undefined_samples=sum(row['n'] - row['finite'] for row in group))
        by_length.append(row)
        units[unit, task, kind, sign, metric].append(row)
    by_unit = []
    for key, group in sorted(units.items()):
        assert len(group) == 21 and {row['length'] for row in group} == set(range(10, 31))
        values = [float(row['mean']) if row['mean'] != '' else np.nan for row in group]
        by_unit.append(dict(unit=key[0], task=key[1], group=key[2], sign=key[3], metric=key[4],
                            mean=scalar(np.mean(values)), fixed_length_weight='1/21', registered_lengths=21,
                            finite_lengths=sum(np.isfinite(values)), samples=sum(row['samples'] for row in group),
                            finite_samples=sum(row['finite_samples'] for row in group), undefined_samples=sum(row['undefined_samples'] for row in group)))
    return dict(by_length=by_length, by_unit=by_unit)


def summarize(spec):
    all_units = []
    for unit in spec['units']:
        path = HERE / 'results' / unit['unit'] / 'by_unit.csv'
        with path.open(newline='') as handle: all_units += list(csv.DictReader(handle))
    write_csv(HERE / 'all_units.csv', all_units)
    grouped = defaultdict(list)
    for row in all_units:
        if row['unit'] != 'archdev001': grouped[row['task'], row['group'], row['sign'], row['metric']].append(row)
    rows = []
    for key, values in sorted(grouped.items()):
        assert len(values) == 4
        means = np.array([float(row['mean']) if row['mean'] != '' else np.nan for row in values])
        finite = means[np.isfinite(means)]
        rows.append(dict(task=key[0], group=key[1], sign=key[2], metric=key[3],
            mean=scalar(np.mean(means)), sample_sd=scalar(np.std(means, ddof=1)),
            minimum=scalar(np.min(means)), maximum=scalar(np.max(means)), registered_units=4, finite_units=len(finite),
            positive_units=int(np.sum(means > 0)), negative_units=int(np.sum(means < 0)),
            raw_unit_values=json.dumps({row['unit']: row['mean'] for row in values}),
            undefined_samples=sum(int(row['undefined_samples']) for row in values)))
    write_csv(HERE / 'four_unit_summary.csv', rows)
    write(HERE / 'summary_completion.json', dict(status='completed', formal_units=['unit02','unit03','unit04','unit05'],
        historical_separate=True, metrics=len(METRICS), four_unit_rows=len(rows), units=5, no_new_fits=True,
        spec_sha256=sha(HERE / 'spec.json'), finished_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'historical', 'new', 'all', 'summarize'])
    args = parser.parse_args(); setup()
    if args.stage == 'freeze': freeze(); return
    verify(); spec = read(HERE / 'spec.json')
    if args.stage == 'summarize': summarize(spec); return
    units = spec['units'] if args.stage == 'all' else spec['units'][:1] if args.stage == 'historical' else spec['units'][1:]
    with threadpool_limits(limits=2):
        for unit in units: verify(); execute(unit, spec)
    if args.stage == 'all': summarize(spec)


if __name__ == '__main__':
    main()
