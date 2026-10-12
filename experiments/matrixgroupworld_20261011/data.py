"""Independent worlds, closed-orbit splits, fixed shared batch tables and label seal.

Preparing analysis matrices does not call labels on them. Labels for that split
are generated only by seal_analysis_labels after the root protocol is frozen.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from . import core

SPLITS = ('train', 'validation', 'analysis')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def config_signature(config):
    fields = {k: config[k] for k in ('units', 'lengths', 'data', 'training', 'engineering')}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sample_gl(rng, n, counts):
    while True:
        a = rng.integers(0, 2, (n, n), dtype=np.uint8)
        counts['iid_binary_proposals'] += 1
        if core.rank(a) == n:
            counts['full_rank_proposals'] += 1
            return a
        counts['singular_rejections'] += 1


def padded(a):
    out = np.zeros((10, 10), dtype=np.uint8)
    out[:len(a), :len(a)] = a
    return out


def padded_input(a):
    out = np.full(128, core.TOKEN_TO_ID['<PAD>'], dtype=np.int16)
    ids = core.encode(a); out[:len(ids)] = ids
    return out


def historical_index(config):
    """Only explicitly located metadata and archives are covered, not all disks.

    Previously located F13/F17 GL2 archives are documented as incompatible.
    A compatible explicit archive requires field_prime=2 and matrix_key; its
    matrices are canonicalized and excluded. Unknown provenance is never silently
    treated as an exact matching historical GL(n,F2) index.
    """
    records = []
    located = (
        core.WORKSPACE / 'results/algebra_relation_v3/matrix/data_audit.json',
        core.WORKSPACE / 'results/readout_null_confirmation/matrix/data_audit.json',
        core.WORKSPACE / 'results/final_mechanism_confirmation/protocol.json',
    )
    for p in located:
        item = {'path': str(p), 'exists': p.exists(), 'purpose': 'Located previous finite-field matrix metadata'}
        if p.exists():
            meta = json.loads(p.read_text())
            item.update(sha256=sha(p), field_prime=meta.get('p', meta.get('field_prime')),
                        status='not_same_field_or_scale' if meta.get('p', meta.get('field_prime')) != 2 else 'requires_explicit_matrix_archive')
            # Those experiments serialize four field coordinates / 2x2 matrices.
            item['located_old_matrix_order'] = 2
        records.append(item)
    keys = set()
    for item in config.get('historical_archives', []):
        p = Path(item['path'])
        record = dict(item, exists=p.exists())
        if not p.exists():
            record['status'] = 'missing_cannot_verify'; records.append(record); continue
        record['sha256'] = sha(p)
        if item.get('sha256') and record['sha256'] != item['sha256']:
            raise RuntimeError(f'Historical archive identity changed: {p}')
        if item.get('field_prime') != 2:
            record['status'] = 'not_same_field'; records.append(record); continue
        if 'matrix_key' not in item:
            raise RuntimeError('Compatible archive requires an explicit matrix_key')
        with np.load(p, allow_pickle=False) as ar:
            matrices = ar[item['matrix_key']]
            lengths = ar[item['length_key']] if item.get('length_key') else None
            count = 0
            for i, a in enumerate(matrices):
                n = int(lengths[i]) if lengths is not None else int(a.shape[-1])
                if n not in core.LENGTHS:
                    continue
                keys.add(core.orbit_key(a[:n, :n])); count += 1
        record.update(status='exact_compatible_orbits_excluded', compatible_inputs=count)
        records.append(record)
    return keys, {'metadata_and_explicit_archives': records, 'excluded_unique_orbits': len(keys),
                  'coverage_limit': 'Only the listed located metadata and explicitly supplied compatible archives were checked; no claim of a comprehensive disk-wide historical inventory.',
                  'located_same_object_scale_index': bool(keys)}


def make_engineering(config, root, historical_keys):
    root = Path(root); out = root / 'engineering/data.npz'
    engineering = config['engineering']
    seed = int(engineering.get('seed', 98000031))
    n = int(engineering.get('overfit_n', engineering.get('length', 6)))
    count = int(engineering.get('overfit_samples', engineering.get('samples', 32)))
    if n != 6 or count != 32:
        raise ValueError('Engineering fixture is frozen to 32 matrices of order six')
    rng = np.random.default_rng(seed); counts = Counter(); matrices = []; keys = set()
    while len(matrices) < count:
        a = sample_gl(rng, n, counts); k = core.orbit_key(a)
        if k in keys or k in historical_keys:
            counts['orbit_rejections'] += 1; continue
        keys.add(k); matrices.append(a)
    y = np.stack([core.labels(a) for a in matrices])
    if any(len(np.unique(y[:, j])) < 2 for j in range(4)):
        raise RuntimeError('Fixed engineering sample has a constant task; stop without resampling')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, matrices=np.stack([padded(a) for a in matrices]),
                        lengths=np.full(count, n, dtype=np.int16), labels=y,
                        input=np.stack([padded_input(a) for a in matrices]),
                        ids=np.array([core.input_id(a) for a in matrices]),
                        orbit_ids=np.array([core.orbit_id(a) for a in matrices]))
    audit = {'status': 'passed', 'seed': seed, 'samples': count, 'n': n,
             'data_path': str(out), 'data_sha256': sha(out), 'counts': dict(counts),
             'label_classes': {task: sorted(map(int, np.unique(y[:, j]))) for j, task in enumerate(core.TASKS)},
             'excluded_from_all_formal_splits': True}
    write_json(root / 'engineering/data_audit.json', audit)
    return keys, audit


def split_count(config, split):
    default = dict(train=4096, validation=256, analysis=512)
    return int(config['data'].get('per_n', {}).get(split,
               config['data'].get(f'{split}_per_length', default[split])))


def make_schedule(config, unit, arrays, out):
    rng = np.random.default_rng(int(unit['batch_seed']))
    updates = int(config['training']['updates'])
    bs = int(config['training'].get('effective_batch', config['training'].get('batch_size', 64)))
    if updates != 20000 or bs != 64:
        raise ValueError('Expected the fixed 20k effective-batch64 schedule')
    steps = np.resize(np.asarray(config['lengths'], dtype=np.int16), updates)
    rng.shuffle(steps)
    choices = {n: np.flatnonzero(arrays['train_lengths'] == n) for n in config['lengths']}
    ids = np.empty((updates, bs), dtype=np.int32)
    for i, n in enumerate(steps):
        ids[i] = rng.choice(choices[int(n)], bs, replace=False)
    np.savez_compressed(out, steps_lengths=steps, batch_ids=ids)
    exposures = np.bincount(ids.reshape(-1), minlength=len(arrays['train_lengths']))
    return {'path': str(out), 'sha256': sha(out), 'updates': updates, 'effective_batch': bs,
            'samples_per_update_distinct': True, 'shared_by_four_tasks': True,
            'length_update_counts': {str(n): int((steps == n).sum()) for n in config['lengths']},
            'total_exposures_per_model': int(ids.size), 'minimum_anchor_exposures': int(exposures.min()),
            'maximum_anchor_exposures': int(exposures.max())}


def learning_audit(arrays, split):
    rows = []
    ns = arrays[f'{split}_lengths']; ys = arrays[f'{split}_labels']
    for n in core.LENGTHS:
        y = ys[ns == n]
        for j, task in enumerate(core.TASKS):
            counts = Counter(map(int, y[:, j]))
            if len(counts) < 2:
                raise RuntimeError(f'Constant {split} identity labels for n{n} {task}')
            winner = min(counts, key=lambda v: (-counts[v], v))
            rows.append(dict(split=split, n=n, task=task, samples=len(y),
                             classes=len(counts), histogram={str(k): v for k, v in sorted(counts.items())},
                             mode=winner, mode_accuracy=counts[winner] / len(y)))
    return rows


def stratum_audit(arrays, split):
    """Uses train/validation only, never final analysis labels or geometry."""
    rows = []
    for n in core.LENGTHS:
        chosen = np.flatnonzero(arrays[f'{split}_lengths'] == n)
        matrices = arrays[f'{split}_matrices'][chosen, :n, :n]
        identity = arrays[f'{split}_labels'][chosen]
        action_labels = {action: np.stack([core.labels(core.transform(a, action)) for a in matrices])
                         for action in set(r[k] for r in core.RELATIONS for k in ('correct', 'wrong'))}
        for r in core.RELATIONS:
            left, right = core.TASKS.index(r['left']), core.TASKS.index(r['right'])
            keys = [(n, int(y[left]), int(y[right]), int(c[right]), int(w[right]))
                    for y, c, w in zip(identity, action_labels[r['correct']], action_labels[r['wrong']])]
            counts = Counter(keys); eligible = [v for v in counts.values() if v >= 3]
            retained = sum(eligible); dof = sum(v - 1 for v in eligible)
            if retained <= 10 or dof < 2:
                raise RuntimeError(f'Degenerate fixed answer strata: {split} n{n} {r["relation"]}')
            rows.append(dict(split=split, n=n, relation=r['relation'], samples=len(keys),
                             groups=len(counts), minimum_group_size=3, retained_samples=retained,
                             retained_groups=len(eligible), residual_degrees_of_freedom=dof,
                             dropped_samples=len(keys) - retained,
                             joint_key='n,left(identity),right(identity),right(correct),right(wrong)'))
    return rows


def audit_split_orbits(arrays):
    """Every anchor and frozen view; no labels read for analysis matrices."""
    all_keys = {}; sizes = Counter(); checks = 0
    for split in SPLITS:
        for n, padded_a, recorded in zip(arrays[f'{split}_lengths'], arrays[f'{split}_matrices'],
                                        arrays[f'{split}_orbit_ids']):
            n = int(n); a = padded_a[:n, :n]; k = core.orbit_key(a)
            if k.hex() != str(recorded) or k in all_keys:
                raise RuntimeError('Invalid recorded orbit identity or duplicate split orbit')
            all_keys[k] = split
            candidates = core.orbit_candidates(a)
            member_keys = {core.key(b) for b in candidates}; sizes[len(member_keys)] += 1
            for action in core.ACTIONS:
                b = core.transform(a, action)
                if core.key(b) not in member_keys or core.orbit_key(b) != k:
                    raise RuntimeError(f'Frozen {action} escaped the full closed orbit')
                checks += 1
    return {'status': 'passed', 'all_anchors': len(all_keys), 'frozen_action_checks': checks,
            'cross_split_orbit_intersections': 0, 'within_split_orbit_duplicates': 0,
            'candidate_members_per_anchor': 72,
            'unique_orbit_size_counts': {str(k): v for k, v in sorted(sizes.items())},
            'full_H_closure_scope': 'Independent exhaustive GL2/GL3 and fixed large-n random mathematical audit; every formal anchor verifies all eight frozen actions and their exact canonical identity.'}


def prepare(config, root):
    root = Path(root); start = time.monotonic()
    if tuple(config['lengths']) != core.LENGTHS or len(config['units']) != 2:
        raise ValueError('Only the frozen two-world n6/8/10 design is supported')
    signature = config_signature(config)
    ready_path = root / 'data_ready.json'
    if ready_path.exists():
        old = json.loads(ready_path.read_text())
        if old['preparation_config_signature'] != signature:
            raise RuntimeError('Existing world preparation config changed')
        for unit in old['units']:
            for path_key, hash_key in (('data_path', 'data_sha256'), ('schedule_path', 'schedule_sha256')):
                if sha(unit[path_key]) != unit[hash_key]:
                    raise RuntimeError('Existing world or schedule identity changed')
        if sha(old['engineering']['data_path']) != old['engineering']['data_sha256']:
            raise RuntimeError('Existing engineering data changed')
        return old
    history_keys, history = historical_index(config)
    eng_keys, engineering = make_engineering(config, root, history_keys)
    forbidden = history_keys | eng_keys; units = []; unit_orbits = []
    for unit in config['units']:
        unit_start = time.monotonic(); rng = np.random.default_rng(int(unit['data_seed']))
        counts = Counter(); seen = set(); arrays = {}; split_counts = []
        for split in SPLITS:
            matrices = []; ns = []; ids = []; orbit_ids = []; inputs = []; y = []
            for n in core.LENGTHS:
                per_n_start = dict(counts); target = split_count(config, split)
                for _ in range(target):
                    while True:
                        a = sample_gl(rng, n, counts); k = core.orbit_key(a)
                        if k in forbidden:
                            counts['historical_or_engineering_orbit_rejections'] += 1; continue
                        if k in seen:
                            counts['within_world_orbit_rejections'] += 1; continue
                        seen.add(k); break
                    matrices.append(padded(a)); ns.append(n); ids.append(core.input_id(a))
                    orbit_ids.append(k.hex()); inputs.append(padded_input(a))
                    if split != 'analysis':
                        y.append(core.labels(a))
                split_counts.append({'split': split, 'n': n, 'accepted': target,
                                     'sampling_counts': {k: counts[k] - per_n_start.get(k, 0) for k in counts}})
            arrays[f'{split}_matrices'] = np.stack(matrices)
            arrays[f'{split}_lengths'] = np.asarray(ns, dtype=np.int16)
            arrays[f'{split}_ids'] = np.asarray(ids)
            arrays[f'{split}_orbit_ids'] = np.asarray(orbit_ids)
            arrays[f'{split}_input'] = np.stack(inputs)
            if split != 'analysis':
                arrays[f'{split}_labels'] = np.stack(y)
        out = root / 'worlds' / unit['unit']; out.mkdir(parents=True, exist_ok=True)
        data_path = out / 'data.npz'; schedule_path = out / 'schedule.npz'
        np.savez_compressed(data_path, **arrays)
        schedule = make_schedule(config, unit, arrays, schedule_path)
        orbit_audit = audit_split_orbits(arrays)
        learning = [*learning_audit(arrays, 'train'), *learning_audit(arrays, 'validation')]
        strata = [*stratum_audit(arrays, 'train'), *stratum_audit(arrays, 'validation')]
        audit = {'status': 'passed', **unit, 'data_path': str(data_path), 'data_sha256': sha(data_path),
                 'schedule_path': str(schedule_path), 'schedule_sha256': sha(schedule_path),
                 'sampling': dict(counts), 'sampling_by_split_n': split_counts, 'schedule': schedule,
                 'orbit_audit': orbit_audit, 'learning_labels_train_validation_only': learning,
                 'answer_strata_train_validation_only': strata,
                 'analysis_labels_present': False, 'analysis_label_function_called': False,
                 'sampling_distribution': 'IID uniform binary matrices rejected by exact rank, then canonical orbit uniqueness/exclusions. Accepted canonical orbits are not claimed uniform.',
                 'elapsed_process_wall_seconds': time.monotonic() - unit_start}
        write_json(out / 'data_audit.json', audit); units.append(audit); unit_orbits.append(seen)
    intersection = unit_orbits[0] & unit_orbits[1]
    result = {'status': 'passed', 'prepared_utc': datetime.now(timezone.utc).isoformat(),
              'preparation_config_signature': signature, 'units': units, 'engineering': engineering,
              'historical_exclusions': history, 'cross_unit_unique_orbit_overlap': len(intersection),
              'cross_unit_overlap_keys': sorted(k.hex() for k in intersection),
              'cross_unit_disjointness_required': False, 'analysis_labels_prepared': False,
              'elapsed_process_wall_seconds': time.monotonic() - start,
              'cost_scope': 'Single CPU preparation process elapsed wall time, not CPU-core time or GPU-active time.'}
    write_json(ready_path, result)
    return result


def seal_analysis_labels(config=None, root=None, freeze_path=None, protocol_path=None):
    """Explicit post-freeze stage. Produces no representations or geometry."""
    from .common import CONFIG, ROOT, DIAG
    root = Path(root or ROOT); protocol_path = Path(protocol_path or CONFIG)
    freeze_path = Path(freeze_path or DIAG / 'freeze_manifest.json')
    frozen = json.loads(freeze_path.read_text()); actual_config = json.loads(protocol_path.read_text())
    if sha(protocol_path) != frozen['protocol_sha256']:
        raise RuntimeError('Analysis labels require the exact frozen final protocol')
    for source, expected in frozen['source_sha256'].items():
        if sha(source) != expected:
            raise RuntimeError(f'Frozen implementation changed: {source}')
    if config is not None and config != actual_config:
        raise RuntimeError('Seal config does not match frozen protocol')
    result = []
    for unit in actual_config['units']:
        directory = root / 'worlds' / unit['unit']; data_path = directory / 'data.npz'
        data_audit = json.loads((directory / 'data_audit.json').read_text())
        if sha(data_path) != data_audit['data_sha256']:
            raise RuntimeError('Analysis matrix identity changed before label seal')
        label_path = directory / 'analysis_labels.npz'; audit_path = directory / 'analysis_label_seal.json'
        if label_path.exists():
            audit = json.loads(audit_path.read_text())
            if sha(label_path) != audit['labels_sha256'] or audit['protocol_sha256'] != frozen['protocol_sha256']:
                raise RuntimeError('Existing analysis seal identity differs')
            result.append(audit); continue
        with np.load(data_path, allow_pickle=False) as ar:
            ns = ar['analysis_lengths']; matrices = ar['analysis_matrices']; labels = []
            for n, a in zip(ns, matrices):
                a = a[:int(n), :int(n)]
                labels.append(np.stack([core.labels(core.transform(a, action)) for action in core.ACTIONS]))
            np.savez_compressed(label_path, labels=np.stack(labels), analysis_labels=np.stack(labels),
                                lengths=ns, ids=ar['analysis_ids'], orbit_ids=ar['analysis_orbit_ids'],
                                actions=np.asarray(core.ACTIONS), tasks=np.asarray(core.TASKS))
        audit = {'status': 'sealed', 'unit': unit['unit'], 'sealed_utc': datetime.now(timezone.utc).isoformat(),
                 'protocol_sha256': frozen['protocol_sha256'], 'freeze_manifest_sha256': sha(freeze_path),
                 'data_sha256': data_audit['data_sha256'], 'labels_path': str(label_path),
                 'labels_sha256': sha(label_path), 'shape': [len(ns), len(core.ACTIONS), len(core.TASKS)],
                 'labels_generated_after_protocol_freeze': True, 'no_model_or_geometry_calls': True}
        write_json(audit_path, audit); result.append(audit)
    return result
