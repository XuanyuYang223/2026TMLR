"""Read-only provenance audit for the exceedance/deficiency alignment pilot.

This reads saved checkpoint dictionaries and cached arrays. It constructs no
neural module, performs no forward, probe/alignment fit, or test scoring, and
writes only this new diagnostic directory's audit JSON and Markdown files.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import ast
import hashlib
import json
import subprocess
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REP = ROOT / 'diagnostics/permworld_repetition_20261106'
BASE = ROOT / 'diagnostics/permworld_peaks_probe_20261109/protocol.json'
INDEX = Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations/evaluation/index.json')
ENCODINGS = ('one_line', 'canonical_cycle')
TASKS = ('exceedances', 'deficiencies', 'fixed_points')
ACTIONS = ('identity', 'complement', 'inverse')


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value): return hashlib.sha256(np.asarray(value).tobytes(order='C')).hexdigest()


def state_sha(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        for descriptor in (name, str(a.dtype), str(a.shape)):
            result.update(descriptor.encode())
        result.update(a.tobytes())
    return result.hexdigest()


def definition_location(path, name):
    node = next(n for n in ast.parse(Path(path).read_text()).body if getattr(n, 'name', '') == name)
    return dict(path=str(path), name=name, start_line=node.lineno, end_line=node.end_lineno, sha256=sha(path))


def main():
    began = time.monotonic()
    base = read(BASE)
    rep = read(REP / 'protocol.json')
    old = read(REP / 'original_audit.json')
    index = read(INDEX)
    oldmodels = {(r['architecture'], r['encoding'], r['task']): r for r in old['models']}
    cache_index = {(r['architecture'], r['encoding'], r['task'], r['condition'], r['split']): r for r in index['records']}
    expected, roles, verified = {}, {}, {}
    missing, failures, units = [], [], []
    checks = 0
    label_checks = 0
    token_checks = 0
    initial_equalities = 0
    git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    git_status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=ROOT, text=True)

    def check(value, name, detail=None):
        nonlocal checks
        checks += 1
        if not bool(value):
            failures.append(dict(check=name, detail=detail))

    def add(path, expected_sha=None, role='source', verify_now=False):
        path = str(Path(path))
        if not Path(path).is_file():
            missing.append(dict(path=path, role=role))
            return False
        digest = expected_sha or sha(path)
        if path in expected:
            check(expected[path] == digest, 'consistent expected file hash', path)
        expected[path] = digest
        roles.setdefault(path, set()).add(role)
        if verify_now and path not in verified:
            actual = sha(path)
            check(actual == digest, 'hash before trusted dictionary read', path)
            if actual != digest:
                return False
            verified[path] = actual
        return True

    for path, role in [(BASE, 'source_unit_inventory'), (REP / 'protocol.json', 'original_repetition_protocol'),
                       (REP / 'original_audit.json', 'historical_identity_inventory'), (INDEX, 'historical_activation_index')]:
        add(path, role=role)
    extraction_files = [
        ROOT / 'experiments/permworld_encoding_archdev_20261009/common.py',
        ROOT / 'experiments/permworld_encoding_archdev_20261009/evaluation.py',
        ROOT / 'experiments/permworld_encoding/core.py',
        ROOT / 'experiments/permworld_repetition_20261106/runner.py',
        ROOT / 'external/neurips/src/neurips_permutations/passage.py',
        ROOT / 'experiments/permworld_peaks_probe_20261109/vendor/analyze.py',
        ROOT / 'experiments/permworld_peaks_probe_20261109/vendor/import_repetition.py',
    ]
    for path in extraction_files:
        add(path, role='source_extraction_encoding_or_probe_definition')

    for u in base['units']:
        unit = u['unit']
        root = Path(u['root']) if not u['historical'] else None
        data_path = Path(u['data_path'])
        if not add(data_path, u['data_sha256'], 'world_data', verify_now=True):
            continue
        with np.load(data_path, allow_pickle=False) as f:
            # Preserve original order, all three transformations, all 21 lengths.
            data = {k: f[k].copy() for k in f.files if k.startswith(('validation_', 'test_'))}
            data_schema = {k: dict(shape=list(f[k].shape), dtype=str(f[k].dtype)) for k in f.files}
        if u['historical']:
            initial_record = next(r for r in old['saved_initializations'] if r['architecture'] == 'gru')
            initial_path = Path(initial_record['path'])
            initial_expected = initial_record['sha256']
        else:
            initial_path = root / 'initializations/gru/step0.pt'
            initial_expected = read(initial_path.with_suffix('.json'))['checkpoint_sha256']
            add(root / 'data_audit.json', role='world_generation_audit')
        add(initial_path.with_suffix('.json'), role='saved_actual_initialization_metadata')
        if not add(initial_path, initial_expected, 'saved_actual_initialization', verify_now=True):
            continue
        init_meta = read(initial_path.with_suffix('.json'))
        saved_init = torch.load(initial_path, map_location='cpu', weights_only=False)
        initial_parameter_sha = state_sha(saved_init['model'])
        check(initial_parameter_sha == saved_init['parameter_sha256'] == init_meta['parameter_sha256'], 'actual initial state_dict SHA', unit)
        check(saved_init['architecture'] == 'gru' and saved_init.get('step', 0) == 0, 'actual untrained initialization header', unit)
        check('before' in saved_init['stage'] and not saved_init.get('finite_gradient_updates', 0), 'initial checkpoint pre-training stage', unit)
        check(saved_init['model_config'] == rep['model'], 'model configuration unchanged', unit)
        vocab = saved_init['vocabulary']
        model_config = saved_init['model_config']
        check(len(vocab) == 188 and all(vocab[f'{v:02d}'] == v for v in range(31)), 'numeric token identity', unit)
        stage = saved_init['stage']
        del saved_init
        for split in ('validation', 'test'):
            ns = data[split + '_lengths']
            count = 256 if split == 'validation' else 512
            check(set(ns) == set(range(10, 31)) and all((ns == n).sum() == count for n in range(10, 31)), 'complete fixed-length anchor inventory', [unit, split])
            for n in range(10, 31):
                ids = np.flatnonzero(ns == n)
                p = data[split + '_permutations'][ids, :n]
                check(np.array_equal(np.sort(p, axis=1), np.broadcast_to(np.arange(1, n + 1), p.shape)), 'valid raw permutations', [unit, split, n])
                transformed = (p, n + 1 - p, np.argsort(p, axis=1) + 1)
                for action, q in enumerate(transformed):
                    labels = {'exceedances': (q > np.arange(1, n + 1)).sum(1),
                              'deficiencies': (q < np.arange(1, n + 1)).sum(1),
                              'fixed_points': (q == np.arange(1, n + 1)).sum(1)}
                    for task, values in labels.items():
                        column = rep['tasks'].index(task)
                        check(np.array_equal(values, data[split + '_labels'][ids, action, column]), 'registered count formula', [unit, split, n, action, task])
                        label_checks += 1
                    # Check all numeric tokens and separators against the same
                    # transformed anchors, including canonical cycle closing edges.
                    for encoding in ENCODINGS:
                        tokens = data[f'{split}_{encoding}_input'][ids, action, :2*n+4]
                        numeric = tokens[:, 4:-1:2]
                        check(np.array_equal(np.sort(numeric, axis=1), np.broadcast_to(np.arange(1, n+1), p.shape)), 'encoded numeric token bijection', [unit, split, n, action, encoding])
                        check(np.all(tokens[:, :2] == [vocab['<BOS>'], vocab['<SIZE>']]) and np.all(tokens[:, 2] == vocab[f'{n:02d}']), 'encoded prefix metadata', [unit, split, n, action, encoding])
                        if encoding == 'one_line':
                            decoded = numeric
                            check(np.all(tokens[:, 3] == vocab['<ONE_START>']) and np.all(tokens[:, -1] == vocab['<ONE_END>']) and np.all(tokens[:, 5:-1:2] == vocab[',']), 'one-line boundary/separator tokens', [unit, split, n, action])
                        else:
                            check(np.all(tokens[:, 3] == vocab['<CYCLE_START>']) and np.all(tokens[:, -1] == vocab['<CYCLE_END>']), 'cycle boundary tokens', [unit, split, n, action])
                            decoded = np.empty_like(numeric)
                            # Decode each cycle using the actual separator tokens.
                            for j, (values, separators) in enumerate(zip(numeric, tokens[:, 5:-1:2])):
                                starts = np.r_[0, np.flatnonzero(separators == vocab[';']) + 1]
                                ends = np.r_[starts[1:], n]
                                for start, end in zip(starts, ends):
                                    cycle = values[start:end]
                                    decoded[j, cycle - 1] = np.roll(cycle, -1)
                                check(np.all(np.isin(separators, [vocab[','], vocab[';']])) and all(values[s] == values[s:e].min() for s, e in zip(starts, ends)) and np.all(np.diff(values[starts]) > 0), 'canonical cycle ordering', [unit, split, n, action, int(ids[j])])
                        check(np.array_equal(decoded, q), 'encoded transformation and original anchor order', [unit, split, n, action, encoding])
                        token_checks += 1
        raw, checkpoints, initial_arrays = [], [], {}
        for encoding in ENCODINGS:
            for task in TASKS:
                if u['historical']:
                    model = oldmodels['gru', encoding, task]
                    checkpoint = Path(model['checkpoint_path']['path'])
                    checkpoint_expected = model['checkpoint_path']['sha256']
                    donepath = Path(model['completed_path']['path'])
                else:
                    folder = root / 'models' / f'{unit}_gru_{encoding}_{task}'
                    checkpoint = folder / 'checkpoint.pt'
                    donepath = folder / 'completed.json'
                    checkpoint_expected = read(donepath)['checkpoint_sha256'] if donepath.is_file() else None
                if not add(donepath, role='trained_checkpoint_completion_receipt') or not add(checkpoint, checkpoint_expected, 'trained_checkpoint', verify_now=True):
                    continue
                done = read(donepath)
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                actual_parameter_sha = state_sha(payload['model'])
                check(done['checkpoint_sha256'] == checkpoint_expected, 'checkpoint completion binding', [unit, encoding, task])
                check(payload['step'] == payload['finite_gradient_updates'] == done['step'] == done['finite_gradient_updates'] == 20000, 'trained actual20k endpoint', [unit, encoding, task])
                check(payload['data_sha256'] == done['data_sha256'] == u['data_sha256'] and str(payload['run_id']).endswith(f'_gru_{encoding}_{task}'), 'trained checkpoint data/task binding', [unit, encoding, task])
                check(payload['initial_parameter_sha256'] == initial_parameter_sha, 'same true initialization lineage', [unit, encoding, task])
                if done.get('parameter_sha256'):
                    check(actual_parameter_sha == done['parameter_sha256'], 'trained actual state_dict SHA', [unit, encoding, task])
                checkpoints.append(dict(unit=unit, encoding=encoding, task=task, path=str(checkpoint), sha256=checkpoint_expected,
                    parameter_sha256=actual_parameter_sha, receipt_path=str(donepath), receipt_status=done.get('status'),
                    actual_step=int(payload['step']), actual_finite_gradient_updates=int(payload['finite_gradient_updates']),
                    initial_parameter_sha256=initial_parameter_sha,
                    caveat='Historical receipt status is stale; actual hash-bound20k checkpoint exists.' if done.get('status') == 'missing_authorized' else None))
                del payload
                for phase in ('initial', 'trained'):
                    cp_path = initial_path if phase == 'initial' else checkpoint
                    cp_sha = initial_expected if phase == 'initial' else checkpoint_expected
                    params_sha = initial_parameter_sha if phase == 'initial' else actual_parameter_sha
                    for split in ('validation', 'test'):
                        rows = len(data[split + '_lengths'])
                        if u['historical']:
                            record = cache_index['gru', encoding, task, phase, split]
                            check(record['source_data_sha256'] == u['data_sha256'] and record['source_checkpoint_sha256'] == cp_sha, 'historical cache data/checkpoint binding', [unit, encoding, task, phase, split])
                            check(record['actions'] == list(ACTIONS) and record['row_order'] == 'Original anchor index', 'historical actions and anchor order', [unit, encoding, task, phase, split])
                            check(record['parameter_sha256_before'] == record['parameter_sha256_after'] == params_sha, 'historical extraction parameter invariance', [unit, encoding, task, phase, split])
                            hdesc, pdesc = record['arrays']['prefix_final'], record['arrays']['prediction']
                            path, pred_path = Path(hdesc['path']), Path(pdesc['path'])
                            if not add(path, hdesc['sha256'], 'historical_prefix_final') or not add(pred_path, pdesc['sha256'], 'historical_count_predictions'):
                                continue
                            hidden = np.load(path, mmap_mode='r', allow_pickle=False)
                            prediction = np.load(pred_path, mmap_mode='r', allow_pickle=False)
                            field, prediction_field, digest = None, None, hdesc['sha256']
                            provenance = dict(source_index_path=str(INDEX), source_index_record=dict(architecture='gru', encoding=encoding, task=task, condition=phase, split=split), prediction_sha256=pdesc['sha256'],
                                declared_array_fields=list(record['arrays']), all_declared_arrays_are_prefix_layer_or_output=True)
                        else:
                            path = root / 'features' / f'gru_{encoding}_{task}_{phase}_{split}.npz'
                            meta_path = path.with_suffix('.json')
                            if not add(meta_path, role='cached_activation_metadata'):
                                continue
                            metadata = read(meta_path)
                            signature = metadata['signature']
                            for key, value in dict(architecture='gru', encoding=encoding, task=task, phase=phase, split=split,
                                data_sha256=u['data_sha256'], checkpoint_sha256=cp_sha, engineering=False,
                                point='input end before task token, final_norm', parameter_dtype='float32', forward_autocast=False).items():
                                check(signature.get(key) == value, 'new cache signature/' + key, [unit, encoding, task, phase, split])
                            check(metadata['parameter_sha256_before'] == metadata['parameter_sha256_after'] == params_sha, 'new cache extraction parameter invariance', [unit, encoding, task, phase, split])
                            digest = metadata['sha256']
                            if not add(path, digest, 'new_unit_prefix_and_count_cache'):
                                continue
                            with np.load(path, allow_pickle=False) as archive:
                                hidden, prediction = archive['prefix_final'].copy(), archive['prediction'].copy()
                                declared = list(archive.files)
                            field, pred_path, prediction_field = 'prefix_final', path, 'prediction'
                            provenance = dict(metadata_path=str(meta_path), metadata_sha256=expected[str(meta_path)],
                                declared_array_fields=declared, all_declared_arrays_are_prefix_layer_or_output=all(k in ['prefix_final', 'output_prob', 'prediction'] for k in declared))
                        check(hidden.shape == (rows, 3, 256) and hidden.dtype == np.float32 and np.isfinite(hidden).all(), 'prefix_final shape dtype finiteness', [unit, encoding, task, phase, split])
                        check(prediction.shape == (rows, 3), 'count prediction shape', [unit, encoding, task, phase, split])
                        if phase == 'initial':
                            key = encoding, split
                            if key in initial_arrays:
                                check(np.array_equal(hidden, initial_arrays[key]), 'same actual initial prefix across specialists', [unit, encoding, task, split])
                                initial_equalities += 1
                            else:
                                initial_arrays[key] = np.asarray(hidden).copy()
                        raw.append(dict(unit=unit, encoding=encoding, task=task, phase=phase, split=split,
                            path=str(path), sha256=digest, field=field, shape=list(hidden.shape), dtype=str(hidden.dtype),
                            actions=list(ACTIONS), data_sha256=u['data_sha256'], checkpoint_path=str(cp_path), checkpoint_sha256=cp_sha,
                            parameter_sha256=params_sha, prediction_path=str(pred_path), prediction_field=prediction_field,
                            prediction_shape=list(prediction.shape), prediction_dtype=str(prediction.dtype),
                            row_order='Original anchor index; complete split; arange(N)',
                            anchor_indices_sha256=array_sha(np.arange(rows, dtype=np.int64)),
                            lengths_sha256=array_sha(data[split + '_lengths']),
                            raw_permutations_sha256=array_sha(data[split + '_permutations']), **provenance))
                        del hidden, prediction
        units.append(dict(unit=unit, historical=u['historical'], root=u.get('root'), data_path=str(data_path), data_sha256=u['data_sha256'],
            data_schema=data_schema, validation_examples=5376, test_examples=10752, validation_per_length=256, test_per_length=512,
            raw_features=raw, trained_checkpoints=checkpoints,
            initialization=dict(path=str(initial_path), sha256=initial_expected, parameter_sha256=initial_parameter_sha,
                identity_status='Initial weight identity verified from actual saved state_dict; no reconstruction.',
                stage=stage, checkpoint_metadata_path=str(initial_path.with_suffix('.json')), model_config=model_config,
                shared_between_encodings_and_tasks=True),
            source_position_labels='E_i(x)=1[x_i>i], including all positions; source fitting uses identity validation inputs.',
            target_position_labels='D_j(inv(x))=1[inv(x)_j<j]; j=x_i, so D_{x_i}(inv(x))=E_i(x).',
            inverse_reordering='For source position i use target own-coordinate score at index x_i-1 exactly once; do not transform target labels before its own probe fit.',
            fixed_points='Own identity-input validation labels x_i=i. Inverse input has same fixed points, but its physical token coordinates depend on encoding.',
            token_state_cache='No per-token time axis exists among the audited declared prefix/layer/output cache fields. Frozen checkpoint forwards are needed for the edge-token diagnostic.'))
        print(json.dumps(dict(event='unit_cache_binding_complete', unit=unit, records=len(raw), checkpoints=len(checkpoints))), flush=True)

    def hash_file(item):
        path, digest = item
        p = Path(path); before = p.stat(); actual = sha(p); after = p.stat()
        check(actual == digest and (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'endpoint file identity', path)
        return path, dict(sha256=actual, expected_sha256=digest, size_bytes=after.st_size, mtime_ns=after.st_mtime_ns, roles=sorted(roles[path]), passed=actual == digest)
    with ThreadPoolExecutor(max_workers=2) as pool:
        files = dict(pool.map(hash_file, sorted(expected.items())))
    extraction = dict(
        model=definition_location(extraction_files[0], 'CausalGRU'),
        saved_prefix_extractor=definition_location(extraction_files[1], 'extract'),
        independent_hidden_path=definition_location(extraction_files[1], 'manual_final'),
        batch=definition_location(extraction_files[2], 'batch'),
        encoding=definition_location(extraction_files[2], 'encode'),
        interpretation='final_norm sequence position t is the state after consuming token t. A pre-token state is at t-1, not the input embedding at t. GRU is causal, so task suffix cannot affect earlier states.',
        one_line_value_token='zero-based t=4+2*(i-1), where that token is x_i; before=t-1, after=t. Use the transformed input permutation when extracting inverse-target edges.',
        cycle_value_token='Integer values occur once each at t=4,6,...,2*n+2, in canonical-cycle order. Adjacent cycle values encode directed arcs; the final-to-first arc is established by the cycle delimiter/end. Physical token indices must be bound to decoded cycle arcs explicitly; they are not one-line positions.',
        frozen_forward_policy='Read hash-bound checkpoints; model.eval(); torch.inference_mode(); float32 parameters/activation export and no autocast; hook net.final_norm for full sequence; do not instantiate optimizer. Verify state_dict hash before/after and saved prefix_final agreement at registered tolerance2e-5.',
        cache_search_scope='Declared fields in the existing historical evaluation index and the120 selected current validation/test NPZ records, plus bounded filename search in historical evaluation and current unit02 feature folder. This is not a search of every disk path.')
    result = dict(status='passed' if not missing and not failures else 'partial_or_failed', captured_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only cached-state/checkpoint/data audit for both GRU encodings, exceedances/deficiencies/fixed_points, actual initialization and20k validation/test endpoints; no fits, neural forwards or scoring.',
        git_head=git_head, git_status_at_audit_start=git_status, units=units, unit_count=len(units),
        tasks=list(TASKS), encodings=list(ENCODINGS), actions=list(ACTIONS),
        raw_feature_records=sum(len(u['raw_features']) for u in units), trained_checkpoint_count=sum(len(u['trained_checkpoints']) for u in units),
        actual_initial_checkpoint_count=len(units), actual_initial_prefix_equalities=initial_equalities,
        registered_count_formula_checks=label_checks, encoded_action_anchor_checks=token_checks, checks=checks,
        required_missing_files=missing, binding_mismatches=failures, extraction=extraction,
        files_hashed=files, unique_files_hashed=len(files), bytes_hashed=sum(r['size_bytes'] for r in files.values()),
        elapsed_seconds=time.monotonic()-began, no_neural_forward=True, no_neural_module_construction=True,
        no_probe_fitting=True, no_alignment_fitting=True, no_threshold_selection=True, no_test_scoring=True, no_old_file_writes=True,
        limitations=['Activation values are checked against hash-bound extraction provenance, not independently re-forwarded by this audit.',
            'Historical shared initialization is a saved value-preserving key migration of the actual pre-training state; it is not reconstructed from a seed.',
            'Some historical completion receipts retain missing_authorized status despite actual valid trained20k payloads. Actual checkpoint headers and states are checked and caveats retained.',
            'No token-state cache is declared in the selected source schemas. The source index has per-layer prefix states, not a token time axis.',
            'Main all-length equal1/21 and count-matched sensitivity are analysis decisions beyond this source inventory; no outcome scores were computed.'])
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE/'cache_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines=['# Exceedance/deficiency source-cache audit', '',
        f'Status: **{result["status"]}**. All five units, both GRU encodings, all three specialists and both actual-initial/trained phases were audited without fitting or neural forward.', '',
        f'Found **{result["raw_feature_records"]} logical validation/test cache records**, **{result["trained_checkpoint_count"]} trained checkpoints**, and **{len(units)} actual saved shared initializations**. Initial prefix equality across the three specialists was checked in {initial_equalities} whole-array comparisons.', '',
        f'Validation arrays are `[5376,3,256]`; test arrays are `[10752,3,256]`, FP32, actions identity/complement/inverse. Each n=10–30 contains256validation and512test anchors in original order. Hash verification covers {len(files)} named physical files ({result["bytes_hashed"]:,} bytes).', '',
        '| Unit | Raw records | Trained endpoints | Actual shared initialization |', '|---|---:|---:|---|']
    for unit in units:
        lines.append(f'| {unit["unit"]} | {len(unit["raw_features"])} | {len(unit["trained_checkpoints"])} | verified actual saved state |')
    lines += ['', '## Mathematical and extraction binding', '',
        '`E_i(x)=[x_i>i]` corresponds to `D_{x_i}(inv(x))=[inv(x)_{x_i}<x_i]`. Target own-coordinate scores must be indexed by `x_i-1` once. The audit checked registered counts for all source actions; it did not fit a probe or score this relation.', '',
        'There is no token time axis in the selected saved schemas. Their `layers` field denotes per-layer prefix states. The exact model and extraction definitions, hashes, line positions and causal token interpretation are recorded in JSON. Necessary token diagnostics can hook `final_norm` on frozen eval/inference forwards; before a value token is at t-1 and after is at t. Cycle edge closure requires explicit delimiter-aware indexing.', '',
        'Historical initialization is a saved value-preserving key migration of the actual pre-training state, not a same-seed reconstruction. Stale historical missing-authorized receipt statuses are retained as caveats after validating the actual hash-bound20k payloads.', '',
        f'Missing named files: {len(missing)}. Binding/check failures: {len(failures)}. All prior files were read only; the endpoint hash scope is the named inputs and sources, not every file on disk.']
    (HERE/'cache_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k: result[k] for k in ['status','unit_count','raw_feature_records','trained_checkpoint_count','actual_initial_checkpoint_count','checks','unique_files_hashed','bytes_hashed','elapsed_seconds']}), flush=True)
    if failures or missing:
        raise SystemExit('Source audit found missing files or bindings; see saved cache_audit.json without repairing sources.')


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        main()
