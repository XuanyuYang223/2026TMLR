"""Endpoint protection and bounded independent saved-output verification.

Run only after all five token units finish. No model forward or fit is made.
Large token shards are checked by recorded extraction receipts, file headers,
and fixed first-two-anchor numerical reads at n=10/20/30, not rehashed in full.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
OUT = Path('/mnt/yangx-projects-f/20261110/permworld_exceedance_alignment')
TOKEN = OUT / 'token_diagnostic'
PV = ROOT / 'diagnostics/permworld_orthogonal_alignment_20261110'
LENGTHS = tuple(range(10, 31))
SELECTED_LENGTHS = (10, 20, 30)
ENCODINGS = ('canonical_cycle', 'one_line')
CONDITIONS = dict(source=('exceedances', 'trained', 0), target_identity=('deficiencies', 'trained', 0),
    target_complement=('deficiencies', 'trained', 1), target_inverse=('deficiencies', 'trained', 2),
    initial_inverse=('deficiencies', 'initial', 2), fixed_inverse=('fixed_points', 'trained', 2))
SEEDS = tuple(range(1026, 1034))
TOLERANCE = 1e-12
TOKEN_TOLERANCE = 2e-12  # Existing frozen scalar/map serialization tolerance.


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def array_sha(value): return hashlib.sha256(np.asarray(value).tobytes(order='C')).hexdigest()


def rows(path):
    with Path(path).open(newline='') as stream: return list(csv.DictReader(stream))


def write(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); temporary.replace(path)


class Checks:
    def __init__(self): self.count = 0; self.failures = []; self.maximum = 0.
    def ok(self, condition, label):
        self.count += 1
        if not bool(condition): self.failures.append(dict(label=label))
    def close(self, actual, expected, label, tolerance=TOLERANCE):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        self.count += 1
        if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
            self.failures.append(dict(label=label, reason='shape or undefined support differs')); return
        finite = np.isfinite(a) & np.isfinite(b)
        difference = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.
        self.maximum = max(self.maximum, difference)
        if difference > tolerance or not np.array_equal(np.isinf(a), np.isinf(b)):
            self.failures.append(dict(label=label, maximum_difference=difference, tolerance=tolerance))


def pv_receipts(config, start, checks, expected, scopes):
    """All prior trained scalar reproductions; no probe/map refitting."""
    summaries = []
    for unit in config['units']:
        name = unit['unit']; receipt = read(HERE / 'results' / name / 'pv_score_rebuild.json')
        checks.ok(receipt['status'] == 'completed' and receipt['protocol_sha256'] == sha(HERE/'protocol.json'), name+'/PV receipt binding')
        checks.ok(receipt['data_sha256'] == unit['data_sha256'], name+'/PV data binding')
        checks.ok(receipt['saved_rotations_reused_byte_for_byte'] and receipt['no_new_mapping_fit'] and receipt['no_neural_forward'], name+'/PV fitted scope')
        original = {(r['encoding'], int(r['length']), r['method'], r['support']): r
                    for r in rows(PV/'results'/name/'by_length.csv') if r['phase'] == 'trained'}
        copied = {(r['encoding'], int(r['length']), r['method'], r['support']): r
                  for r in rows(receipt['reproduced_by_length_path'])}
        recorded = {(r['encoding'], int(r['length']), r['method'], r['support']): r for r in receipt['checks']}
        checks.ok(len(original) == len(copied) == len(recorded) == receipt['check_count'] == 1848, name+'/PV1848 complete cells')
        checks.ok(set(original) == set(copied) == set(recorded), name+'/PV complete registered keyset')
        for key in sorted(set(original) & set(copied) & set(recorded)):
            prior, fresh, check = original[key], copied[key], recorded[key]
            checks.close(fresh['auc'], prior['auc'], name+'/PV copied scalar')
            checks.close(check['actual'], prior['auc'], name+'/PV actual scalar')
            checks.close(check['previous'], prior['auc'], name+'/PV prior scalar')
            checks.close(check['absolute_error'], abs(float(check['actual'])-float(check['previous'])), name+'/PV error arithmetic')
            checks.ok(int(prior['n']) == int(fresh['n']) and int(prior['finite']) == int(fresh['finite']), name+'/PV support counts')
        for path, desc in receipt['source_bindings'].items():
            if path in expected:
                checks.ok(expected[path] == desc['sha256'], name+'/PV binding against start expectation')
            else:
                expected[path] = desc['sha256']
                scopes[path] = 'additional_PV_execution_binding_not_original_start_snapshot'
        checks.ok(receipt['maximum_auroc_error'] <= TOLERANCE, name+'/PV original tolerance')
        summaries.append(dict(unit=name, reproduced_trained_cells=1848, source_bindings=len(receipt['source_bindings']),
                              maximum_recorded_error=receipt['maximum_auroc_error']))
    return summaries


def independent_edges(permutation, encoding):
    """Edge offsets from a separate direct traversal, not producer oracles."""
    p = list(map(int, permutation)); n = len(p)
    if encoding == 'one_line':
        after = 4 + 2*np.arange(n)
        return dict(before=after-1, after=after, kind=np.zeros(n, np.int8))
    unseen = set(range(1, n+1)); cycles = []
    while unseen:
        first = min(unseen); cycle = []; current = first
        while current in unseen:
            unseen.remove(current); cycle.append(current); current = p[current-1]
        cycles.append(cycle)
    before = np.empty(n, np.int64); after = before.copy(); kind = np.empty(n, np.int8)
    cursor = 4
    for cycle in cycles:
        numeric_positions = dict(zip(cycle, cursor + 2*np.arange(len(cycle))))
        for k, source in enumerate(cycle):
            if k+1 < len(cycle):
                at = numeric_positions[cycle[k+1]]; code = 1
            else:
                at = numeric_positions[source]+1; code = 3 if len(cycle) == 1 else 2
            after[source-1], before[source-1], kind[source-1] = at, at-1, code
        cursor += 2*len(cycle)
    return dict(before=before, after=after, kind=kind)


def independent_auc(scores, targets, selected=None):
    """Positive/negative pair comparisons, including half credit for ties."""
    score = np.asarray(scores); truth = np.asarray(targets, bool)
    valid = np.isfinite(score) if selected is None else selected & np.isfinite(score)
    result = np.full(len(score), np.nan)
    for i in range(len(score)):
        positive = score[i, valid[i] & truth[i]]; negative = score[i, valid[i] & ~truth[i]]
        if len(positive) and len(negative):
            result[i] = ((positive[:,None] > negative[None,:]).sum()+.5*(positive[:,None] == negative[None,:]).sum())/(len(positive)*len(negative))
    return result


def scalar_predict(states, probe, role):
    value = np.array(states, copy=True)
    value -= probe[role+'_scaler_mean'].astype(value.dtype)
    value /= probe[role+'_scaler_scale'].astype(value.dtype)
    return value.astype(np.float64) @ probe[role+'_ridge_coef'] + probe[role+'_ridge_intercept']


def token_checks(config, checks):
    """Metadata all shards; numerical first2 anchors at fixed n10/20/30."""
    from experiments.permworld_exceedance_alignment_20261110.shared import vendor
    _, read_length, _, _, _, _, shuffled = vendor()
    frozen_path = HERE/'token_results/freeze.json'; frozen = read(frozen_path)
    complete = read(HERE/'token_results/run_completion.json')
    names = [u['unit'] for u in config['units']]
    checks.ok(complete['status'] == 'complete' and complete['units'] == names, 'all five token units finished')
    checks.ok(complete['token_freeze_sha256'] == sha(frozen_path), 'token final freeze binding')
    report = []; metadata_records = 0; shard_headers = 0; selected_score_files = 0; numerical_states = 0
    for unit in config['units']:
        name = unit['unit']; receipt = read(HERE/'token_results'/name/'completion.json')
        checks.ok(receipt['status'] == 'complete' and receipt['token_freeze_sha256'] == sha(frozen_path), name+'/token unit complete')
        checks.ok(receipt['data_sha256'] == unit['data_sha256'] and receipt['no_base_training'], name+'/token unit data/scope')
        # Small per-unit scalar tables are hash-checked, not state shards.
        for path, digest in receipt['result_sha256'].items():
            checks.ok(sha(path) == digest, name+'/token scalar result SHA')
        with np.load(unit['data_path'], allow_pickle=False) as z:
            data = {key:z[key].copy() for key in z.files if key.endswith(('_lengths','_permutations'))}
        by_length = {(r['encoding'], int(r['n']), r['support'], r['stratum'], r['method'], r['point']):r
                     for r in rows(HERE/'token_results'/name/'by_length.csv')}
        fit_table = rows(HERE/'token_results'/name/'probe_fit.csv')
        raw_unit = next(r for r in read(HERE/'cache_audit.json')['units'] if r['unit'] == name)
        for enc in ENCODINGS:
            for condition, (task, phase, action) in CONDITIONS.items():
                for split in ('validation','test'):
                    folder = TOKEN/'features'/name/enc/condition/split
                    metadata = read(folder/'completion.json'); metadata_records += 1
                    desc = next(r for r in raw_unit['raw_features'] if r['encoding'] == enc and r['task'] == task and r['phase'] == phase and r['split'] == split)
                    checks.ok(metadata['status'] == 'complete' and metadata['data_sha256'] == unit['data_sha256'], name+'/token extraction data')
                    checks.ok(metadata['checkpoint_sha256'] == desc['checkpoint_sha256'] and metadata['parameter_sha256_before'] == metadata['parameter_sha256_after'] == desc['parameter_sha256'], name+'/token extraction checkpoint/parameter invariance')
                    checks.ok(metadata['oracle_sha256'] == frozen['source_sha256'][str(ROOT/'experiments/permworld_exceedance_alignment_20261110/oracles.py')], name+'/token extraction coordinate-code binding')
                    checks.ok(metadata['no_optimizer'] and metadata['no_backward'] and metadata['maximum_prefix_error'] <= 2e-5 and metadata['tolerance'] == 2e-5, name+'/token extraction scope/prefix')
                    checks.ok([s['n'] for s in metadata['shards']] == list(LENGTHS), name+'/token shard inventory')
                    for shard in metadata['shards']:
                        indices = np.flatnonzero(data[split+'_lengths'] == shard['n'])
                        checks.ok(shard['anchor_indices_sha256'] == array_sha(indices), name+'/token shard anchor order')
                        value = np.load(folder/shard['filename'], mmap_mode='r', allow_pickle=False)
                        checks.ok(list(value.shape) == shard['shape'] == [len(indices),shard['n'],2,256] and value.dtype == np.float32, name+'/token shard header')
                        checks.ok((folder/shard['filename']).stat().st_size == value.offset+value.nbytes, name+'/token shard file size')
                        checks.ok(shard['maximum_prefix_error'] <= 2e-5 and len(shard['sha256']) == 64, name+'/token recorded shard integrity/prefix')
                        del value; shard_headers += 1
            parameter_folder = TOKEN/'parameters'/name/enc
            with np.load(parameter_folder/'maps.npz', allow_pickle=False) as z:
                maps = dict(z)
            checks.ok(set(maps) == {'target_identity','target_complement','target_inverse','initial_inverse','fixed_inverse',*(f'shuffle_{s}' for s in SEEDS)}, name+'/token13mapinventory')
            for key, rotation in maps.items():
                checks.ok(np.max(np.abs(rotation.T@rotation-np.eye(256))) <= 2e-12, name+'/token map orthogonality')
            # Check every saved alpha decision using the frozen scalar-fit table.
            for n in LENGTHS:
                for role in ('source','target'):
                    grid = [r for r in fit_table if r['encoding'] == enc and int(r['n']) == n and r['role'] == role]
                    checks.ok([float(r['alpha']) for r in grid] == [1.,10.,100.,1000.], name+'/token source-only alpha inventory')
                    values = np.asarray([float(r['dev_per_anchor_auc']) for r in grid]); chosen = int(np.nanargmax(values))
                    checks.ok([r['selected'] for r in grid] == ['True' if i == chosen else 'False' for i in range(4)], name+'/token dev-only alpha choice')
            for n in SELECTED_LENGTHS:
                ids = np.flatnonzero(data['test_lengths'] == n); chosen = ids[:2]; p = data['test_permutations'][ids,:n]
                with np.load(TOKEN/'scores'/name/enc/f'n{n}.npz', allow_pickle=False) as z: saved = dict(z)
                with np.load(parameter_folder/f'n{n}_probes.npz', allow_pickle=False) as z: probe = dict(z)
                with np.load(parameter_folder/f'n{n}_means_priors.npz', allow_pickle=False) as z: means = dict(z)
                selected_score_files += 1
                methods = list(map(str,saved['methods'])); labels = p > np.arange(1,n+1)
                checks.ok(np.array_equal(saved['anchor_indices'],ids) and np.array_equal(saved['labels'],labels) and np.array_equal(saved['target_position'],p), name+'/token score ID/label/j binding')
                source_coords = {k:np.stack([independent_edges(row,enc)[k] for row in p]) for k in ('before','after','kind')}
                inverse = np.argsort(p,axis=1)+1
                native_coords = {k:np.stack([independent_edges(row,enc)[k] for row in inverse]) for k in ('before','after','kind')}
                target_coords = {k:np.take_along_axis(v,p-1,axis=1) for k,v in native_coords.items()}
                for side, coords in [('source',source_coords),('target',target_coords)]:
                    for key, field in [('before','before_token'),('after','after_token'),('kind','kind')]:
                        checks.ok(np.array_equal(saved[f'{side}_{field}'],coords[key]), name+'/independent token edge coordinates/'+side+'/'+key)
                states = {}
                for condition in ('source','target_inverse','initial_inverse','fixed_inverse'):
                    path = TOKEN/'features'/name/enc/condition/'test'/f'n{n}.npy'
                    value = np.load(path,mmap_mode='r',allow_pickle=False)
                    states[condition] = np.array(value[:2],copy=True); del value
                    if condition != 'source':
                        states[condition] = np.take_along_axis(states[condition],(p[:2]-1)[...,None,None],axis=1)
                    numerical_states += 1
                x, target = states['source'], states['target_inverse']; mx = means['source_mean']; mi = means['target_inverse_mean']
                reconstructed = dict(source_own=scalar_predict(x,probe,'source'), direct=scalar_predict(target,probe,'source'),
                    means_only=scalar_predict(target-mi+mx,probe,'source'), target_own=scalar_predict(target,probe,'target'))
                for map_name, rotation in maps.items():
                    condition = map_name if map_name in ('initial_inverse','fixed_inverse') else 'target_inverse'
                    mapped = (np.asarray(states[condition],np.float64)-means[condition+'_mean'])@rotation+mx
                    reconstructed['fit_'+map_name] = scalar_predict(mapped,probe,'source')
                for method, value in reconstructed.items():
                    checks.close(value,saved['scores'][methods.index(method),:2],name+'/token explicit coefficient/map score/'+method,TOKEN_TOLERANCE)
                checks.ok(np.array_equal(np.isfinite(saved['scores'][:4]), np.ones_like(saved['scores'][:4],bool)), name+'/token own/direct finite')
                # AUROC is recomputed by pair comparisons, independent of the
                # producer's rankdata implementation, for all512 saved anchors
                # at these selected lengths and registered strata/supports.
                strata = {'all':np.ones_like(labels,bool),'both_internal':(source_coords['kind']==1)&(target_coords['kind']==1)}
                for side, coords in [('source',source_coords),('target',target_coords)]:
                    for code, kind in enumerate(('one_line_value','cycle_internal','cycle_closure','cycle_singleton')):
                        strata[f'{side}_{kind}'] = coords['kind']==code
                for m, method in enumerate(methods):
                    for point, point_name in enumerate(('before','after')):
                        score = saved['scores'][m,:,:,point]
                        for stratum, mask in strata.items():
                            auc = independent_auc(score,labels,mask)
                            for support in ('eligible','matched'):
                                keep = saved[support]; finite = auc[keep][np.isfinite(auc[keep])]
                                row = by_length[enc,n,support,stratum,method,point_name]
                                checks.close(row['per_anchor_auc'],finite.mean() if len(finite) else np.nan,name+'/independent token AUROC')
                                checks.ok(int(row['finite_anchors']) == len(finite) and int(row['anchors']) == int(keep.sum()), name+'/token AUROC coverage')
                del saved, states
        report.append(dict(unit=name, selected_numerical_lengths=list(SELECTED_LENGTHS), selected_anchors_per_length=2))
    return dict(unit_reports=report, extraction_metadata_records=metadata_records, shard_headers_verified=shard_headers,
        selected_score_files=selected_score_files, fixed_subset_state_arrays_read=numerical_states,
        whole_token_state_payloads_rehashed=0,
        numerical_scope='First2anchors at n10/20/30, both encodings/all5units for explicit score reconstruction; all512anchors at those30 scorefiles for independent pair-comparison AUROC/strata. All2520 token shard headers and120 extraction receipts checked; state SHA digests rely on extraction-time records, not another full read.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    if not (HERE/'token_results/run_completion.json').exists():
        raise SystemExit('Token stage pending: run this helper only after all five token units finish.')
    began = datetime.now(timezone.utc).isoformat(); clock = time.monotonic()
    config, freeze, start = read(HERE/'protocol.json'), read(HERE/'freeze.json'), read(HERE/'protection_start.json')
    checks = Checks(); expected = dict(start['files']); scopes = {p:'original_start_snapshot' for p in expected}
    for path, digest in freeze['source_sha256'].items():
        absolute = str(ROOT/path)
        if absolute in expected: checks.ok(expected[absolute]==digest,'main source/start SHA consistent')
        expected[absolute] = digest; scopes.setdefault(absolute,'main_frozen_source')
    expected[str(HERE/'protocol.json')] = freeze['protocol_sha256']; scopes.setdefault(str(HERE/'protocol.json'),'main_frozen_protocol')
    checks.ok(read(HERE/'main_run_completion.json')['status']=='completed','main ED+PV execution completed')
    summary = read(HERE/'summary/completion.json')
    checks.ok(summary['status']=='completed_saved_position_alignment_summary' and summary['protocol_sha256']==freeze['protocol_sha256'],'main summary completed/bound')
    main_audit = read(HERE/'main_saved_score_audit.json')
    checks.ok(main_audit['status']=='passed' and not main_audit['failures'] and main_audit['absolute_tolerance']==TOLERANCE,'independent main-score audit passed')
    pv = pv_receipts(config,start,checks,expected,scopes)
    token = token_checks(config,checks)
    token_freeze = read(HERE/'token_results/freeze.json')
    for path,digest in token_freeze['source_sha256'].items():
        if path in expected: checks.ok(expected[path]==digest,'token/main frozen SHA consistent')
        expected[path]=digest; scopes.setdefault(path,'separately_frozen_token_source')
    print(json.dumps(dict(event='numeric_checks_finished',checks=checks.count,failures=len(checks.failures))),flush=True)
    def hash_one(item):
        path,digest=item; file=Path(path); before=file.stat(); actual=sha(file); after=file.stat()
        return dict(path=path,scope=scopes[path],expected_sha256=digest,actual_sha256=actual,size_bytes=after.st_size,
                    passed=actual==digest and (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns))
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as pool: files=list(pool.map(hash_one,sorted(expected.items())))
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    end_tracked=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=no'],cwd=ROOT,text=True)
    end_status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True)
    audit_start_status=read(HERE/'cache_audit.json')['git_status_at_audit_start']
    initial_tracked=[r for r in audit_start_status.splitlines() if not r.startswith('?? ')]
    checks.ok(not initial_tracked and not end_tracked,'same HEAD tracked files clean at recorded endpoints')
    checks.ok(head==start['git_head'],'Git HEAD unchanged')
    scope_counts={scope:sum(r['scope']==scope for r in files) for scope in sorted(set(scopes.values()))}
    protection=dict(status='passed' if all(r['passed'] for r in files) and head==start['git_head'] and not end_tracked else 'failed',
        started_utc=began,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Endpoint SHA check of original550 named start paths, frozen sources, and additional PV execution source bindings. Additional PV paths are not retroactively part of the start snapshot. Not continuous monitoring or every file.',
        original_start_files=len(start['files']),scope_counts=scope_counts,unique_files_hashed=len(files),
        bytes_hashed=sum(r['size_bytes'] for r in files),files=files,failed_files=[r for r in files if not r['passed']],
        git_head_start=start['git_head'],git_head_end=head,git_head_unchanged=head==start['git_head'],
        recorded_start_tracked_changes=initial_tracked,end_tracked_status=end_tracked,end_status_including_untracked=end_status,
        whole_token_state_payloads_rehashed=0,no_old_file_writes=True)
    provenance_paths=[HERE/'finalize_audit.py',HERE/'audit_saved_main.py',HERE/'main_saved_score_audit.json',HERE/'token_readonly_review.json',HERE/'freeze.json',HERE/'token_results/freeze.json',
        ROOT/'tests/test_permworld_exceedance_token_diagnostic.py',HERE/'token_results/synthetic_pytest.log']
    final=dict(status='passed' if not checks.failures and protection['status']=='passed' else 'failed',
        completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks.count,failures=checks.failures,
        maximum_checked_difference=checks.maximum,main_PV_tolerance=TOLERANCE,token_serialization_tolerance=TOKEN_TOLERANCE,
        independent_main_audit=main_audit,PV_reproduction_receipts=pv,token_verification=token,
        provenance_snapshot={str(p):sha(p) for p in provenance_paths},
        no_neural_forward=True,no_probe_fitting=True,no_alignment_fitting=True,no_new_hyperparameters=True,
        scope='Read-only source/protection endpoint audit plus saved receipt arithmetic and bounded independent token score checks. Validation checks are technical checks, not independent statistical repetitions.',
        elapsed_seconds=time.monotonic()-clock)
    write(HERE/'protection_end.json',protection);write(HERE/'final_verification.json',final)
    print(json.dumps(dict(protection_status=protection['status'],verification_status=final['status'],checks=checks.count,failures=len(checks.failures),maximum_difference=checks.maximum,files_hashed=len(files),elapsed_seconds=final['elapsed_seconds'])),flush=True)
    if final['status']!='passed':raise SystemExit('Final audit failed; inspect failures without changing frozen results/tolerances.')


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
