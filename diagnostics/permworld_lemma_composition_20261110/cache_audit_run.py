"""Read-only inventory for cached one-line downstream lemma composition.

No neural model, position probe, orthogonal fit, threshold, or downstream
prediction is constructed here. Only input identities and saved maps are read.
"""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ALIGN=ROOT/'diagnostics/permworld_orthogonal_alignment_20261110'
PROBE=ROOT/'diagnostics/permworld_peaks_probe_20261109'
REP=ROOT/'diagnostics/permworld_repetition_20261106'
TASKS=('peaks','valleys','fixed_points')


def read(path):return json.loads(Path(path).read_text())
def digest_array(x):return hashlib.sha256(np.asarray(x).tobytes(order='C')).hexdigest()
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    started=time.monotonic();cfg=read(ALIGN/'protocol.json');rep=read(REP/'protocol.json')
    original=read(REP/'original_audit.json')
    oldmodels={(r['architecture'],r['encoding'],r['task']):r for r in original['models']}
    indexpath=Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations/evaluation/index.json')
    index=read(indexpath);lookup={(r['architecture'],r['encoding'],r['task'],r['condition'],r['split']):r for r in index['records']}
    expected={};roles={};units=[];failures=[];array_checks=0;label_checks=0
    def add(path,expected_hash=None,role='source'):
        path=str(Path(path));actual=expected_hash or sha(path)
        if path in expected:assert expected[path]==actual,(path,'inconsistent expected SHA')
        expected[path]=actual;roles.setdefault(path,[])
        if role not in roles[path]:roles[path].append(role)
    add(ALIGN/'protocol.json',role='prior_alignment_protocol');add(REP/'protocol.json',role='prior_training_protocol')
    add(indexpath,role='historical_activation_index')
    fpcolumn=rep['tasks'].index('fixed_points')
    for u in cfg['units']:
        name=u['unit'];data_path=Path(u['data_path']);add(data_path,u['data_sha256'],'data')
        with np.load(data_path,allow_pickle=False) as f:
            data={k:f[k].copy() for k in f.files if k.endswith(('_lengths','_permutations','_labels'))}
        for split in ('validation','test'):
            ns=data[split+'_lengths'];assert set(ns)==set(range(10,31))
            expected_n=256 if split=='validation' else 512
            assert all((ns==n).sum()==expected_n for n in range(10,31))
            for n in range(10,31):
                p=data[split+'_permutations'][ns==n,:n];target=np.arange(1,n+1)
                for action,perms in [(0,p),(1,n+1-p)]:
                    labels=(perms==target).sum(1)
                    assert np.array_equal(labels,data[split+'_labels'][ns==n,action,fpcolumn])
                    label_checks+=1
        receipt_path=PROBE/'results'/name/'completion.json';receipt=read(receipt_path);add(receipt_path,role='previous_probe_receipt')
        standardized={};raw=[];cpfiles=[]
        for task in TASKS:
            if u['historical']:
                model=oldmodels['gru','one_line',task]
                cp=Path(model['checkpoint_path']['path']);cp_expected=model['checkpoint_path']['sha256'];donepath=Path(model['completed_path']['path'])
            else:
                folder=Path(u['root'])/'models'/f'{name}_gru_one_line_{task}'
                cp=folder/'checkpoint.pt';donepath=folder/'completed.json';cp_expected=read(donepath)['checkpoint_sha256']
            done=read(donepath)
            assert done['step']==done['finite_gradient_updates']==20000
            assert done['data_sha256']==u['data_sha256'] and done['checkpoint_sha256']==cp_expected
            add(cp,cp_expected,'trained_checkpoint');add(donepath,role='checkpoint_completion_receipt')
            cp_info=dict(task=task,path=str(cp),sha256=cp_expected,metadata_path=str(donepath),
                         parameter_sha256=done.get('parameter_sha256'),receipt_status=done.get('status'),
                         receipt_step=int(done['step']),receipt_finite_gradient_updates=int(done['finite_gradient_updates']))
            if u['historical'] and task=='fixed_points':
                # Read a trusted, hash-bound checkpoint dictionary, never a model.
                assert sha(cp)==cp_expected
                import torch
                payload=torch.load(cp,map_location='cpu',weights_only=False)
                header={k:payload.get(k) for k in ['run_id','step','finite_gradient_updates','data_sha256','architecture','task']}
                assert header['step']==header['finite_gradient_updates']==20000
                assert header['data_sha256']==u['data_sha256']
                assert str(header['run_id']).endswith('_gru_one_line_fixed_points')
                cp_info['actual_checkpoint_header_verified_now']=header
                cp_info['stale_status_note']='Receipt status is historical missing_authorized, while actual hash-verified checkpoint header is trained20k and caches bind to it; status field is not used as absence evidence.'
                del payload
            cpfiles.append(cp_info)
            std=None
            if task in ('peaks','valleys'):
                path=Path(u['features'])/f'one_line-{task}.npz';expected_std=receipt['imported_feature_sha256'][path.name]
                add(path,expected_std,'previous_schema_adapted_features')
                with np.load(path,allow_pickle=False) as f:std={k:f[k].copy() for k in f.files}
                assert str(std['checkpoint_sha256'])==cp_expected
                standardized[task]=dict(path=str(path),sha256=expected_std,checkpoint_sha256=cp_expected,
                    note='Raw FP32 final_norm values in an adapted NPZ schema; no global z-scoring.')
            for split in ('validation','test'):
                nrows=len(data[split+'_lengths'])
                if u['historical']:
                    record=lookup['gru','one_line',task,'trained',split]
                    assert record['source_data_sha256']==u['data_sha256']
                    assert record['source_checkpoint_sha256']==cp_expected
                    assert record['actions']==['identity','complement','inverse'] and record['row_order']=='Original anchor index'
                    desc=record['arrays']['prefix_final'];path=Path(desc['path']);payload=desc['sha256']
                    add(path,payload,'historical_raw_prefix_final')
                    hidden=np.load(path,mmap_mode='r',allow_pickle=False)
                    extra=dict(source_index_path=str(indexpath),source_index_sha256=sha(indexpath))
                else:
                    path=Path(u['root'])/'features'/f'gru_one_line_{task}_trained_{split}.npz'
                    meta_path=path.with_suffix('.json');meta=read(meta_path);sig=meta['signature']
                    for key,value in dict(architecture='gru',encoding='one_line',task=task,phase='trained',split=split,
                            engineering=False,data_sha256=u['data_sha256'],checkpoint_sha256=cp_expected,
                            point='input end before task token, final_norm').items():assert sig[key]==value,(path,key)
                    assert meta['parameter_sha256_before']==meta['parameter_sha256_after']==done['parameter_sha256']
                    payload=meta['sha256'];add(path,payload,'new_unit_raw_prefix_final_NPZ');add(meta_path,role='activation_metadata')
                    with np.load(path,allow_pickle=False) as f:hidden=f['prefix_final'].copy()
                    extra=dict(metadata_path=str(meta_path),metadata_sha256=expected[str(meta_path)])
                assert hidden.shape==(nrows,3,256) and hidden.dtype==np.float32
                if std is not None:
                    assert np.array_equal(std[split+'_indices'],np.arange(nrows));array_checks+=1
                    actions=[0] if split=='validation' or task=='peaks' else [0,1,2]
                    for action in actions:
                        assert np.array_equal(hidden[:,action],std[f'{split}_{action}_hidden'])
                        array_checks+=1
                raw.append(dict(unit=name,task=task,split=split,path=str(path),sha256=payload,field='prefix_final' if path.suffix=='.npz' else None,
                    shape=list(hidden.shape),dtype=str(hidden.dtype),actions=['identity','complement','inverse'],
                    required_actions=[0,1] if task=='peaks' else [0,1,2] if task=='valleys' and split=='validation' else [1] if split=='test' else [0],
                    data_sha256=u['data_sha256'],checkpoint_path=str(cp),checkpoint_sha256=cp_expected,
                    row_order='Original anchor index; complete split; arange(N)',**extra))
                del hidden
        map_prefix=Path(cfg['output'])/'map_audits'/name/'alignment'
        metadata=read(map_prefix.with_suffix('.json'));archive=map_prefix.with_suffix('.npz')
        assert metadata['status']=='completed';add(map_prefix.with_suffix('.json'),role='actual_alignment_map_descriptors')
        add(archive,metadata['parameter_archive_sha256'],'actual_saved_orthogonal_maps')
        maps=[]
        with np.load(archive,allow_pickle=False) as f:
            for item in metadata['maps']:
                if item.get('encoding')!='one_line' or item.get('phase')!='trained':continue
                key=f'R_{item["call"]:03d}';q=f[key]
                assert q.shape==(256,256) and q.dtype==np.float64
                assert digest_array(q)==item['map_sha256']
                err=float(np.max(np.abs(q.T@q-np.eye(256))));assert err<=2e-12
                maps.append(dict(fit_condition=item['fit_condition'],key=key,map_sha256=item['map_sha256'],cross_sha256=item['cross_sha256'],maximum_orthogonality_error=err))
        assert [r['fit_condition'] for r in maps]==['identity','complement','inverse',*[f'shuffle_{j}' for j in range(1026,1034)]]
        previous_validation=Path(cfg['output'])/'validation'/name/'one_line-trained.npz'
        previous_adaptation=read(previous_validation.parent/'adaptation.json')
        prev=next(r for r in previous_adaptation['files'] if r['output_path']==str(previous_validation))
        add(previous_validation,prev['output_sha256'],'previous_target_validation_complement_inverse')
        add(previous_validation.parent/'adaptation.json',role='previous_validation_adaptation_receipt')
        units.append(dict(unit=name,historical=u['historical'],data_path=str(data_path),data_sha256=u['data_sha256'],
            validation_examples=5376,test_examples=10752,validation_per_length=256,test_per_length=512,
            raw_features=raw,standardized_features=standardized,trained_checkpoints=cpfiles,
            alignment=dict(archive_path=str(archive),archive_sha256=metadata['parameter_archive_sha256'],metadata_path=str(map_prefix.with_suffix('.json')),
                maps=maps,raw_row_vector_direction='target to source: (h_valleys_C - mu_C) @ R + mu_x',
                previous_validation_path=str(previous_validation),centering_means_serialized=False,
                mean_reconstruction='For each length use exactly original read_length probe-fit mask; mu_x=mean(peaks validation action0),mu_C=mean(valleys validation action1),FP64. Use mu_C for every map on fixed complement input.'),
            source_probe_coefficients_saved=False,
            probe_reconstruction='Rebuild unchanged vendor fit_probe on peaks validation identity and its own interior peak labels using frozen fit/dev masks and source-only alpha selection. Scalers/coefficients were not serialized.',
            route_semantics=dict(input='pi = C(x)',first_peak='peaks specialist test action1, scored against peak positions of pi',
                same_model_second='peaks specialist test action0; peak(x) positions equal valleys(pi)',
                aligned_second='valleys specialist test action1 mapped by saved trained one-line map; one source peaks probe',
                fixed_point_fit='fixed_points specialist validation action0 with own fixed-point labels for x including endpoints',
                fixed_point_test='fixed_points specialist test action1, labels fixed points of pi=C(x) only for scoring'),
            fixed_point_label_column=fpcolumn))
    files={}
    def check(item):
        path,digest=item;p=Path(path);before=p.stat();actual=sha(p);after=p.stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        assert actual==digest,(path,'SHA mismatch')
        return path,dict(sha256=actual,size_bytes=after.st_size,mtime_ns=after.st_mtime_ns,roles=roles[path])
    with ThreadPoolExecutor(max_workers=2) as pool:
        for path,item in pool.map(check,sorted(expected.items())):files[path]=item
    result=dict(status='passed',captured_utc=datetime.now(timezone.utc).isoformat(),
        scope='Read-only current payload/metadata/map audit for existing one-line GRU downstream composition; no fitting or predictions.',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),units=units,
        unit_count=len(units),trained_checkpoint_count=15,raw_feature_records=30,
        fixed_point_count_formula_checks=label_checks,existing_schema_array_equality_checks=array_checks,
        reused_trained_one_line_maps=55,missing_required_files=[],binding_mismatches=[],
        files_hashed=files,unique_files_hashed=len(files),bytes_hashed=sum(v['size_bytes'] for v in files.values()),elapsed_seconds=time.monotonic()-started,
        no_neural_forward=True,no_probe_fitting=True,no_alignment_fitting=True,no_threshold_selection=True,no_downstream_predictions=True,no_old_file_writes=True,
        limitations=['Activation identities are checked against existing extraction provenance and payload hashes, not independently re-forwarded.',
            'Saved rotations do not include per-length means or probe coefficients/scalers. Those must be reconstructed from unchanged source-only validation protocol; no new rotation fit is needed.',
            'Original standardized peaks files lack test action1. The raw source prefix_final contains it and is required for both routes shared first activation.',
            'Fixed-point test action1 concerns fixed points of C(x), not of x. Its new source probe must fit only own identity-input fixed-point position labels.',
            'All wrong/shuffled maps on the common valleys complement input must use complement centering, matching the previous experiment.'])
    HERE.mkdir(parents=True,exist_ok=True);(HERE/'cache_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Downstream-composition cache audit','',
        f'All five units have the required one-line GRU payloads: **30 raw validation/test feature records**, **15 trained checkpoints**, and **55 saved trained one-line maps**. No missing file or binding mismatch was found.',
        '',f'Current SHA verification covers **{len(files)} physical files**, {result["bytes_hashed"]:,} bytes. Raw feature shapes are validation `[5376,3,256]` and test `[10752,3,256]`, FP32, in original anchor order. All 21 lengths have 256 validation and 512 test anchors.',
        '',f'Fixed-point counts from position labels match the registered task column `{fpcolumn}` in {label_checks} per-length/action checks. Existing peak/valley adapted identity/test arrays match raw source slices in {array_checks} array/order checks.',
        '', '| Unit | Peak raw C available | Fixed-point validation I/test C | Saved trained R keys |','|---|---|---|---|']
    for u in units:lines.append(f'| {u["unit"]} | yes | yes, including endpoints | R_034 identity, R_035 complement, R_036 inverse, R_037–R_044 shuffles |')
    lines += ['', '## Reuse and interpretation','',
        'The main input is `pi=C(x)`. The shared first activation is `peaks(C(x))`. The two-call same-model baseline uses `peaks(x)` for the valley mask; the two-model route maps `valleys(C(x))` to source peaks coordinates and uses the same source probe. All mapping controls use that same valleys input.',
        '', 'The fixed-point specialist fits on identity validation inputs `x` with its own fixed-point labels, and is evaluated on cached action 1 for `C(x)`. True fixed counts are not needed as input to prediction or composition.',
        '', 'Only rotations and shuffle permutations were saved previously. Per-length centering and ridge scalers/coefficients were not serialized. Rebuild probes with the frozen source-only fitter and reconstruct means from the original probe-fit rows; use complement target means for every map applied to the common complement target input.',
        '', 'The former “standardized” files are schema-adapted raw FP32 states. No new global normalization should be applied. Activation provenance and SHA were verified; no neural forward, probe, alignment fit, threshold selection or downstream scoring was run by this audit.']
    (HERE/'cache_audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['units','files_hashed','limitations']}))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
