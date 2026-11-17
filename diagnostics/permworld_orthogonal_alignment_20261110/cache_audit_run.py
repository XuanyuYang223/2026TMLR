"""Read-only existing-cache/provenance audit; no model or probe construction."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'diagnostics/permworld_peaks_probe_20261109'
REP=Path('/mnt/yangx-projects-f/20261106/permworld_repetition')
STANDARD=Path('/mnt/yangx-projects-f/20261109/permworld_peaks_probe/features')
ENCODINGS=['canonical_cycle','one_line']


def read(path):return json.loads(Path(path).read_text())


def main():
    started=time.monotonic();checked={};issues=[];standardized=[];sources=[];data_records=[];checkpoints={}
    def check(path,expected=None):
        path=Path(path);key=str(path)
        if key not in checked:
            before=path.stat()
            with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),('File changed while reading',path)
            checked[key]=dict(path=key,sha256=actual,size_bytes=after.st_size,mtime_ns=after.st_mtime_ns)
        if expected is not None:assert checked[key]['sha256']==expected,('SHA mismatch',path)
        return checked[key]['sha256']

    old_audit=read(OLD/'cache_audit.json')
    previous_records={(r['unit'],r['encoding'],r['task'],r['phase'],r['split']):r for r in old_audit['cache_records']}
    prior_protocol=ROOT/'diagnostics/permworld_repetition_20261106/protocol.json'
    protocol_sha=check(prior_protocol)
    for i in range(2,6):
        unit=f'unit{i:02d}';base=REP/unit;data_path=base/'data.npz';data_audit=read(base/'data_audit.json');check(base/'data_audit.json')
        check(data_path,data_audit['data_sha256'])
        with np.load(data_path,allow_pickle=False) as data:
            sizes={s:len(data[s+'_lengths']) for s in ['validation','test']}
            assert sizes==dict(validation=5376,test=10752)
            assert np.array_equal(np.unique(data['validation_lengths']),np.arange(10,31))
            assert all((data['validation_lengths']==n).sum()==256 for n in range(10,31))
            assert all((data['test_lengths']==n).sum()==512 for n in range(10,31))
        initial_meta=read(base/'initializations/gru/step0.json');check(base/'initializations/gru/step0.json')
        assert initial_meta['actual_initial_identity_verified'] and initial_meta['protocol_sha256']==protocol_sha
        initial_path=base/'initializations/gru/step0.pt';check(initial_path,initial_meta['checkpoint_sha256'])
        checkpoints[str(initial_path)]=dict(path=str(initial_path),sha256=initial_meta['checkpoint_sha256'],role='genuine_saved_step0')
        receipt_path=OLD/'results'/unit/'completion.json';receipt=read(receipt_path);check(receipt_path)
        assert receipt['status']=='completed' and receipt['unit']==unit
        cp_by_model={}
        for enc in ENCODINGS:
            for name in ['initial','peaks','valleys','descents','recoils']:
                task,phase=('peaks','initial') if name=='initial' else (name,'trained')
                if phase=='initial':cp_path=initial_path;cp_sha=initial_meta['checkpoint_sha256'];parameter_sha=initial_meta['parameter_sha256']
                else:
                    cp_path=base/'models'/f'{unit}_gru_{enc}_{task}'/'checkpoint.pt';done_path=cp_path.with_name('completed.json');done=read(done_path);check(done_path)
                    assert done['step']==done['finite_gradient_updates']==20000
                    assert done['architecture']=='gru' and done['encoding']==enc and done['task']==task
                    assert done['data_sha256']==data_audit['data_sha256']
                    assert done['signature']['protocol_sha256']==protocol_sha
                    assert done['actual_initial_identity_verified']
                    assert done['initial_checkpoint_sha256']==initial_meta['checkpoint_sha256']
                    assert done['initial_parameter_sha256']==done['actual_loaded_initial_state_dict_sha256']==initial_meta['parameter_sha256']
                    cp_sha=done['checkpoint_sha256'];parameter_sha=done['parameter_sha256']
                    check(cp_path,cp_sha);checkpoints[str(cp_path)]=dict(path=str(cp_path),sha256=cp_sha,role='fixed20k_trained_specialist')
                cp_by_model[enc,task,phase]=(cp_path,cp_sha,parameter_sha)
                path=STANDARD/unit/f'{enc}-{name}.npz';check(path,receipt['imported_feature_sha256'][path.name])
                with np.load(path,allow_pickle=False) as z:
                    assert str(z['checkpoint_sha256'])==cp_sha
                    assert np.array_equal(z['validation_indices'],np.arange(sizes['validation']))
                    assert np.array_equal(z['test_indices'],np.arange(sizes['test']))
                    shapes={k:list(z[k].shape) for k in z.files}
                    for k in z.files:
                        if k.endswith('_hidden'):
                            n=sizes['validation' if k.startswith('validation') else 'test']
                            assert z[k].shape==(n,256) and z[k].dtype==np.float32
                standardized.append(dict(unit=unit,encoding=enc,name=name,task=task,phase=phase,path=str(path),sha256=checked[str(path)]['sha256'],checkpoint_path=str(cp_path),checkpoint_sha256=cp_sha,receipt_path=str(receipt_path),receipt_sha256=checked[str(receipt_path)]['sha256'],shapes=shapes,row_order='Exact arange(N) for each complete original split',reusable=True))
        plan=[(enc,'initial','peaks','initial',[1,2]) for enc in ENCODINGS]
        plan += [(enc,'trained','valleys','trained',[1,2]) for enc in ENCODINGS]
        plan += [('one_line',task,task,'trained',[1]) for task in ['descents','recoils']]
        for enc,name,task,phase,actions in plan:
            path=base/'features'/f'gru_{enc}_{task}_{phase}_validation.npz';meta_path=path.with_suffix('.json');meta=read(meta_path);check(meta_path)
            sig=meta['signature'];cp_path,cp_sha,parameter_sha=cp_by_model[enc,task,phase]
            expected=dict(checkpoint_sha256=cp_sha,data_sha256=data_audit['data_sha256'],protocol_sha256=protocol_sha,architecture='gru',encoding=enc,task=task,phase=phase,split='validation',point='input end before task token, final_norm',parameter_dtype='float32',forward_autocast=False,engineering=False)
            for key,value in expected.items():assert sig[key]==value,(unit,enc,name,key)
            assert meta['parameter_sha256_before']==meta['parameter_sha256_after']==parameter_sha
            previous=previous_records.get((unit,enc,task,phase,'validation'))
            if previous:
                assert checked[str(meta_path)]['sha256']==previous['metadata_sha256']
                assert meta['sha256']==previous['expected_payload_sha256']
            check(path,meta['sha256'])
            with np.load(path,allow_pickle=False) as z:
                features=z['prefix_final'];assert features.shape==(5376,3,256) and features.dtype==np.float32 and np.isfinite(features).all()
                if name in ['initial','trained']:
                    standard_name='initial' if name=='initial' else 'valleys'
                    with np.load(STANDARD/unit/f'{enc}-{standard_name}.npz',allow_pickle=False) as old:
                        assert np.array_equal(features[:,0],old['validation_0_hidden'])
            sources.append(dict(unit=unit,encoding=enc,output_name=f'{enc}-{name}.npz',task=task,phase=phase,path=str(path),metadata_path=str(meta_path),metadata_sha256=checked[str(meta_path)]['sha256'],payload_sha256=meta['sha256'],data_path=str(data_path),data_sha256=data_audit['data_sha256'],checkpoint_path=str(cp_path),checkpoint_sha256=cp_sha,parameter_sha256=parameter_sha,source_field='prefix_final',source_shape=[5376,3,256],dtype='float32',actions_to_export=actions,output_fields=[f'validation_{a}_hidden' for a in actions],validation_indices='arange(5376), original anchor order',identity_slice_matches_existing_standard_cache=name in ['initial','trained']))
        data_records.append(dict(unit=unit,path=str(data_path),sha256=data_audit['data_sha256'],split_examples=sizes,validation_per_length=256,test_per_length=512,source_data_audit_sha256=checked[str(base/'data_audit.json')]['sha256']))

    # Historical calibration can also use existing arrays, without invoking the ZIP's extractor.
    h=old_audit['historical_reproduction'];index_path=Path(h['index_path']);check(index_path,h['index_sha256']);index=read(index_path)
    table={(r['architecture'],r['encoding'],r['task'],r['condition'],r['split']):r for r in index['records']}
    check(h['data_path'],h['data_sha256']);historical_sources=[]
    old_receipt_path=OLD/'results/archdev001/completion.json';old_receipt=read(old_receipt_path);check(old_receipt_path)
    for enc in ENCODINGS:
        for name in ['initial','peaks','valleys','descents','recoils']:
            path=STANDARD/'archdev001'/f'{enc}-{name}.npz';check(path,old_receipt['imported_feature_sha256'][path.name])
            task,phase=('peaks','initial') if name=='initial' else (name,'trained')
            r=table['gru',enc,task,phase,'validation' if task not in ['descents','recoils'] else 'test']
            check(r['source_checkpoint_path'],r['source_checkpoint_sha256']);checkpoints[r['source_checkpoint_path']]=dict(path=r['source_checkpoint_path'],sha256=r['source_checkpoint_sha256'],role='historical_saved_checkpoint')
            with np.load(path,allow_pickle=False) as z:
                assert str(z['checkpoint_sha256'])==r['source_checkpoint_sha256']
                assert np.array_equal(z['validation_indices'],np.arange(5376)) and np.array_equal(z['test_indices'],np.arange(10752))
            standardized.append(dict(unit='archdev001',encoding=enc,name=name,path=str(path),sha256=checked[str(path)]['sha256'],checkpoint_path=r['source_checkpoint_path'],checkpoint_sha256=r['source_checkpoint_sha256'],reusable=True,row_order='Exact arange(N), full historical validation/test splits'))
    for enc,name,task,phase,actions in plan:
        r=table['gru',enc,task,phase,'validation'];assert r['source_data_sha256']==h['data_sha256']
        field=r['arrays']['prefix_final'];path=Path(field['path']);check(path,field['sha256']);a=np.load(path,mmap_mode='r',allow_pickle=False)
        assert a.shape==(5376,3,256) and a.dtype==np.float32 and r['row_order']=='Original anchor index'
        if name in ['initial','trained']:
            standard_name='initial' if name=='initial' else 'valleys'
            with np.load(STANDARD/'archdev001'/f'{enc}-{standard_name}.npz',allow_pickle=False) as z:assert np.array_equal(a[:,0],z['validation_0_hidden'])
        historical_sources.append(dict(unit='archdev001',encoding=enc,output_name=f'{enc}-{name}.npz',task=task,phase=phase,path=str(path),payload_sha256=field['sha256'],data_path=h['data_path'],data_sha256=h['data_sha256'],checkpoint_path=r['source_checkpoint_path'],checkpoint_sha256=r['source_checkpoint_sha256'],source_field='prefix_final',source_shape=[5376,3,256],actions_to_export=actions,dtype='float32',validation_indices='arange(5376), original anchor order'))

    result=dict(status='passed',captured_utc=datetime.now(timezone.utc).isoformat(),scope='Read-only SHA, shape/order and saved producer-receipt audit. It does not independently recompute activations from neural weights.',git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        standardized_new_unit_files=40,standardized_historical_files=10,all_standardized_files_reusable=True,required_new_validation_sources=len(sources),historical_validation_sources=len(historical_sources),physical_checkpoint_files=len(checkpoints),physical_checkpoints=list(checkpoints.values()),
        data_units=data_records,standardized_features=standardized,validation_sources=sources,historical=dict(possible_without_neural_forward=True,data_path=h['data_path'],data_sha256=h['data_sha256'],index_path=str(index_path),index_sha256=h['index_sha256'],sources=historical_sources),
        missing_required_files=[],binding_or_shape_mismatches=[],files_hashed=checked,unique_files_hashed=len(checked),bytes_hashed=sum(x['size_bytes'] for x in checked.values()),elapsed_seconds=time.monotonic()-started,
        no_model_construction=True,no_probe_fitting=True,no_alignment_fitting=True,no_neural_forward=True,no_old_file_writes=True,
        limitations=['Checkpoint payloads are SHA-verified; no tensor reload or new neural forward is performed. Activation identity relies on producer metadata bound to these checkpoints and data, plus current payload SHA.', 'Existing standardized copies are compared with their completed prior-probe receipts. Newly needed validation action0 is compared exactly with existing copies; actions1/2 are read directly from hash-verified source caches.', 'Historical CPU-extracted pilot values can differ numerically from saved cached features even with identical checkpoint/data/point; calibration must use a prespecified numerical tolerance.'])
    HERE.mkdir(parents=True,exist_ok=True);(HERE/'cache_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    text=['# Existing cache audit','','All 40 standardized feature files for unit02–unit05 and all 10 historical standardized files match their completed prior-probe SHA receipts and can be reused. No feature rebuild is required.','',f'All {len(sources)} required newer-unit validation NPZ sources and {len(historical_sources)} historical validation NPY sources contain `[5376, 3, 256]` FP32 pre-task `final_norm` states. Both encodings use peaks/initial and valleys/trained for the requested target files; one-line descents/recoils add trained complement controls.','',f'All {len(checkpoints)} associated physical checkpoint files and five data files were rehashed. Checkpoint/data/protocol/parameter bindings, full anchor order, and existing validation identity slices agree. Validation has256 and test512 examples per registered length10–30.','',f'Current audit hashed {len(checked)} unique files ({result["bytes_hashed"]:,} bytes) in {result["elapsed_seconds"]:.2f} seconds. This is a named-file audit, not continuous or repository-wide immutability evidence.','', 'No model or probe was constructed or fitted; no forward, backward, alignment fit or old-file write was performed. The ZIP historical extractor is unnecessary for these inputs. Activation provenance relies on the saved producer metadata and current SHA, not an independent forward recomputation.','', 'The JSON records every exact path, expected/current content hash, shape, order, checkpoint binding, and action to export.']
    (HERE/'cache_audit.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['status','standardized_new_unit_files','standardized_historical_files','required_new_validation_sources','historical_validation_sources','physical_checkpoint_files','unique_files_hashed','bytes_hashed','elapsed_seconds']}))


if __name__=='__main__':main()
