"""Independently check saved validation slices, bindings and feature reuse."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_orthogonal_alignment')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    begun=time.monotonic();audit=read(HERE/'cache_audit.json');assert audit['status']=='passed'
    source_records=audit['validation_sources']+audit['historical']['sources']
    expected={(r['unit'],r['output_name']):r for r in source_records};assert len(expected)==30
    records=[];reused=[];actions_checked=0
    for unit in ['archdev001','unit02','unit03','unit04','unit05']:
        folder=OUT/'validation'/unit;sidecar=read(folder/'adaptation.json');receipt=read(HERE/'results'/unit/'completion.json')
        assert sidecar['unit']==receipt['unit']==unit and sidecar['no_forward']
        assert receipt['status']=='completed' and receipt['no_neural_forward'] and receipt['no_base_training']
        assert receipt['features']['reused'] is True
        assert receipt['validation_adaptation']==sidecar['files']
        actual={Path(r['output_path']).name:r for r in sidecar['files']}
        assert len(actual)==6 and set(actual)=={name for u,name in expected if u==unit}
        assert {p.name for p in folder.glob('*.npz')}==set(actual)
        for name,meta in actual.items():
            source=expected[unit,name];path=folder/name;source_path=Path(source['path'])
            assert str(path)==meta['output_path']
            assert meta['source_path']==source['path']
            assert meta['source_sha256']==source['payload_sha256']
            assert meta['data_sha256']==source['data_sha256']
            assert meta['checkpoint_sha256']==source['checkpoint_sha256']
            assert meta['actions']==source['actions_to_export']
            assert meta['validation_rows']==5376 and meta['dtype']=='float32'
            assert meta['original_order_verified'] and meta['raw_states_no_additional_normalization']
            assert sha(path)==meta['output_sha256']
            # Current source-content SHA is covered by the independent endpoint
            # audit. Here we independently compare every exported float bit.
            if source_path.suffix=='.npz':
                with np.load(source_path,allow_pickle=False) as z:original=z['prefix_final']
            else:original=np.load(source_path,allow_pickle=False,mmap_mode='r')
            assert original.shape==(5376,3,256) and original.dtype==np.float32
            per_action=[]
            with np.load(path,allow_pickle=False) as z:
                assert set(z.files)=={'validation_indices','checkpoint_sha256',*[f'validation_{a}_hidden' for a in source['actions_to_export']]}
                assert np.array_equal(z['validation_indices'],np.arange(5376))
                assert z['validation_indices'].dtype==np.int64
                assert z['checkpoint_sha256'].shape==() and str(z['checkpoint_sha256'])==source['checkpoint_sha256']
                for action in source['actions_to_export']:
                    current=z[f'validation_{action}_hidden'];assert current.shape==(5376,256) and current.dtype==np.float32
                    assert np.array_equal(current.view(np.uint32),original[:,action].view(np.uint32)),(unit,name,action,'float-bit mismatch')
                    actions_checked+=1
                    per_action.append(dict(action=action,shape=list(current.shape),dtype='float32',bit_identical_to_source=True,content_sha256=hashlib.sha256(current.tobytes()).hexdigest()))
            data_path=Path(source['data_path'])
            with np.load(data_path,allow_pickle=False) as z:
                assert len(z['validation_lengths'])==5376
                assert np.array_equal(np.unique(z['validation_lengths']),np.arange(10,31))
                assert all((z['validation_lengths']==n).sum()==256 for n in range(10,31))
            records.append(dict(unit=unit,file=name,path=str(path),sha256=meta['output_sha256'],source_path=str(source_path),source_sha256=source['payload_sha256'],checkpoint_sha256=source['checkpoint_sha256'],data_sha256=source['data_sha256'],indices='Exact arange(5376), no subsampling/reordering',actions=per_action,sidecar_binding_passed=True))
        standardized=[r for r in audit['standardized_features'] if r['unit']==unit]
        assert len(standardized)==10 and len(receipt['features']['files'])==10
        for r in standardized:
            path=Path(r['path']);current=path.stat();initial=audit['files_hashed'][str(path)]
            assert receipt['features']['files'][str(path)]==r['sha256']==initial['sha256']
            assert current.st_size==initial['size_bytes'] and current.st_mtime_ns==initial['mtime_ns']
        reused.append(dict(unit=unit,completion_reports_reused=True,standardized_files=10,all_prior_hash_bindings_match=True,original_file_sizes_and_mtimes_unchanged=True))
    assert len(records)==30 and actions_checked==50
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),source_audit_sha256=sha(HERE/'cache_audit.json'),adapted_files=30,exported_action_arrays=50,total_feature_values_bit_compared=50*5376*256,
        units=5,new_units=4,all_reuse_receipts_true=True,standardized_files_reused=50,source_identity_and_order_bindings_passed=True,
        source_payload_hash_scope='Uses the current-task source audit expected SHA and exact array comparisons; independent global endpoint audit rehashes original source files. All30 output NPZ current bytes hashed here.',
        no_source_file_rebuild_evidence='All five completion receipts select reuse; all50 prior standardized-file SHA bindings, sizes and mtime_ns agree with start audit. This is endpoint/receipt evidence, not continuous system-call monitoring.',
        adapted=records,reused=reused,elapsed_seconds=time.monotonic()-begun,no_fitting=True,no_neural_forward=True,no_original_file_writes=True)
    (HERE/'adapter_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'adapter_verification.md').write_text('\n'.join([
        '# Validation-format adapter verification','',
        'All **30 adapted validation files** (six per unit, including the historical calibration) passed. Every one of the **50 exported action arrays**, comprising **68,812,800 FP32 values**, is bit-identical to its audited source `prefix_final[:, action, :]`. No extra normalization, fitting, subsampling, or reordering was introduced.','',
        'Every output preserves `validation_indices = arange(5376)` and the correct scalar checkpoint SHA. Output SHA, source/data/checkpoint bindings, action list, row count, and original-order flags match both `adaptation.json` and the unit completion receipt. All21 lengths retain256 validation rows.','',
        'All five completion receipts report feature reuse. All **50 original standardized files** retain their prior SHA bindings, sizes and modification times. No importer rebuild is recorded. This is endpoint/receipt evidence rather than continuous system-call monitoring; the separate global protection audit checks current original-file hashes.','',
        'This verification reads saved arrays and scalar metadata only. It performs no neural forward, model/probe/alignment fit, or old-file write. Exact paths and per-array content hashes are in `adapter_verification.json`.',''
    ]))
    print(json.dumps({k:result[k] for k in ['status','adapted_files','exported_action_arrays','total_feature_values_bit_compared','standardized_files_reused','elapsed_seconds']}))


if __name__=='__main__':main()
