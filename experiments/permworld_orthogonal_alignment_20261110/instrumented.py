"""Run unchanged vendor entry points, recording their actual maps/shuffles."""
from pathlib import Path
import hashlib
import importlib
import json
import sys

import numpy as np

VENDOR=Path(__file__).resolve().parent/'vendor'


def digest(array):return hashlib.sha256(np.asarray(array).tobytes(order='C')).hexdigest()


def main():
    stage,unit,prefix=sys.argv[1:4];args=sys.argv[4:]
    assert stage in ['alignment','other_specialists']
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(VENDOR))
    module=importlib.import_module('alignment')
    original_rotation=module.rotation;original_shuffle=module.shuffled_indices
    maps=[];shuffles=[];arrays={}

    def rotation(cross):
        result=original_rotation(cross)
        error=float(np.max(np.abs(result.T@result-np.eye(result.shape[0]))))
        record=dict(call=len(maps),dimension=result.shape[0],cross_sha256=digest(cross),map_sha256=digest(result),
                    maximum_orthogonality_error=error,determinant_sign=float(np.linalg.slogdet(result)[0]))
        if stage=='alignment' and not maps:
            rng=np.random.default_rng(21);rng.normal(size=(50,8));q,_=np.linalg.qr(rng.normal(size=(8,8)))
            record['synthetic_target_to_source_orientation_error']=float(np.max(np.abs(result-q.T)))
            assert record['synthetic_target_to_source_orientation_error']<=2e-12
            record['role']='vendor_synthetic_orientation_check'
        else:
            index=len(maps)-(1 if stage=='alignment' else 0)
            names=['identity','complement','inverse',*[f'shuffle_{s}' for s in range(1026,1034)]]
            if stage=='alignment':
                block=index//11;record.update(encoding=['canonical_cycle','one_line'][block//2],phase=['initial','trained'][block%2],fit_condition=names[index%11],role='main_map')
            elif index<11:record.update(encoding='one_line',phase='trained',fit_condition=names[index],role='unused_valleys_preparation_map')
            else:record.update(encoding='one_line',phase='trained',fit_condition=['descents','recoils'][index-11],role='other_specialist_map')
        maps.append(record);arrays[f'R_{len(maps)-1:03d}']=result.copy()
        assert np.isfinite(result).all() and error<=2e-12
        return result

    def shuffled_indices(counts,rng):
        permutation=original_shuffle(counts,rng)
        assert np.array_equal(np.sort(permutation),np.arange(len(counts)))
        assert np.array_equal(counts[permutation],counts)
        _,inverse,freq=np.unique(counts,return_inverse=True,return_counts=True)
        assert np.all(permutation[freq[inverse]>1]!=np.flatnonzero(freq[inverse]>1))
        j=len(shuffles);block=j//(21*8)
        record=dict(call=j,length=10+(j//8)%21,shuffle_seed=1026+j%8,
                    encoding=['canonical_cycle','one_line'][block//2] if stage=='alignment' else 'one_line',
                    phase=['initial','trained'][block%2] if stage=='alignment' else 'trained',
                    fit_pairs=len(counts),moved=int((permutation!=np.arange(len(counts))).sum()),
                    singleton_anchors=int((freq[inverse]==1).sum()),counts_sha256=digest(counts),permutation_sha256=digest(permutation))
        shuffles.append(record);arrays[f'permutation_{j:03d}']=permutation.copy()
        return permutation

    module.rotation=rotation;module.shuffled_indices=shuffled_indices
    sys.argv=[str(VENDOR/(stage+'.py')),*args];status='failed'
    try:
        target=module if stage=='alignment' else importlib.import_module('other_specialists')
        target.main();status='completed'
    finally:
        np.savez_compressed(prefix.with_suffix('.npz'),**arrays)
        record=dict(status=status,stage=stage,unit=unit,maps=maps,shuffle_calls=shuffles,
                    map_calls=len(maps),shuffle_call_count=len(shuffles),
                    maximum_orthogonality_error=max((r['maximum_orthogonality_error'] for r in maps),default=None),
                    strict_orthogonality_tolerance=2e-12,strict_relative_tolerance=0,
                    parameter_archive_path=str(prefix.with_suffix('.npz')),
                    parameter_archive_sha256=hashlib.sha256(prefix.with_suffix('.npz').read_bytes()).hexdigest(),
                    original_vendor_files_unchanged=True,wrappers_return_original_numeric_outputs=True,
                    no_neural_forward=True,no_base_training=True)
        prefix.with_suffix('.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    assert len(maps)==(45 if stage=='alignment' else 13)
    assert len(shuffles)==(672 if stage=='alignment' else 168)


if __name__=='__main__':main()
