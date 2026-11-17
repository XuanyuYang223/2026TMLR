"""Independent reconstruction of saved final-state scores without any fitting."""
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.permworld_exceedance_alignment_20261110.shared import (
    DIAG,OUT,ENCODINGS,LENGTHS,sha,array_sha,read,write,read_csv,
    load_data,raw_arrays,split_rows,vendor,
)
from experiments.permworld_exceedance_alignment_20261110.oracles import (
    batch_labels,pullback_target_owned_batch,
)


def main():
    checks=0
    maximum=0.
    units=read(DIAG/'cache_audit.json')['units']
    _,_,example_auc,average,_,_,shuffle=vendor()
    for unit in units:
        receipt=read(DIAG/'results'/unit['unit']/'completion.json')
        for path,digest in receipt['files'].items():
            assert sha(path)==digest,path
            checks+=1
        data=load_data(unit)
        splits=split_rows(data)
        detail={(r['encoding'],int(r['length']),r['method'],r['support']):r
                for r in read_csv(DIAG/'results'/unit['unit']/'by_length.csv')}
        for enc in ENCODINGS:
            with np.load(OUT/'main'/unit['unit']/f'{enc}-scores.npz',allow_pickle=False) as f:
                saved={key:f[key].copy() for key in f.files}
            with np.load(OUT/'main'/unit['unit']/f'{enc}-parameters.npz',allow_pickle=False) as f:
                params={key:f[key].copy() for key in f.files}
            assert np.array_equal(saved['anchor_indices'],np.arange(len(data['test_lengths'])))
            assert np.array_equal(saved['lengths'],data['test_lengths'])
            assert saved['data_sha256'].item()==unit['data_sha256']
            checks+=3
            # Every saved method/sample, not only favorable or finite cells.
            for n,row in splits.items():
                ti=row['test_indices']
                e=batch_labels(row['test_permutations'])['exceedances']
                assert np.array_equal(saved['labels'][ti,:n],e)
                assert np.array_equal(saved['matched'][ti],row['matched'])
                assert np.array_equal(saved['eligible'][ti],row['eligible'])
                checks+=3
                for mi,method in enumerate(saved['method_names']):
                    calculated=example_auc(saved['scores'][mi,ti,:n],e)
                    np.testing.assert_allclose(calculated,saved['per_sample_auroc'][mi,ti],atol=0,rtol=0,equal_nan=True)
                    for support in ['eligible','matched']:
                        chosen=calculated[row[support]]
                        r=detail[enc,n,str(method),support]
                        value=average(chosen)
                        stored=float(r['auc'])
                        if np.isfinite(value):
                            error=abs(stored-value)
                            maximum=max(maximum,error)
                            assert error<=1e-12
                        else:
                            assert not np.isfinite(stored)
                        assert int(r['n'])==len(chosen)
                        assert int(r['finite'])==int(np.isfinite(chosen).sum())
                        checks+=3
                    checks+=1
            # Reconstruct source predictions from the serialized actual linear
            # coefficients, and independently verify map/mean/gather semantics
            # on fixed first two anchors of every registered length.
            raw={}
            for label,task,phase in [('source','exceedances','trained'),('target','deficiencies','trained'),
                                     ('step0','deficiencies','initial'),('fixed','fixed_points','trained')]:
                raw[label]=raw_arrays(unit,enc,task,phase,'test')[0]
            rngs=[np.random.default_rng(seed) for seed in range(1026,1034)]
            for n,row in splits.items():
                ti=row['test_indices'][:2]
                e=row['source_test_labels'][:2]
                perm=row['test_permutations'][:2]
                def probe(h,task='source'):
                    # Original StandardScaler.transform does these operations
                    # in place in input dtype, then the vendor casts to FP64.
                    scaled=np.array(h,copy=True)
                    # sklearn1.9 casts each stored statistic to input dtype
                    # before the in-place operation (not only its result).
                    scaled-=params[f'{task}_n{n}_mean'].astype(scaled.dtype)
                    scaled/=params[f'{task}_n{n}_scale'].astype(scaled.dtype)
                    scaled=scaled.astype(np.float64)
                    return scaled@params[f'{task}_n{n}_coef'].T+params[f'{task}_n{n}_intercept']
                mx=params[f'n{n}_source_mean']
                invmean=params[f'n{n}_inverse_mean']
                y=raw['target'][ti,2]
                tests=dict(direct=probe(y),means_only=probe(y-invmean+mx),
                           within_source=probe(raw['source'][ti,0]),
                           within_target_reindexed=pullback_target_owned_batch(probe(y,'target'),perm))
                for name in ['identity','complement','inverse',*[f'shuffle_{seed}' for seed in range(1026,1034)]]:
                    rotation=params['R_'+name]
                    assert np.max(np.abs(rotation.T@rotation-np.eye(256)))<=2e-12
                    tests[f'fit_{name}__test_inverse']=probe((y-invmean)@rotation+mx)
                for name,label in [('step0','step0'),('fixed','fixed')]:
                    method='aligned_step0_target' if name=='step0' else 'aligned_fixed_points_target'
                    tests[method]=probe((raw[label][ti,2]-params[f'n{n}_{name}_mean'])@params['R_'+name]+mx)
                for method,predicted in tests.items():
                    mi=list(saved['method_names']).index(method)
                    error=float(np.max(np.abs(predicted-saved['scores'][mi,ti,:n])))
                    maximum=max(maximum,error)
                    assert error<=1e-12,(unit['unit'],enc,n,method,error)
                    checks+=1
                counts=row['source_val_labels'][row['fit']].sum(1)
                for seed,rng in zip(range(1026,1034),rngs):
                    indices=shuffle(counts,rng)
                    assert np.array_equal(indices,params[f'n{n}_shuffle_{seed}_local'])
                    assert np.array_equal(counts[indices],counts)
                    checks+=2
            del raw,saved,params
    write(DIAG/'main_saved_score_audit.json',dict(status='passed',checks=checks,failures=[],
          maximum_absolute_difference=maximum,absolute_tolerance=1e-12,
          scope='All saved ED method AUROC samples and by-length aggregates; independent coefficient/mean/rotation/own-target gather reconstruction on fixed first2 anchors per length; all count-preserving shuffle RNGs; no fits or forwards.',
          no_probe_fitting=True,no_neural_forward=True))
    print(dict(status='passed',checks=checks,maximum_absolute_difference=maximum),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
