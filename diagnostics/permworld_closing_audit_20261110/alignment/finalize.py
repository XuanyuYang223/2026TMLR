"""Bounded independent arithmetic and end-SHA audit for the alignment check."""
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_closing_audit/alignment')


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pairwise_auc(scores,labels):
    positive=scores[labels];negative=scores[~labels]
    if not len(positive) or not len(negative):return np.nan
    difference=positive[:,None]-negative[None,:]
    return float((np.count_nonzero(difference>0)+.5*np.count_nonzero(difference==0))/difference.size)


def main():
    plan=read(HERE/'source_plan.json');freeze=read(HERE/'freeze.json')
    assert sha(HERE/'source_plan.json')==freeze['source_plan_sha256']
    assert sha(HERE/'spec.json')==freeze['spec_sha256']
    def check_source(row):
        actual=sha(row['path']);assert actual==row['sha256'],row['path']
        return dict(path=row['path'],expected_sha256=row['sha256'],actual_sha256=actual,
                    size_bytes=Path(row['path']).stat().st_size,passed=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        protected=list(pool.map(check_source,plan['inputs'].values()))
    checks=0;max_auc_error=0.;orthogonality=0.;outputs=[];unit_rows=[]
    for entry in plan['units']:
        name=entry['unit'];receipt=read(HERE/'results'/name/'completion.json')
        assert receipt['same_trained_probe_all_three_targets'] and receipt['no_probe_fits']
        assert receipt['maximum_trained_route_score_error']==0.
        assert receipt['source_probe_and_scaler_sha256']==entry['readouts']['sha256']
        for path,digest in receipt['outputs'].items():
            assert sha(path)==digest;outputs.append(dict(path=path,sha256=digest,size_bytes=Path(path).stat().st_size));checks+=1
        with np.load(OUT/name/'maps_and_means.npz',allow_pickle=False) as params:
            assert str(params['source_probe_scaler_archive_sha256'])==entry['readouts']['sha256']
            for matrix in params['R']:
                error=float(np.max(np.abs(matrix.T@matrix-np.eye(256))))
                orthogonality=max(orthogonality,error);assert error<=2e-12;checks+=1
        with np.load(OUT/name/'test_scores.npz',allow_pickle=False) as archive:
            assert str(archive['source_probe_scaler_archive_sha256'])==entry['readouts']['sha256']
            assert np.array_equal(archive['anchor_indices'],np.arange(10752));checks+=1
            assert np.array_equal(archive['target_names'],['trained_valleys','actual_saved_step0','trained_fixed_points']);checks+=1
            lengths=archive['lengths'];scores=archive['rawscore'];labels=archive['labels'];auc=archive['per_sample_auroc']
            for n in [10,20,30]:
                ids=np.flatnonzero(lengths==n)
                for local in [0,17,255,511]:
                    sample=ids[local]
                    for route in range(3):
                        expected=pairwise_auc(scores[route,sample,:n-2],labels[sample,:n-2])
                        actual=float(auc[route,sample])
                        if np.isnan(expected):assert np.isnan(actual)
                        else:
                            error=abs(expected-actual);max_auc_error=max(max_auc_error,error);assert error<=1e-12
                        checks+=1
            for n in range(10,31):
                ids=np.flatnonzero(lengths==n)
                assert np.isnan(scores[:,ids,n-2:]).all()
                assert np.isfinite(scores[:,ids,:n-2]).all();checks+=2
                # Test support and undefined-quantity denominators from saved arrays.
                for support,key in [('full_test','eligible'),('count_matched','matched')]:
                    keep=archive[key][ids]
                    for route in range(3):
                        selected=auc[route,ids][keep]
                        expected=float(np.mean(selected[np.isfinite(selected)]))
                        with (HERE/'results'/name/'by_length.csv').open() as stream:
                            row=next(r for r in csv.DictReader(stream) if r['target']==str(archive['target_names'][route]) and r['support']==support and int(r['length'])==n)
                        assert int(row['n'])==int(keep.sum())
                        assert int(row['finite'])==int(np.isfinite(selected).sum())
                        assert int(row['undefined'])==int((~np.isfinite(selected)).sum())
                        assert abs(float(row['auroc'])-expected)<=1e-12;checks+=4
        unit_rows.append(dict(unit=name,historical_reference=entry['historical'],no_probe_fits=True,
            same_trained_source_for_all_targets=True,new_maps=2,trained_route_score_error=0.))
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks,
        protection_scope='All90 explicitly used code/protocol/probe/scaler/map/activation/data/checkpoint files rehashed at end; endpoint evidence, not whole disk or continuous monitoring.',
        protected_source_count=len(protected),protected_inputs=protected,new_outputs=outputs,
        source_plan_sha256=sha(HERE/'source_plan.json'),spec_sha256=sha(HERE/'spec.json'),
        maximum_independent_pairwise_auc_error=max_auc_error,maximum_orthogonality_error=orthogonality,
        independent_auc_subset='all5units × n10/20/30 × original length-local indices0/17/255/511 ×3routes=180examples',
        no_neural_forward=True,no_probe_fits=True,no_model_training=True,new_maps_fitted=10,
        units=unit_rows)
    write(HERE/'final_audit.json',result)
    print(json.dumps({k:result[k] for k in ['status','checks','protected_source_count','maximum_independent_pairwise_auc_error','maximum_orthogonality_error']}))


if __name__=='__main__':main()
