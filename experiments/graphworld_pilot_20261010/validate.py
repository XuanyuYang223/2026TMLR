"""Separate frozen CPU metric verification; never trains or extracts activations."""
import argparse
import csv
import importlib.util
import math
import time
from .common import *
from .evaluation import selections

INDEPENDENT=WORKSPACE/'diagnostics/permworld_controlled_comparison_20261010/metric_review/independent_metrics.py'
VERIFY_FILE=DIAG/'verification_freeze.json'
SCORE_TOLERANCE=1e-8

def independent_module():
    spec=importlib.util.spec_from_file_location('graphworld_independent_fixed_metrics',INDEPENDENT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def array_hash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def register():
    c=verify();data,_,audit=load_data(c,('validation','analysis'));r=RELATIONS[0];selectors={}
    assert r['relation']=='reverse_max'
    for control in ('raw_within_n','answer_strata'):
        row={}
        for split in ('validation','analysis'):
            mask,keys=selections(data,r,split,6,control);ids=np.flatnonzero(mask)
            selected=ids[:64].tolist() if len(ids)>=64 else None
            row[split]=dict(qualifying_count=len(ids),status='available' if selected is not None else 'unavailable',
                selected_indices=selected,full_mask_sha256=array_hash(mask),full_keys_sha256=array_hash(keys) if keys is not None else None,
                selected_anchor_ids=data[split+'_anchor_id'][selected].tolist() if selected is not None else None)
        selectors[control]=row
    independent=independent_module();fixture=independent.run_fixture()
    assert fixture['status']=='passed'
    paths=[Path(__file__).resolve(),INDEPENDENT,WORKSPACE/'diagnostics/permworld_scope_inventory_20261009/metrics.py',
           Path(__file__).with_name('core.py'),Path(__file__).with_name('evaluation.py'),Path(__file__).with_name('common.py')]
    frozen=dict(version='graphworld-independent-verification/v1',protocol_sha256=sha(CONFIG),main_source_version_sha256=source_version(),
        data_sha256=audit['data_sha256'],source_sha256={str(p):sha(p) for p in paths},
        fixed_relation='reverse_max',fixed_n=6,fixed_first_qualifying_anchors=64,selectors=selectors,
        control_selection='Use the same full-split golden-label selector, then freeze first64 qualifying anchors; fewer64 means unavailable. Recenter selected64 within their keys for numerical verification only, not a new scientific endpoint.',
        conditions=['initial','trained'],branches=['prefix_final','output_prob'],roles=['correct','wrong','identity'],
        knn_k=10,procrustes_rank_cap=8,singular_cutoff_relative=1e-6,score_tolerance=SCORE_TOLERANCE,
        knn_tolerance=1e-12,rank_not_adapted=True,undefined_preserved=True,
        independent_paths=['feature covariance CKA','explicit off-diagonal U-centering loops','direct Euclidean pair distances/stable-ID neighbors',
                           'covariance eigendecomposition PCA8, independent fit RMS and held-out Procrustes'],
        no_new_model_or_forward=True,no_new_training=True,fixture_status=fixture['status'])
    if VERIFY_FILE.exists():assert read(VERIFY_FILE)==frozen,'Refuse changing frozen verification selection'
    else:json_write(VERIFY_FILE,frozen)
    json_write(DIAG/'verification_fixture.json',fixture)
    return frozen

def verify_registration():
    c=verify();f=read(VERIFY_FILE)
    assert f['protocol_sha256']==sha(CONFIG) and f['main_source_version_sha256']==source_version()
    for path,digest in f['source_sha256'].items():assert sha(path)==digest,('Verification source changed',path)
    assert sha(ROOT/'world/data.npz')==f['data_sha256']
    return c,f

def actual():
    c,f=verify_registration();data,_,_=load_data(c,('validation','analysis'));ind=independent_module()
    r=RELATIONS[0];features={};cache_provenance=[];records=[];input_records=[];begun=time.monotonic()
    for task in (r['left'],r['right']):
        for cond in ('initial','trained'):
            for split in ('validation','analysis'):
                path=ROOT/'evaluation_cache'/f'{task}_{cond}_{split}.npz'
                if not path.exists():
                    result=dict(status='unavailable',reason='Required frozen actual cache not yet produced',missing_path=str(path),
                                protocol_sha256=sha(CONFIG),no_new_forward=True,no_new_training=True)
                    json_write(DIAG/'independent_actual_small_batch.json',result);return result
                meta=read(path.with_suffix('.json'));assert sha(path)==meta['sha256']
                for k,v in dict(protocol_sha256=sha(CONFIG),data_sha256=f['data_sha256'],source_version_sha256=source_version(),
                               task=task,split=split,condition=cond).items():assert meta['signature'][k]==v
                cp=Path(c['initial_checkpoint_path']) if cond=='initial' else ROOT/'models'/('graphpilot001_'+task)/'checkpoint.pt'
                assert sha(cp)==meta['signature']['checkpoint_sha256']
                with np.load(path,allow_pickle=False) as ar:features[task,cond,split]={b:ar[b] for b in f['branches']}
                cache_provenance.append(dict(task=task,condition=cond,split=split,path=str(path),sha256=meta['sha256']))
    def check(label,production,independent,tolerance=SCORE_TOLERANCE):
        missing=production is None or independent is None
        error=None if missing else abs(float(production)-float(independent))
        passed=(production is None and independent is None) if missing else error<=tolerance
        records.append(dict(check=label,production=production,independent=independent,absolute_error=error,tolerance=tolerance,
                            status='undefined_consistent' if missing and passed else 'passed' if passed else 'failed'))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        for control,selector in f['selectors'].items():
            if any(v['selected_indices'] is None for v in selector.values()):
                records.append(dict(check=control,status='unavailable',reason='Fewer64 qualifying anchors; threshold/count unchanged'));continue
            fi=np.array(selector['validation']['selected_indices']);ti=np.array(selector['analysis']['selected_indices']);ids=ti
            kf0=selections(data,r,'validation',6,control)[1];kt0=selections(data,r,'analysis',6,control)[1]
            kf=kf0[:64] if kf0 is not None else None;kt=kt0[:64] if kt0 is not None else None
            orbits=data['analysis_orbit_id'][ti,None]
            for cond in f['conditions']:
                for branch in f['branches']:
                    for role,ai in [('correct',1),('wrong',2),('identity',0)]:
                        raw=[features[r['left'],cond,'validation'][branch][fi,0],features[r['right'],cond,'validation'][branch][fi,ai],
                             features[r['left'],cond,'analysis'][branch][ti,0],features[r['right'],cond,'analysis'][branch][ti,ai]]
                        prod=[metrics.center(x,keys) for x,keys in zip(raw,(kf,kf,kt,kt))]
                        independent=[ind.independent_center(x,keys) for x,keys in zip(raw,(kf,kf,kt,kt))]
                        label=f'{control}/{cond}/{branch}/{role}'
                        check(label+'/centering_max_abs',max(float(np.max(np.abs(a-b))) for a,b in zip(prod,independent)),0.,1e-10)
                        ref,rank=metrics.scores(*prod,ids=ids,duplicate_keys=orbits)
                        xf,yf,xt,yt=independent
                        error,irank=ind.independent_procrustes(xf,yf,xt,yt,rank_cap=8)
                        assert irank==rank
                        expected=dict(linear_cka=ind.covariance_cka(xt,yt),debiased_cka=ind.independent_u_cka(xt,yt),
                                      knn_overlap=ind.brute_overlap(xt,yt,k=10,ids=ids,duplicate_keys=orbits),
                                      procrustes=-error if error is not None else None)
                        for metric,value in expected.items():check(label+'/'+metric,ref[metric][0],value,1e-12 if metric=='knn_overlap' else SCORE_TOLERANCE)
                        a=metrics.neighbors(prod[2],ids=ids,duplicate_keys=orbits);b=ind.brute_neighbors(xt,ids=ids,duplicate_keys=orbits)
                        same=a==b;records.append(dict(check=label+'/left_neighbor_ID_sets',status='passed' if same else 'failed',
                            changed_sets=sum(u!=v for u,v in zip(a,b)) if a is not None and b is not None else None))
            # Independently check exact static input isometries, no fake training increment.
            def input_x(split,index,ai,keys):
                a=data[split+'_graphs'][index,ai,:6,:6];edge=a[:,~np.eye(6,dtype=bool)]
                return ind.independent_center(np.eye(2)[edge].reshape(64,60),keys)
            xfit=input_x('validation',fi,0,kf);xtest=input_x('analysis',ti,0,kt)
            for ai,action in enumerate(ACTIONS):
                yf=input_x('validation',fi,ai,kf);yt=input_x('analysis',ti,ai,kt)
                errors=[float(np.max(np.abs(a-b))) for a,b in [(xfit@xfit.T,yf@yf.T),(xtest@xtest.T,yt@yt.T),(xtest@xfit.T,yt@yf.T)]]
                knn=ind.brute_overlap(xtest,yt,ids=ids)
                gram_variance=np.linalg.norm(xtest@xtest.T)>1e-20
                cka=ind.covariance_cka(xtest,yt)
                passed=max(errors)<1e-10 and (knn==1. or knn is None) and (cka is None or abs(cka-1.)<1e-10)
                input_records.append(dict(control=control,action=action,fit_test_cross_gram_max_abs=max(errors),independent_cka=cka,
                    independent_knn_overlap=knn,kernel_has_variance=bool(gram_variance),status='passed' if passed else 'failed',static_no_B0_or_Delta=True))
    failures=[r for r in records+input_records if r['status']=='failed']
    result=dict(status='passed' if not failures else 'failed',protocol_sha256=sha(CONFIG),verification_freeze_sha256=sha(VERIFY_FILE),
        actual_cache_provenance=cache_provenance,checks=records,input_kernel_checks=input_records,failure_count=len(failures),
        wall_seconds=time.monotonic()-begun,new_training=False,new_forward=False,scientific_metric_definition_changed=False,
        fixed64_subset_is_numerical_verification_only=True)
    json_write(DIAG/'independent_actual_small_batch.json',result);return result

def audit_report():
    """Additional four-state coverage/provenance review; does not edit frozen report."""
    c,f=verify_registration();coverage=[];source_records=[];problems=[]
    for job in c['training_manifest']:
        p=ROOT/'models'/job['run_id']/'completed.json'
        if not p.exists():coverage.append(dict(unit=job['run_id'],kind='source',status='missing',reason='Fixed20k endpoint not yet produced'));continue
        r=read(p);cp=ROOT/'models'/job['run_id']/'checkpoint.pt'
        good=r['step']==20000 and r['protocol_sha256']==sha(CONFIG) and r['source_version_sha256']==source_version() and sha(cp)==r['checkpoint_sha256']
        good &=r['initial_parameter_sha256']==c['initial_parameter_sha256'] and r['starting_rng_hash']==c['starting_rng_hash'] and r['actual_initial_identity_verified']
        if not good:problems.append(job['run_id']+' provenance mismatch')
        coverage.append(dict(unit=job['run_id'],kind='source',status='completed' if good else 'missing',reason='Fixed20k hash-bound actual endpoint' if good else 'Unverified endpoint'))
        source_records.append(r)
    for r in RELATIONS:coverage.append(dict(unit=r['relation'],kind='dependence',status='coupled',reason='Fourtasks/twotransforms/oneworld/oneinitial; not independent relation repeats'))
    table=DIAG/'geometry_per_n.csv'
    if table.exists():
        with table.open(newline='') as stream:rows=list(csv.DictReader(stream))
        expected={(r['relation'],str(n),control,branch,metric) for r in RELATIONS for n in (6,7,8) for control in c['evaluation']['controls']
                  for branch in ('prefix_final','output_prob','gold_onehot','gold_scalar','input_onehot') for metric in ('linear_cka','knn_overlap','procrustes','debiased_cka')}
        keys=[tuple(r[k] for k in ('relation','n','control','branch','metric')) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:problems.append('Scientific table coverage differs from720 unique cells')
        for r in rows:
            undefined=any(r[k]=='' for k in ('trained_correct','trained_wrong'))
            static=r['static_reference']=='True'
            if static and any(r[k]!='' for k in ('initial_correct','initial_wrong','initial_identity','B0','dc','dw','di','Delta')):problems.append('Fabricated static increment')
            coverage.append(dict(unit='/'.join(r[k] for k in ('relation','n','control','branch','metric')),kind='metric',
                status='undefined' if undefined else 'completed',reason='No variable/rank/eligible input defined' if undefined else 'Defined exploratory cell; dependence listed separately'))
    else:coverage.append(dict(unit='all720scientific_cells',kind='metric',status='missing',reason='No actual geometry table yet; do not impute zero'))
    csv_write(DIAG/'coverage_review.csv',coverage)
    protected=check_protected()
    result=dict(status='passed' if not problems else 'failed',protocol_sha256=sha(CONFIG),verification_freeze_sha256=sha(VERIFY_FILE),
        four_coverage_states=['completed','coupled','undefined','missing'],coverage=coverage,problems=problems,
        source_count=len(source_records),world_count=1,shared_initial_count=1,source_recorded_process_seconds=sum(r['elapsed_seconds'] for r in source_records),
        training_cost_is_process_wall_not_GPU_active_time=True,old_protected_files=protected,
        static_increment_is_not_learning_evidence=True,no_report_or_scientific_result_modified=True,no_new_training_or_forward=True)
    json_write(DIAG/'report_reader_audit.json',result);return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['register','actual','audit-report'],required=True)
    args=parser.parse_args();result={'register':register,'actual':actual,'audit-report':audit_report}[args.stage]()
    print(json.dumps({k:v for k,v in result.items() if k in ('status','failure_count','source_count','world_count','protocol_sha256','verification_freeze_sha256')}))
