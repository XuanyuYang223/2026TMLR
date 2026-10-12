"""Coverage, arithmetic, provenance and protection audit; no hypothesis tests."""
from common import *
from datetime import datetime,timezone
import shutil
import time

def main():
    started=time.monotonic();p=verify();records=rows(DIAG/'per_length_results.csv')
    assert len(records)==6048
    keys=['architecture','encoding','relation','control','branch','layer','metric','length']
    assert len({tuple(r[k] for k in keys) for r in records})==len(records)
    groups={};checks=0;identity_checks=0;max_decomposition_error=0.;nan_reasons={}
    source_rows=rows(SOURCE_ROOT/'geometry_length.csv')
    original={(r['architecture'],r['encoding'],r['relation'],r['control'],r['branch'],r['metric'],r['length']):r for r in source_rows}
    reused=0
    for r in records:
        groups.setdefault(tuple(r[k] for k in keys[:-1]),[]).append(int(r['length']))
        f={field:number(r.get(field)) for field in FIELDS}
        for target,x,y in [('B0','initial_correct','initial_wrong'),('Bt','trained_correct','trained_wrong'),
                            ('dc','trained_correct','initial_correct'),('dw','trained_wrong','initial_wrong'),
                            ('di','trained_identity','initial_identity'),('Delta','Bt','B0'),('Delta','dc','dw')]:
            if all(math.isfinite(f[x]) for x in [target,x,y]):
                error=abs(f[target]-(f[x]-f[y]));assert error<=1e-10
                max_decomposition_error=max(max_decomposition_error,error);checks+=1
        if not math.isfinite(f['Delta']):
            reason=r.get('undefined_reason',r.get('nan_reason',''));assert reason not in ('','{}'),('Undefined without reason',r)
            nan_reasons[reason]=nan_reasons.get(reason,0)+1
        if r['branch']=='hidden' and math.isfinite(f['initial_identity']):
            expected=0. if r['metric']=='procrustes' else 1.
            assert abs(f['initial_identity']-expected)<=1e-8,(r,f['initial_identity'])
            identity_checks+=1
        if r['layer'] in ['final_norm','output']:
            oldbranch='prefix_final' if r['branch']=='hidden' else 'output_prob'
            old=original[r['architecture'],r['encoding'],r['relation'],r['control'],oldbranch,r['metric'],r['length']]
            for field in FIELDS:
                alias={'dc':'change_correct','dw':'change_wrong','di':'change_identity'}.get(field,field)
                x,y=f[field],number(old[alias])
                assert (math.isnan(x) and math.isnan(y)) or x==y,('Original endpoint changed',field,r)
            reused+=1
    assert reused==2688 and all(sorted(ns)==p['lengths'] for ns in groups.values())
    summary=rows(DIAG/'contrast_summary.csv');summaries={}
    for r in summary:
        if r['contrast_type']=='metric_direction':continue
        f={field:number(r.get(field)) for field in FIELDS}
        for t,x,y in [('B0','initial_correct','initial_wrong'),('Bt','trained_correct','trained_wrong'),('Delta','Bt','B0'),('Delta','dc','dw')]:
            if all(math.isfinite(f[k]) for k in [t,x,y]):assert abs(f[t]-(f[x]-f[y]))<=1e-10;checks+=1
        if r['contrast_type']=='condition_summary':
            key=tuple(r[k] for k in keys[:-1]);summaries[key]=r
            group=[x for x in records if tuple(x[k] for k in keys[:-1])==key]
            for field in FIELDS:
                expected=equal_length([number(x.get(field)) for x in group]);actual=f[field]
                assert (math.isnan(actual) and math.isnan(expected)) or abs(actual-expected)<=1e-12
    assert len(summaries)==288
    cross=rows(DIAG/'cross_arch_same_task.csv');assert len(cross)==672
    crossgroups={}
    for r in cross:
        key=tuple(r[k] for k in ['task','encoding','control','metric']);crossgroups.setdefault(key,[]).append(int(r['length']))
        a,b,c=[number(r[f]) for f in ['initial_similarity','trained_similarity','similarity_change']]
        if all(math.isfinite(x) for x in [a,b,c]):assert abs(c-(b-a))<=1e-10;checks+=1
    assert len(crossgroups)==32 and all(sorted(g)==p['lengths'] for g in crossgroups.values())
    index=js(FROOT/'layers/index.json');assert index['status']=='completed'
    assert len(index['records'])==40 and len(index['job_condition_split_map'])==64
    feature_bytes=0;max_cache_error=0.
    for r in index['records']:
        assert r['source_updates']==0 and r['parameter_sha256_before']==r['parameter_sha256_after']
        assert r['old_prediction_disagreements']==0
        max_cache_error=max(max_cache_error,r['max_final_norm_difference'])
        assert r['max_final_norm_difference']<=2e-5
        for kind in ['features','lengths','answer_ce','answer_argmax']:
            if kind+'_path' in r:assert sha(r[kind+'_path'])==r[kind+'_sha256']
        feature_bytes+=Path(r['features_path']).stat().st_size
    benchmark=js(DIAG/'extraction_review/benchmark.json');assert benchmark['status']=='passed' and len(benchmark['records'])==180
    assert max(r['manual_same_shape_layers_max_abs'] for r in benchmark['records'])<=2e-5
    assert max(r['old_final_norm_max_abs'] for r in benchmark['records'])<=2e-5
    metric_audit=js(DIAG/'metric_review/analysis_audit.json');assert metric_audit['status']=='passed'
    learning=js(DIAG/'source_learning_audit.json');assert learning['status']=='passed' and learning['rows']==4032
    protected=js(DIAG/'protected_before.json')
    differences=[name for name,digest in protected.items() if not Path(name).is_file() or sha(name)!=digest]
    assert not differences,('Existing artifacts changed',differences)
    # Attach observed feature hashes to the frozen comparison descriptors, leaving the before file intact.
    manifest=rows(DIAG/'comparison_manifest_before_extraction.csv')
    physical={r['source_key']:r for r in index['records']}
    for row in manifest:
        for side in ['left','right']:
            endpoint=json.loads(row[side]);layer=endpoint['layer']
            if layer not in ['final_norm','embedding']:
                new=[]
                for job in endpoint['checkpoints']:
                    for condition in ['initial','trained']:
                        for split in ['validation','test']:
                            logical='|'.join([job['run_id'],condition,split])
                            actual=physical[index['job_condition_split_map'][logical]]
                            new.append(dict(run_id=job['run_id'],condition=condition,split=split,
                                path=actual['features_path'],sha256=actual['features_sha256'],layer_order=actual['layer_order'],
                                requested_layer=layer,cache_layer_name=('block_' if layer.startswith('block') else 'layer_')+f'{int(layer[-1]):02d}',
                                layer_index=int(layer[-1])-1,source_key=actual['source_key'],shape=actual['features_shape']))
                endpoint['feature_files']=new
            if layer=='embedding':endpoint['feature_files']=[dict(status='structural_constant_within_length; no feature file; metrics undefined')]
            row[side]=json.dumps(endpoint,ensure_ascii=False,sort_keys=True)
        row['observed_feature_hashes_attached']=True
    csv_write(DIAG/'comparison_manifest.csv',manifest)
    scopes=[dict(item=r['comparison_id'],comparison_class=r['comparison_class'],status=r['match_status'],
                 changed=r['intended_changed_fields'],remaining_coupling=r['unmatched_fields'],
                 reason='Structural constant embedding; explicitly undefined' if r['match_status']=='不可比较或缺失' else 'Fixed-source controlled diagnostic') for r in manifest]
    scopes.extend([dict(item='GroupWorld',comparison_class='excluded',status='未启动',changed='',remaining_coupling='',reason='Current scope forbids automatic expansion'),
        dict(item='F17',comparison_class='protected',status='保持不变',changed='',remaining_coupling='',reason='Existing protected hashes verified'),
        dict(item='second8k16models',comparison_class='excluded',status='未并入',changed='',remaining_coupling='',reason='Different data world and update budget'),
        dict(item='old82layers',comparison_class='historical',status='分开引用',changed='',remaining_coupling='Original initial identity unverified',reason='Reconstructed by original program; excluded from primary queue'),
        dict(item='initial_answer_CE',comparison_class='learning_auxiliary',status='缺失或未做',changed='',remaining_coupling='',reason='Not originally saved; shared initial hidden cannot supply task-specific CE')])
    csv_write(DIAG/'scope_status.csv',scopes)
    historical=WORKSPACE/'diagnostics/permworld_layers_20261009/layer_metric_summary.csv'
    shutil.copyfile(historical,DIAG/'historical_old82_summary.csv')
    json_write(DIAG/'historical_supplement.json',dict(source=str(historical),source_sha256=sha(historical),
        sources=82,initialization_identity='Reconstructed with original program; actual initial identity unverified',
        separate_historical_only=True,used_in_current_primary_contrasts=False))
    result=dict(status='passed',generated_utc=datetime.now(timezone.utc).isoformat(),models=16,source_updates=20000,
        lengths=21,length_weights='1/21',per_length_rows=len(records),condition_summary_rows=len(summaries),
        source_final_output_rows_reused_exactly=reused,cross_arch_same_task_rows=len(cross),
        comparisons=len(manifest),physical_feature_splits=40,logical_feature_splits=64,new_feature_bytes=feature_bytes,
        arithmetic_checks=checks,max_decomposition_error=max_decomposition_error,initial_identity_checks=identity_checks,
        undefined_Delta_rows=sum(nan_reasons.values()),undefined_reason_counts=nan_reasons,
        new_final_norm_matches_old_within_fixed_tolerance=True,max_final_norm_difference=max_cache_error,
        metric_independent_audit=metric_audit,source_learning_audit=learning,
        protected_files=len(protected),protected_hash_differences=differences,
        new_training=False,optimizer_constructed=False,backward=False,new_inputs=False,
        cached_existing_test_evaluated=True,second8k_test_used=False,F17_changed=False,GroupWorld_started=False,
        elapsed_seconds=time.monotonic()-started)
    json_write(DIAG/'validation_audit.json',result)
    print(dict(audit='passed',rows=len(records),arithmetic_checks=checks,old_files_unchanged=len(protected)),flush=True)

if __name__=='__main__':main()
