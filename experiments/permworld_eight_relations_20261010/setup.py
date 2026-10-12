"""Audit unchanged world/labels, save fixed selections, freeze missing-source list."""
from datetime import datetime,timezone
from itertools import permutations
import argparse
import copy
import time
from .common import *
from experiments.permutation_audit import transform

def math_audit():
    DIAG.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();source=original.verify();path=SOURCE_ROOT/'worlds/archdev001/data.npz'
    assert sha(path)==js(OLD_ANALYSIS/'protocol.json')['source_data_sha256']
    from neurips_permutations.property_task_geometry_experiments import verify_relation_identities
    exhaustive=verify_relation_identities(WORKSPACE/'external/neurips/configs/property_task_geometry.toml',max_n=8)
    with np.load(path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    checks=0;formulas=0;input_rows=0;counts={};random_records=[]
    for split in ['train','validation','test']:
        actions=['identity'] if split=='train' else list(ACTIONS)
        counts[split]={}
        for row,n in enumerate(data[split+'_lengths']):
            n=int(n);p=tuple(map(int,data[split+'_permutations'][row,:n]));input_rows+=1
            counts[split][str(n)]=counts[split].get(str(n),0)+1
            for ai,action in enumerate(actions):
                q=transform(p,action)
                computed=np.array([int(PROPERTY_FUNCTIONS[t](q)) for t in TASKS],dtype=np.int64)
                assert np.array_equal(computed,data[split+'_labels'][row,ai]),('Cached label mismatch',split,row,action)
                assert np.all((computed>=0)&(computed<=30));checks+=len(TASKS)
            if split!='train':
                for relation in RELATIONS:
                    ci=ACTIONS.index(relation['input_transform'])
                    a=int(data[split+'_labels'][row,0,TASKS.index(relation['left'])])
                    b=int(data[split+'_labels'][row,ci,TASKS.index(relation['right'])])
                    assert b==a+relation['right_label_offset'];formulas+=1
        print(dict(math_labels_complete=split,anchors=len(data[split+'_lengths'])),flush=True)
    rng=np.random.default_rng(202610100816)
    for n in range(10,31):
        for _ in range(256):
            p=tuple(map(int,rng.permutation(np.arange(1,n+1))))
            for r in RELATIONS:
                assert PROPERTY_FUNCTIONS[r['right']](transform(p,r['input_transform']))==PROPERTY_FUNCTIONS[r['left']](p)+r['right_label_offset']
        random_records.append(dict(length=n,permutations=256,relations=8))
    # Fixed first eight per length/split/action; no geometric outcomes used.
    roundtrips=0
    for split in ['train','validation','test']:
        for n in range(10,31):
            for row in np.flatnonzero(data[split+'_lengths']==n)[:8]:
                p=tuple(map(int,data[split+'_permutations'][row,:n]))
                for action in (['identity'] if split=='train' else ACTIONS):
                    q=transform(p,action);ai=ACTIONS.index(action)
                    for enc in ENCODINGS:
                        encoded=encode(q,enc)
                        assert decode(encoded,enc)==q
                        assert tuple(map(int,data[f'{split}_{enc}_input'][row,ai,:2*n+4]))==encoded
                        roundtrips+=1
    result=dict(status='passed',data_path=str(path),data_sha256=sha(path),label_task_names=list(TASKS),
        full_cached_label_recomputation_checks=checks,full_cached_relationship_checks=formulas,
        anchors_reused=input_rows,exhaustive=exhaustive,random_lengths=random_records,
        fixed_roundtrip_checks=roundtrips,counts=counts,
        anti_fixed_definition='Number of positions i with pi(i)=n+1-i; complement maps fixed to anti-fixed',
        subsequence_not_run=True,global_components_direction='components(C(pi))=global_descents(pi)+1',
        no_new_training_anchors=True,elapsed_cpu_wall_seconds=time.monotonic()-started)
    json_write(DIAG/'math_label_audit.json',result)
    return result

def freeze():
    DIAG.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    if CONFIG.exists():return verify()
    source=original.verify();inventory=js(DIAG/'inventory/sources.json')
    assert js(DIAG/'math_label_audit.json')['status']=='passed'
    old=js(OLD_ANALYSIS/'protocol.json')
    data_path=SOURCE_ROOT/'worlds/archdev001/data.npz';batchpath=data_path.with_name('batch_indices.npy')
    inits={};starts={}
    for arch in ARCHITECTURES:
        path=SOURCE_ROOT/'initializations'/arch/'step0.pt';meta=js(path.with_suffix('.json'))
        assert sha(path)==meta['checkpoint_sha256']
        inits[arch]=dict(path=str(path),sha256=sha(path),parameter_sha256=meta['parameter_sha256'],evidence='saved_actual_identity_verified')
        path=path.with_name('start_rng_descents.pt');saved=torch.load(path,map_location='cpu',weights_only=False)
        assert rng_hash(saved['rng'])==saved['rng_hash'] and saved['parameter_sha256']==meta['parameter_sha256']
        starts[arch]=dict(path=str(path),sha256=sha(path),rng_hash=saved['rng_hash'],source_task='descents',
            policy='New task had no original same-task stream; reuse saved architecture-specific descents state, not seeded reconstruction')
    missing=rows(DIAG/'training_manifest.csv');assert len(missing)<=48
    expected={(a,e,t) for a in ARCHITECTURES for e in ENCODINGS for t in TASKS if t not in original.DEV_TASKS}
    assert {(j['architecture'],j['encoding'],j['task']) for j in missing}==expected
    for j in missing:
        j['model_seed']=source['architecture_model_seeds'][j['architecture']]
        j['updates']=20000;j['data_seed']=source['units'][0]['data_seed'];j['batch_seed']=source['units'][0]['batch_seed']
    c=dict(version='permworld-eight-relations/v1',registered_utc=datetime.now(timezone.utc).isoformat(),
        exploratory=True,prior_results_seen=True,source_root=str(SOURCE_ROOT),root=str(ROOT),diagnostic=str(DIAG),
        data_path=str(data_path),data_sha256=sha(data_path),source_data_path=str(data_path),source_data_sha256=sha(data_path),
        source_protocol_path=str(original.CONFIG),source_protocol_sha256=sha(original.CONFIG),
        batch_schedule_path=str(batchpath),batch_schedule_sha256=sha(batchpath),
        model=copy.deepcopy(source['model']),training=copy.deepcopy(source['training']),
        architecture_model_seeds=source['architecture_model_seeds'],initializations=inits,new_task_start_rng=starts,
        initialization_control='True saved original step0 per arch; new12tasks share saved descents RNG per arch, paired encodings exact; old4 retain their own original task RNG.',
        label_task_names=list(TASKS),tasks=list(TASKS),architectures=list(ARCHITECTURES),encodings=list(ENCODINGS),
        lengths=list(range(10,31)),length_weights={str(n):1/21 for n in range(10,31)},splits=['validation','test'],actions=list(ACTIONS),
        relations=copy.deepcopy(RELATIONS),training_manifest=missing,
        checkpoint_inventory=copy.deepcopy(old['checkpoint_inventory']),
        expected_complete_models=64,existing_complete_models=16,maximum_new_models=48,
        controls=['raw_within_length','answer_strata'],
        answer_strata='joint[n,left identity,right identity-offset,right complement-offset,right inverse-offset]; min3 per split; subtract each split stratum mean; integer offset does not change partitions',
        same_task_answer_strata='joint[n,own identity/complement/inverse]; min3 per split; isolated secondary comparison',
        layer_rules=old['layer_rules'],primary_layer='final_norm',pooling=old['pooling'],
        metrics=old['metrics'],cross_arch_same_task=old['cross_arch_same_task'],validation=old['validation'],
        inference=old['inference'],learning=old['learning'],old_analysis_dir=str(OLD_ANALYSIS),
        old_analysis_artifact_sha256=sha(OLD_ANALYSIS/'artifact_manifest.json'),
        source_cache_inventory=old['source_cache_inventory'],
        old_layer_index_path='/mnt/yangx-projects-f/20261010/permworld_controlled_comparison/layers/index.json',
        selection_path=str(ROOT/'fixed_selections.npz'),evaluation_index_path=str(ROOT/'evaluation/index.json'),
        metrics_source_path=str(original.METRICS_PATH),metrics_source_sha256=sha(original.METRICS_PATH),
        execution='One GPU serial missing-source jobs; exact old optimizer/batches/budget; no geometry or accuracy endpoint gate',
        graph_stage='Only fixed4 graph models after complete PermWorld training and analysis',
        no_F17_changes=True,no_release_changes=True,no_old_deletion=True,no_automatic_publishing=True,
        dependence='One world, one initialization per arch;64models/32relation cells/8instances/21lengths/3metrics not independent; only inverse+complement types',
        stopping='Complete bounded PermWorld8relations and graph4tasks plus reader reports/entry/index, then stop')
    c['old_table_sha256']={name:sha(OLD_ANALYSIS/name) for name in [
        'per_length_results.csv','cross_arch_same_task.csv','gold_baseline_length.csv',
        'source_learning.csv','source_validation_eos_original.csv']}
    c['cache_root']=str(ROOT/'evaluation')
    for job in missing:
        arch=job['architecture'];ref=inits[arch]
        c['checkpoint_inventory'].append(dict(
            run_id=job['run_id'],architecture=arch,encoding=job['encoding'],task=job['task'],
            model_seed=job['model_seed'],checkpoint_path=str(ROOT/'models'/job['run_id']/'checkpoint.pt'),
            completed_path=str(ROOT/'models'/job['run_id']/'completed.json'),
            initial_checkpoint_path=ref['path'],initial_checkpoint_sha256=ref['sha256'],
            initial_parameter_sha256=ref['parameter_sha256'],init_evidence=ref['evidence'],
            parameters=js(Path(ref['path']).with_suffix('.json'))['parameters'],
            exposures=1280000,supervised_tokens=2560000,updates=20000,
            starting_rng_hash=starts[arch]['rng_hash'],starting_rng_path=starts[arch]['path'],
            starting_rng_checkpoint_sha256=starts[arch]['sha256'],
            final_hash_status='Future completion binds final SHA; frozen plan is never rewritten',
            reused_existing=False))
    assert len(c['checkpoint_inventory'])==64
    with np.load(data_path,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    saved={};selection_rows=[]
    for r in RELATIONS:
        for control in c['controls']:
            for n in c['lengths']:
                for split in c['splits']:
                    ids,keys=selections(data,r,split,n,control);name=f'{r["pair_id"]}_{control}_{n}_{split}'
                    saved[name+'_indices']=ids
                    if keys is not None:saved[name+'_keys']=keys
                    selection_rows.append(dict(relation=r['pair_id'],control=control,length=n,split=split,samples=len(ids),
                        indices_sha256=array_sha(ids),keys_sha256='none' if keys is None else array_sha(keys),right_label_offset=r['right_label_offset']))
    np.savez(ROOT/'fixed_selections.npz',**saved);c['selection_sha256']=sha(ROOT/'fixed_selections.npz')
    csv_write(DIAG/'selection_index.csv',selection_rows);json_write(CONFIG,c)
    protected=js(OLD_ANALYSIS/'protected_before.json')
    for file,meta in js(OLD_ANALYSIS/'artifact_manifest.json')['files'].items():protected[str(OLD_ANALYSIS/file)]=meta['sha256']
    old_index=Path(c['old_layer_index_path']);protected[str(old_index)]=sha(old_index)
    for rec in js(old_index)['records']:
        for k,v in rec.items():
            if k.endswith('_path') and k[:-5]+'_sha256' in rec:protected[v]=rec[k[:-5]+'_sha256']
    json_write(DIAG/'protected_files.json',dict(files=protected,source_data_sha256=c['data_sha256'],
        policy='Old actual weights/data/source/reports/F17/releases remain unchanged; no deletion or publishing',
        parent_protection_sha256=sha(OLD_ANALYSIS/'protected_before.json')))
    sources={str(WORKSPACE/k):h for k,h in js(SOURCE_ROOT/'freeze_manifest.json')['source_sha256'].items()}
    sources.update({str(Path(__file__).parent/x):sha(Path(__file__).parent/x) for x in ['common.py','train.py','setup.py']})
    json_write(DIAG/'training_freeze.json',dict(protocol_sha256=sha(CONFIG),source_sha256=sources,
        protected_manifest_sha256=sha(DIAG/'protected_files.json'),training_manifest_sha256=sha(DIAG/'training_manifest.csv'),
        before_new_source_training=True,analysis_rules_frozen_before_training=True,
        separate_analysis_source_freeze_before_new_geometry=True))
    print(dict(frozen=True,missing_models=len(missing),models_reused=16,selections=len(selection_rows),protected_files=len(protected)),flush=True)
    return c

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['math','freeze'],required=True)
    a=p.parse_args();math_audit() if a.stage=='math' else freeze()
