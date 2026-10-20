"""Freeze the16 fixed20k models, inherited inputs and controlled contrasts."""
from common import *
import argparse
from datetime import datetime,timezone
import subprocess
import time

def main():
    assert not (DIAG/'protocol.json').exists(),'Existing protocol is immutable'
    code=[DIAG/'common.py',Path(__file__),DIAG/'contrasts.py',DIAG/'source_learning.py',
          DIAG/'extraction_review/extract_layers.py',DIAG/'metric_review/analyze.py',
          DIAG/'metric_review/independent_metrics.py',DIAG/'inventory/report.py']
    assert all(p.exists() for p in code),'Wait for diagnostic implementations before setup'
    start=time.monotonic();FROOT.mkdir(parents=True,exist_ok=True)
    inventory=js(DIAG/'inventory/sources.json')
    jobs=inventory['checkpoint_inventory'];assert len(jobs)==16
    source=js(SOURCE_CONFIG)
    frozen=js(SOURCE_ROOT/'freeze_manifest.json')
    assert sha(SOURCE_CONFIG)==frozen['protocol_sha256']
    for p,h in frozen['source_sha256'].items():assert sha(WORKSPACE/p)==h
    source_version=payload_sha(frozen['source_sha256'])
    data_path=SOURCE_ROOT/'worlds/archdev001/data.npz'
    data_audit=js(data_path.with_name('data_audit.json'))
    assert sha(data_path)==data_audit['data_sha256']
    protected=js(WORKSPACE/'diagnostics/permworld_fullset_eval_20261010/protected_before.json')
    for root in [WORKSPACE/'diagnostics/permworld_layers_20261009',
                 WORKSPACE/'diagnostics/permworld_learnability_20261010',
                 WORKSPACE/'diagnostics/permworld_fullset_eval_20261010',SOURCE_ROOT]:
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:protected[str(p)]=sha(p)
    json_write(DIAG/'protected_before.json',protected)
    with np.load(data_path,allow_pickle=False) as z:
        data={f'{s}_{name}':z[f'{s}_{name}'] for s in ['validation','test'] for name in
              ['permutations','lengths','labels','one_line_input','canonical_cycle_input']}
    lengths=list(range(10,31));assert source['lengths']==lengths
    anchors=[];orbit_sets={}
    for split in ['validation','test']:
        orbit_sets[split]=set()
        for i,(p,n) in enumerate(zip(data[split+'_permutations'],data[split+'_lengths'])):
            obj=tuple(map(int,p[:int(n)]));orbit_id=hashlib.sha256(orbit_key(obj)).hexdigest()
            assert orbit_id not in orbit_sets[split];orbit_sets[split].add(orbit_id)
            anchors.append(dict(split=split,row_index=i,length=int(n),anchor_id=fingerprint(obj).hex(),orbit_id=orbit_id))
    assert not (orbit_sets['validation']&orbit_sets['test'])
    csv_write(DIAG/'anchor_index.csv',anchors)
    selected={};index=[]
    for rel in source['relations']:
        for control in source['evaluation']['controls']:
            for n in lengths:
                for split in ['validation','test']:
                    use,keys=selections(data,rel,split,n,control);ids=np.flatnonzero(use).astype(np.int64)
                    name=f"{rel['pair_id']}_{control}_{n}_{split}"
                    selected[name+'_indices']=ids
                    if keys is not None:selected[name+'_keys']=keys
                    index.append(dict(relation=rel['pair_id'],control=control,length=n,split=split,
                        samples=len(ids),indices_sha256=array_sha(ids),
                        keys_sha256=array_sha(keys) if keys is not None else 'none',selection_key=name))
    selection_path=FROOT/'fixed_selections.npz'
    np.savez_compressed(selection_path,**selected)
    csv_write(DIAG/'selection_index.csv',index)
    pairhash={}
    for rel in source['relations']:
        correct=ACTIONS.index(rel['input_transform']);wrong=1 if correct==2 else 2
        for enc in source['encodings']:
            for control in source['evaluation']['controls']:
                for role,ai in [('correct',correct),('wrong',wrong),('identity',0)]:
                    payload=[]
                    for n in lengths:
                        for split in ['validation','test']:
                            name=f"{rel['pair_id']}_{control}_{n}_{split}"
                            ids=selected[name+'_indices'];inputs=data[f'{split}_{enc}_input']
                            payload.append([n,split,array_sha(ids),array_sha(inputs[ids,0]),array_sha(inputs[ids,ai])])
                    pairhash[rel['pair_id']+'|'+enc+'|'+control+'|'+role]=payload_sha(payload)
    json_write(DIAG/'input_pairing_hashes.json',pairhash)
    cache_index=inventory['caches']
    def endpoint(a,e,rel,layer,metric,control,role):
        pairjobs=[j for j in jobs if j['architecture']==a and j['encoding']==e and j['task'] in [rel['left'],rel['right']]]
        caches=[c for c in cache_index if c['architecture']==a and c['encoding']==e and c['task'] in [rel['left'],rel['right']]]
        return dict(architecture=a,encoding=e,task_pair=[rel['left'],rel['right']],layer=layer,
            pooling='input_end_before_task',metric=metric,pairing=role,control=control,
            checkpoints=[{k:j[k] for k in ['run_id','checkpoint_sha256','initial_checkpoint_sha256',
                'parameter_sha256','initial_parameter_sha256','vocabulary_sha256','init_evidence']} for j in pairjobs],
            feature_files=[{k:c[k] for k in ['path','sha256','condition','split','task']} for c in caches]
                if layer=='final_norm' else [{'status':'pending_before_extraction','root':str(FROOT),'layer':layer}],
            input_pairing_sha256=pairhash[rel['pair_id']+'|'+e+'|'+control+'|'+role],
            metric_preprocessing=source['evaluation']['answer_strata'] if control=='answer_strata' else 'within length split centering')
    manifest=[]
    def add(kind,left,right,changed,unmatched,status):
        manifest.append(dict(comparison_id=f'{kind}{len(manifest):04d}',comparison_class=kind,
            relation='--'.join(left['task_pair']),
            left=json.dumps(left,ensure_ascii=False,sort_keys=True),right=json.dumps(right,ensure_ascii=False,sort_keys=True),
            lengths='10..30',length_weights='1/21 each; NaN on incomplete coverage',split='validation fit / test eval',
            row_order='source anchor index within each fixed length, never permuted',
            data_sha256=data_audit['data_sha256'],batch_indices_sha256=data_audit['batch_indices_sha256'],
            supervision=source['training']['loss'],updates=20000,sample_exposures=1280000,
            checkpoint_selection=source['training']['endpoint'],
            intended_changed_fields=json.dumps(changed,ensure_ascii=False),unmatched_fields=json.dumps(unmatched,ensure_ascii=False),match_status=status))
    for rel in source['relations']:
        for control in source['evaluation']['controls']:
            for role in ['correct','wrong','identity']:
                for a in source['architectures']:
                    for e in source['encodings']:
                        base=endpoint(a,e,rel,'final_norm','linear_cka',control,role)
                        for metric in ['knn_overlap','procrustes','debiased_cka']:
                            other=endpoint(a,e,rel,'final_norm',metric,control,role)
                            add('A',base,other,['metric','metric-specific preprocessing'],[], 'strictly_matched')
                        last=('block4' if a=='transformer' else 'layer4')
                        for layer in [('block'+str(i) if a=='transformer' else 'layer'+str(i)) for i in range(1,4)]+['final_norm','embedding']:
                            norm=layer=='final_norm'
                            add('D',endpoint(a,e,rel,last,'linear_cka',control,role),endpoint(a,e,rel,layer,'linear_cka',control,role),
                                ['final normalization'] if norm else ['layer'],[],
                                'incomparable_or_missing' if layer=='embedding' else 'strictly_matched')
                    add('C',endpoint(a,'one_line',rel,'final_norm','linear_cka',control,role),endpoint(a,'canonical_cycle',rel,'final_norm','linear_cka',control,role),
                        ['encoding grammar','serialized permutation','end token'],['task mastery not matched'], 'strictly_matched')
                for e in source['encodings']:
                    add('B',endpoint('transformer',e,rel,'final_norm','linear_cka',control,role),endpoint('gru',e,rel,'final_norm','linear_cka',control,role),
                        ['model configuration'],['parameter count','computational path','architecture initialization','task mastery'], 'configuration comparison')
    # Isolated secondary comparison: same task/input across model configurations.
    cross_input_hashes={}
    for task in source['tasks']:
        for e in source['encodings']:
            for control in source['evaluation']['controls']:
                payload=[]
                for n in lengths:
                    for split in ['validation','test']:
                        ns=data[split+'_lengths'];y=data[split+'_labels'][:,:,TASKS.index(task)]
                        keys=np.column_stack([ns,y]);use=ns==n
                        if control=='answer_strata':
                            _,inv,count=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
                            use &= count[inv]>=3
                        ids=np.flatnonzero(use)
                        payload.append([n,split,array_sha(ids),array_sha(data[f'{split}_{e}_input'][ids,0]),
                                        array_sha(keys[ids]) if control=='answer_strata' else 'none'])
                ph=payload_sha(payload);cross_input_hashes[task+'|'+e+'|'+control]=ph
                for metric in ['linear_cka','debiased_cka']:
                    sides=[]
                    for a in source['architectures']:
                        j=next(j for j in jobs if (j['architecture'],j['encoding'],j['task'])==(a,e,task))
                        cs=[c for c in cache_index if (c['architecture'],c['encoding'],c['task'])==(a,e,task)]
                        sides.append(dict(architecture=a,encoding=e,task_pair=[task],layer='final_norm',
                            pooling='input_end_before_task',metric=metric,pairing='identity',control=control,
                            checkpoints=[{k:j[k] for k in ['run_id','checkpoint_sha256','initial_checkpoint_sha256',
                                'parameter_sha256','initial_parameter_sha256','vocabulary_sha256','init_evidence']}],
                            feature_files=[{k:c[k] for k in ['path','sha256','condition','split','task']} for c in cs],
                            input_pairing_sha256=ph,metric_preprocessing='joint[n,same-task identity/complement/inverse] min3 per split; isolated secondary control'))
                    add('X_same_task',*sides,['model configuration'],['parameter count','computational path','initialization','task mastery'],'configuration comparison')
    json_write(DIAG/'cross_arch_input_pairing_hashes.json',cross_input_hashes)
    csv_write(DIAG/'comparison_manifest_before_extraction.csv',manifest)
    csv_write(DIAG/'comparison_manifest.csv',manifest)
    protocol=dict(version='permworld-controlled-comparison/v1',registered_utc=datetime.now(timezone.utc).isoformat(),
        exploratory=True,prior_results_seen=True,source_root=str(SOURCE_ROOT),source_protocol_path=str(SOURCE_CONFIG),
        source_protocol_sha256=sha(SOURCE_CONFIG),source_version_sha256=source_version,
        source_data_path=str(data_path),source_data_sha256=data_audit['data_sha256'],
        cache_root=str(FROOT),checkpoint_inventory=jobs,source_cache_inventory=cache_index,
        selection_path=str(selection_path),selection_sha256=sha(selection_path),anchor_index_sha256=sha(DIAG/'anchor_index.csv'),
        input_pairing_hashes_sha256=sha(DIAG/'input_pairing_hashes.json'),
        lengths=lengths,length_weights={str(n):1/21 for n in lengths},relations=source['relations'],
        architectures=source['architectures'],encodings=source['encodings'],tasks=source['tasks'],
        splits=['validation','test'],actions=list(ACTIONS),controls=source['evaluation']['controls'],
        layer_rules={'transformer':['block1','block2','block3','block4','final_norm','embedding'],
                     'gru':['layer1','layer2','layer3','layer4','final_norm','embedding']},
        primary_layer='final_norm',pooling='input_end_before_task_at2n+3',
        metrics={'primary':'linear_cka','other':['knn_overlap','procrustes'],'sensitivity':'debiased_cka',
            'knn_k':10,'neighbor_rule':'Squared Euclidean; stable global anchor ordinal ties; exclude self and duplicate anchor permutations',
            'Procrustes':'fit-only PCA rank=min(8,d1,d2), fit RMS scaling, fit orthogonal; eval independently centered as frozen oracle measurement',
            'preprocessing':'fit and eval each centered independently; answer_strata keys [n,left identity,right identity/complement/inverse], min3 separately per split',
            'new_layer_metrics':['linear_cka','debiased_cka'],
            'undefined':'NaN plus reason; no shortened-length renormalization',
            'static_gold':'no fictitious initial/trained Delta','output_prob':'numeric0..30 conditional probabilities, not full-vocab probability'},
        cross_arch_same_task={'metrics':['linear_cka','debiased_cka'],'pairing':'identity--identity',
            'controls':['raw_within_length','answer_strata'],
            'answer_strata':'joint[n,same task identity/complement/inverse] min3; differs from relation-task joint grouping, isolated secondary analysis'},
        validation={'fixed_lengths':[10,20,30],'first_anchors_per_length':8,'activation_abs_tolerance':2e-5,
            'metric_abs_tolerance':1e-8,'identity_decomposition_abs_tolerance':1e-10,
            'metric_fixture':{'architecture':'transformer','encoding':'one_line','relation':'descent_inverse',
                'length':10,'controls':['raw_within_length','answer_strata'],'fit_rows':64,'eval_rows':64,'metric_abs_tolerance':1e-8,
                'selection':'first retained rows after fixed within-length preprocessing; both controls, all phases and pairings'}},
        inference={'batch_size':128,'eval':True,'autocast':False,'FP32':True,'TF32':False,'optimizer':False,'backward':False},
        learning={'scope':'original validation and already-frozen geometry evaluation anchors; no model selection',
            'new_answer_CE':'all188 vocabulary, measured on fixed original prompts',
            'EOS':'old teacher-forced validation statistic only, clearly separate; no new EOS path',
            'length_mode_baseline':'train-label fixed mode smallest tie; same old frozen test supplementary baseline'},
        history={'old82':'separate historical only; reconstructed initialization original identity unverified',
                 'second8k':'excluded from every main table'},
        no_new_training=True,no_new_inputs=True,no_optimizer=True,no_GroupWorld=True,no_F17_changes=True,
        stopping='Complete feasible fixed comparisons, independent checks and report; stop.',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKSPACE,text=True).strip())
    json_write(DIAG/'protocol.json',protocol)
    json_write(DIAG/'freeze.json',dict(protocol_sha256=sha(DIAG/'protocol.json'),
        comparison_manifest_sha256=sha(DIAG/'comparison_manifest_before_extraction.csv'),
        protected_manifest_sha256=sha(DIAG/'protected_before.json'),
        source_sha256={**{str(WORKSPACE/p):h for p,h in frozen['source_sha256'].items()},**{str(p):sha(p) for p in code}},
        scientific_flow_frozen_before_new_extraction=True,prior_results_seen=True))
    json_write(DIAG/'setup_audit.json',dict(status='passed',models=16,lengths=21,anchor_rows=len(anchors),
        selections=len(index),comparisons=len(manifest),protected_files=len(protected),elapsed_seconds=time.monotonic()-start,
        validation_test_orbits_disjoint=True,training_orbit_proof_reused=str(data_path.with_name('data_audit.json')),
        source_training_updates=20000,new_training=False))
    print(dict(setup='passed',models=16,comparisons=len(manifest),protected_files=len(protected)),flush=True)

if __name__=='__main__':main()
