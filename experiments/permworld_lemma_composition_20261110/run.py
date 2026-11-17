"""Frozen source-position readouts composed without access to permutations."""
from pathlib import Path
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .calibration import select_threshold,micro_f1
from .combiners import compose_batch
from .oracles import oracle_tasks

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(__file__).resolve().parent
DIAG=ROOT/'diagnostics/permworld_lemma_composition_20261110'
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_lemma_composition')
PROBE=ROOT/'experiments/permworld_peaks_probe_20261109/vendor'
ALIGN=ROOT/'diagnostics/permworld_orthogonal_alignment_20261110'
ROUTES=['same_model_twice','aligned_complement','direct_transfer','means_only','map_identity','map_inverse',*[f'map_shuffle_{s}' for s in range(1026,1034)],'own_probes']
SEEDS=list(range(611101,611109))
VARIANTS=['original',*[f'turn_shuffle_{s}' for s in SEEDS],*[f'fixed_shuffle_{s}' for s in SEEDS]]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def array_sha(array):return hashlib.sha256(np.asarray(array).tobytes(order='C')).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def freeze():
    if (DIAG/'protocol.json').exists():return config()
    audit=read(DIAG/'cache_audit.json');assert audit['status']=='passed'
    c=dict(version='permworld-frozen-lemma-composition/v1',created_utc=datetime.now(timezone.utc).isoformat(),client_date='2026-11-10',
        output=str(OUT),units=[dict(unit=u['unit'],historical=u['historical'],data_path=u['data_path'],data_sha256=u['data_sha256']) for u in audit['units']],
        architecture='gru',encoding='one_line',lengths=list(range(10,31)),length_weight='1/21',routes=ROUTES,variants=VARIANTS,mask_shuffle_seeds=SEEDS,
        input_definition='Every new question is about pi=C(x). First peaks mask uses peaks testaction1; same-model valley mask uses peaks testaction0. Pairedroute valley mask uses valleys testaction1 under savedtarget-to-source map. Fixedpointtestusesfixed_points testaction1; sourcefitsallusevalidationaction0.',
        source_probe='Rebuild frozen original fit_probe: perlength sourcefit/tuningorbitgroups,StandardScaler,Ridge,cholesky,alphas1/10/100/1000 selected bysourcepositionAUROC. NewFPprobe sees ONLY its own full-position fixedpoint labels including endpoints. No downstream answers or other labels enter fitting.',
        threshold='Per unit/task/length maximize source-tuning micro-F1 over distinct source scores, score>=threshold; high-threshold tie, no-positive source labels ->empty mask. Fit/dev areoriginalsourcevalidationorbit split. Peaks threshold shared by both uses/allmaps; own-valleyreference has ownsource threshold; same FPthreshold/decoder acrossallroutes/tasks. No true-count top-k.',
        map_policy='Reuse savedone_line/trainedR identity/complement/inverse/8count-stratifiedshuffle matrices byte-for-byte. Reconstruct ONLY original fit-validation means; everymap scores same valleys(Cx) features centered by mu_C thenrestoremu_peaks. No newalignment fit.',
        tasks=dict(longest='Longest maximal strictmonotone contiguous interval, inclusive1-based endpoints, tiesleftmost.',fixed_longest='Longest such interval containing a predicted fixedpoint, NONEifnopredictedfixedpoint. Boundaryfixedpoints belongbothadjacent intervals.',turns_before_fixed='Count realinterior turnsmask strictlybeforefirstpredictedfixedpoint, NONEifnone.'),
        composer_interface='ONLY predicted Boolean turn/fixed masks and length. No permutation,features,truecount,truepositions or downstreamlabels. Endpoints added asboundaries,not counted asturns.',
        shuffle_policy='8pre-fixed count-preserving positionpermutations for each masktype; turningpositions interioronly,fixedpositions includeendpoints. Perunit/length/seed/stream permutations independentoflabels andsharedacrossroutes. FPmask commonallroutes. One kind shuffledatatime; no newfitting/calibration. Replicatesarecontrols,notunits.',
        shuffle_rng='default_rng(SeedSequence([seed,unitindex,n,stream])); unitindex1historical,2..5new; stream0turns/1FP; rng.permuted(broadcastarange(N,npositions),axis=1) createsindependentrowpermutations in originalanchororderwithinlength.',
        support='All originalheldouttestanchors and SAME originalmatchedmask onanchor x fromread_length,not recomputedonpi=Cx. Truthusedonlyafterpredictionsfrozen for scoring/stratification.',
        strata='Primaryall/matched. E2 additionallytruthFPpresent/absent; constrainedsegmentdiagnostic truthFPpresent ANDoracleconstrainedinterval!=oracleunconstrainedinterval, excludingtrivialNONEcases. Bothsupports retainstrata; no subset chosenbyperformance.',
        metrics='Primary exact-answeraccuracy. Intervalendpoint/start/end MAE andlengthabs errorconditionalonbothintervalsfinite withcoverage; unconditionalnormalizederror=1ifpresence mismatch,0bothNONE,finiteerror/n. Never silentlyreweightundefinedlengths. CountMAEconditionalbothfinite ancillary.',
        statistics='Fournewunit rawvalues,mean,sampleSD,minmax; originalforengineering/reference only. Coupledlengths,questiontypes,routes,and8shufflesarenot independentreplicates. Reportwithinunitshufflemean andindividualcontrols; no newpvalue/equivalence/CI.',
        numerical=dict(previous_probe_AUROC_absolute=1e-12,map_orthogonality_absolute=2e-12),
        outcomes_policy='Retain everyunit/route and adverseoutcome. No gate ondownstreamaccuracy. Historicalrun validatesimplementationonly; never usedtochoosethresholdrule ornewhyperparameters.',
        operations='Cachedactivations,positionridgefits,source-onlycalibration,predicted-maskcomposition,scoring. No neuralforward,basemodeltraining,newalignment,extraunits,F17/paperchange,automaticpublication orfollow-upexperiment.',
        exploratory=True,prior_readout_results_seen=True,downstream_outcomes_not_yet_computed=True,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    assert [u['unit'] for u in c['units']]==['archdev001','unit02','unit03','unit04','unit05']
    write(DIAG/'protocol.json',c)
    paths=list(PACKAGE.glob('*.py'))+[PROBE/'analyze.py',DIAG/'cache_audit.json']
    paths+=list((ROOT/'tests').glob('test_permworld_lemma_*.py'))
    write(DIAG/'freeze.json',dict(protocol_sha256=sha(DIAG/'protocol.json'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},before_actual_source_threshold_and_downstream_predictions=True))
    protected={p:r['sha256'] for p,r in audit['files_hashed'].items()}
    for d in [ALIGN,ROOT/'diagnostics/permworld_peaks_probe_20261109']:
        for p,r in read(d/'delivery_manifest.json')['files'].items():protected[str(ROOT/p)]=r['sha256']
    for p in [ROOT/'README.md',ROOT/'CURRENT_RESEARCH.txt',ROOT/'manuscripts/representation_selectivity_20261012/manuscript.tex',ROOT/'manuscripts/representation_selectivity_20261012/manuscript.pdf',ROOT/'manuscripts/representation_selectivity_20261012/manuscript_source.zip']:
        protected[str(p)]=sha(p)
    write(DIAG/'protection_start.json',dict(files=protected,git_head=c['git_head'],scope='Named prior scientific sources/results/checkpoints/caches and currentpaper/index bytes. Endpoint check only, not continuous monitoring or every file.'))
    return c


def config():
    c=read(DIAG/'protocol.json');f=read(DIAG/'freeze.json');assert sha(DIAG/'protocol.json')==f['protocol_sha256']
    for p,h in f['source_sha256'].items():assert sha(ROOT/p)==h,('Frozen source changed',p)
    return c


def source_modules():
    sys.path.insert(0,str(PROBE))
    from analyze import fit_probe,read_length,example_auc,average
    return fit_probe,read_length,example_auc,average


def source_array(desc):
    path=Path(desc['path']);assert sha(path)==desc['sha256']
    if desc['field']:
        with np.load(path,allow_pickle=False) as archive:a=archive[desc['field']].copy()
    else:a=np.load(path,mmap_mode='r',allow_pickle=False)
    assert a.shape==tuple(desc['shape']) and a.dtype==np.float32
    return a


def save_probe(probe,task,n,params):
    closure=dict(zip(probe.__code__.co_freevars,[c.cell_contents for c in probe.__closure__]))
    model=closure['best_model'];scaler=closure['scaling'];state={}
    for key,value in dict(coef=model.coef_,intercept=model.intercept_,mean=scaler.mean_,scale=scaler.scale_).items():
        name=f'{task}_n{n}_{key}';params[name]=np.asarray(value).copy();state[key]=array_sha(params[name])
    return state


def permutations(seed,unit,n,count,stream):
    size=n-2 if stream==0 else n
    rng=np.random.default_rng(np.random.SeedSequence([seed,1 if unit=='archdev001' else int(unit[-2:]),n,stream]))
    return rng.permuted(np.broadcast_to(np.arange(size,dtype=np.int16),(count,size)),axis=1)


def downstream_predictions(c,signals):
    ns=signals['lengths'];count=len(ns);r=len(ROUTES);v=len(VARIANTS)
    result=dict(lengths=ns,anchor_indices=np.arange(count),route_names=np.asarray(ROUTES),variant_names=np.asarray(VARIANTS),
        pred_longest=np.empty((r,v,count,2),np.int16),pred_fixed_longest=np.empty((r,v,count,2),np.int16),pred_turns_before_fixed=np.empty((r,v,count),np.int16))
    shuffle_records=[];shuffle_arrays={}
    for n in c['lengths']:
        ids=np.flatnonzero(ns==n);fp=signals['fixed_mask'][ids,:n]
        turn_perms=[];fp_perms=[]
        for seed in SEEDS:
            a=permutations(seed,signals['unit'].item(),n,len(ids),0);b=permutations(seed,signals['unit'].item(),n,len(ids),1)
            turn_perms.append(a);fp_perms.append(b);shuffle_arrays[f'turn_n{n}_s{seed}']=a;shuffle_arrays[f'fixed_n{n}_s{seed}']=b
            shuffle_records.append(dict(length=n,seed=seed,rows=len(ids),turn_permutation_sha256=array_sha(a),fixed_permutation_sha256=array_sha(b)))
        for ri in range(r):
            turns=signals['turn_masks'][ri,ids,:n]
            masks=[(turns,fp)]
            for a in turn_perms:
                shuffled=turns.copy();shuffled[:,1:-1]=np.take_along_axis(turns[:,1:-1],a,axis=1)
                assert np.array_equal(shuffled.sum(1),turns.sum(1));masks.append((shuffled,fp))
            for b in fp_perms:
                shuffled=np.take_along_axis(fp,b,axis=1)
                assert np.array_equal(shuffled.sum(1),fp.sum(1));masks.append((turns,shuffled))
            for vi,(turn_mask,fixed_mask) in enumerate(masks):
                answers=compose_batch(turn_mask,fixed_mask,n)
                for task in ['longest','fixed_longest','turns_before_fixed']:result['pred_'+task][ri,vi,ids]=answers[task]
            for vi in range(9,17):assert np.array_equal(result['pred_longest'][ri,vi,ids],result['pred_longest'][ri,0,ids])
    return result,shuffle_records,shuffle_arrays


def oracle_truth(data,lengths):
    count=len(lengths);truth=dict(truth_longest=np.empty((count,2),np.int16),truth_fixed_longest=np.full((count,2),-1,np.int16),truth_turns_before_fixed=np.full(count,-1,np.int16),truth_fixed_present=np.zeros(count,bool),truth_constraint_changes=np.zeros(count,bool))
    pi_all=np.zeros((count,30),np.int16)
    for i,n in enumerate(lengths):
        pi=(int(n)+1-data['test_permutations'][i,:n]).astype(np.int16);pi_all[i,:n]=pi
        answer=oracle_tasks(pi.tolist());truth['truth_longest'][i]=answer['longest']
        if answer['fixed_longest'] is not None:
            truth['truth_fixed_longest'][i]=answer['fixed_longest'];truth['truth_turns_before_fixed'][i]=answer['turns_before_fixed'];truth['truth_fixed_present'][i]=True
            truth['truth_constraint_changes'][i]=answer['fixed_longest']!=answer['longest']
    # Algorithm verification only; truth masks are never fed to formal routes.
    for n in range(10,31):
        ids=np.flatnonzero(lengths==n);pi=pi_all[ids,:n];turns=np.zeros_like(pi,dtype=bool)
        turns[:,1:-1]=((pi[:,1:-1]>pi[:,:-2])&(pi[:,1:-1]>pi[:,2:]))|((pi[:,1:-1]<pi[:,:-2])&(pi[:,1:-1]<pi[:,2:]))
        fixed=pi==np.arange(1,n+1);check=compose_batch(turns,fixed,n)
        for key,value in check.items():assert np.array_equal(value,truth['truth_'+key][ids]),('Independent oracle disagreement',n,key)
    return truth,pi_all


def execute(c,u):
    unit=u['unit'];dest=OUT/'units'/unit;dest.mkdir(parents=True,exist_ok=True);record_dir=DIAG/'results'/unit;done=record_dir/'completion.json'
    if done.exists():
        receipt=read(done);assert receipt['protocol_sha256']==sha(DIAG/'protocol.json')
        for desc in receipt['artifacts'].values():assert sha(desc['path'])==desc['sha256']
        return receipt
    started=time.monotonic();audit_unit=next(x for x in read(DIAG/'cache_audit.json')['units'] if x['unit']==unit)
    assert sha(u['data_path'])==u['data_sha256']
    with np.load(u['data_path'],allow_pickle=False) as archive:data={k:archive[k].copy() for k in archive.files if k.endswith(('_lengths','_permutations'))}
    raw={(d['task'],d['split']):source_array(d) for d in audit_unit['raw_features']}
    alignment=audit_unit['alignment'];assert sha(alignment['archive_path'])==alignment['archive_sha256']
    maps={}
    with np.load(alignment['archive_path'],allow_pickle=False) as archive:
        for desc in alignment['maps']:
            q=archive[desc['key']].copy();assert array_sha(q)==desc['map_sha256'];assert np.max(np.abs(q.T@q-np.eye(256)))<=2e-12;maps[desc['fit_condition']]=q
    ns=data['test_lengths'];count=len(ns);fit_probe,read_length,example_auc,average=source_modules()
    signals=dict(unit=np.asarray(unit),lengths=ns,peak_scores=np.full((count,30),np.nan),valley_scores=np.full((len(ROUTES),count,30),np.nan),fixed_scores=np.full((count,30),np.nan),
        peak_mask=np.zeros((count,30),bool),valley_masks=np.zeros((len(ROUTES),count,30),bool),turn_masks=np.zeros((len(ROUTES),count,30),bool),fixed_mask=np.zeros((count,30),bool))
    matched=np.zeros(count,bool);eligible=np.zeros(count,bool);params={};calibrations=[];component=[];baseline_checks=[]
    import csv
    with (ALIGN/'results'/unit/'by_length.csv').open(newline='') as stream:previous=list(csv.DictReader(stream))
    previous={(int(r['length']),r['method'],r['support']):float(r['auc']) for r in previous if r['encoding']=='one_line' and r['phase']=='trained'}
    methods={'same_model_twice':'within_peaks','aligned_complement':'fit_complement__test_complement','direct_transfer':'direct','means_only':'means_only','map_identity':'fit_identity__test_complement','map_inverse':'fit_inverse__test_complement','own_probes':'within_valleys',**{f'map_shuffle_{s}':f'fit_shuffle_{s}__test_complement' for s in range(1026,1034)}}
    for n in c['lengths']:
        fit,dev,test,common,peak_val,valley_val,peak_x,_=read_length(data,n)
        val=data['validation_lengths']==n;ids=np.flatnonzero(ns==n);eligible[ids]=test;matched[ids]=common
        fp_val=data['validation_permutations'][val,:n]==np.arange(1,n+1)
        targets={'peaks':peak_val,'valleys':valley_val,'fixed_points':fp_val};probes={};thresholds={}
        for task,y in targets.items():
            x=raw[task,'validation'][val,0];probe,alpha=fit_probe(x,y,fit,dev)
            tuning_scores=probe(x[dev]);threshold,info=select_threshold(tuning_scores,y[dev]);probes[task]=probe;thresholds[task]=threshold
            params[f'{task}_n{n}_source_tuning_scores']=tuning_scores.copy();params[f'{task}_n{n}_source_tuning_targets']=y[dev].copy()
            params[f'{task}_n{n}_source_fit_indices']=np.flatnonzero(val)[fit];params[f'{task}_n{n}_source_tuning_indices']=np.flatnonzero(val)[dev]
            state=save_probe(probe,task,n,params);calibrations.append(dict(unit=unit,task=task,length=n,alpha=alpha,probe_fit_examples=int(fit.sum()),source_tuning_examples=int(dev.sum()),threshold_info=info,probe_parameter_hashes=state,source_tuning_scores_sha256=array_sha(tuning_scores),source_tuning_targets_sha256=array_sha(y[dev]),uses_downstream_answers=False,source_validation_action=0))
        p,q,f=probes['peaks'],probes['valleys'],probes['fixed_points']
        xp=raw['peaks','validation'][val,0];yv=raw['valleys','validation'][val,1]
        mx=xp[fit].astype(np.float64).mean(0);mc=yv[fit].astype(np.float64).mean(0);params[f'n{n}_mu_peaks']=mx;params[f'n{n}_mu_valleys_complement']=mc
        peak_pi=p(raw['peaks','test'][ids,1]);peak_original=p(raw['peaks','test'][ids,0]);valley_pi=raw['valleys','test'][ids,1];centered=valley_pi-mc
        scores={'same_model_twice':peak_original,'aligned_complement':p(centered@maps['complement']+mx),'direct_transfer':p(valley_pi),'means_only':p(centered+mx),'map_identity':p(centered@maps['identity']+mx),'map_inverse':p(centered@maps['inverse']+mx),'own_probes':q(valley_pi)}
        scores.update({f'map_shuffle_{s}':p(centered@maps[f'shuffle_{s}']+mx) for s in range(1026,1034)})
        fp_pi=f(raw['fixed_points','test'][ids,1]);assert np.isfinite(peak_pi).all() and np.isfinite(fp_pi).all()
        signals['peak_scores'][ids,1:n-1]=peak_pi;signals['peak_mask'][ids,1:n-1]=peak_pi>=thresholds['peaks']
        signals['fixed_scores'][ids,:n]=fp_pi;signals['fixed_mask'][ids,:n]=fp_pi>=thresholds['fixed_points']
        for ri,route in enumerate(ROUTES):
            z=scores[route];assert np.isfinite(z).all();tau=thresholds['valleys'] if route=='own_probes' else thresholds['peaks']
            signals['valley_scores'][ri,ids,1:n-1]=z;signals['valley_masks'][ri,ids,1:n-1]=z>=tau
            signals['turn_masks'][ri,ids,:n]=signals['peak_mask'][ids,:n]|signals['valley_masks'][ri,ids,:n]
            auc=example_auc(z,peak_x)
            for support,selection in [('eligible',test),('matched',common)]:
                value=average(auc[selection]);expected=previous[n,methods[route],support];error=abs(value-expected)
                assert error<=c['numerical']['previous_probe_AUROC_absolute'],('Reused probe/map changed',unit,n,route,error)
                baseline_checks.append(dict(length=n,route=route,support=support,actual=value,previous=expected,error=error))
        print(json.dumps(dict(event='position_readouts',unit=unit,length=n)),flush=True)
    assert eligible.all(),'Unexpected original test/train orbit overlap; do not score leaked rows'
    signals['route_names']=np.asarray(ROUTES)
    np.savez_compressed(dest/'signals.npz',**signals);np.savez_compressed(dest/'frozen_readouts.npz',**params)
    write(record_dir/'source_calibration.json',dict(records=calibrations,source_only=True,thresholds_frozen_across_routes_tasks_and_ablations=True))
    write(record_dir/'upstream_reproduction.json',dict(status='passed',checks=baseline_checks,maximum_AUROC_difference=max(x['error'] for x in baseline_checks)))
    predictions,shuffle_records,shuffle_arrays=downstream_predictions(c,signals)
    np.savez_compressed(dest/'prediction_only.npz',**predictions)
    frozen_prediction_sha=sha(dest/'prediction_only.npz')
    # Downstream answers are computed only after prediction-only bytes exist.
    truth,pi=oracle_truth(data,ns)
    np.savez_compressed(dest/'truth.npz',**truth,pi=pi,lengths=ns,anchor_indices=np.arange(count),matched=matched)
    np.savez_compressed(dest/'mask_shuffle_permutations.npz',**shuffle_arrays)
    write(record_dir/'mask_shuffle_audit.json',dict(seeds=SEEDS,records=shuffle_records,uses_only_length_unit_seed_and_row_order=True,predicted_counts_preserved=True))
    scored=dict(predictions,**truth,matched=matched,eligible=eligible)
    np.savez_compressed(dest/'predictions.npz',**scored)
    # Scoring-only component diagnostics are not used to change masks.
    for n in c['lengths']:
        ids=np.flatnonzero(ns==n);a=pi[ids,:n];peak=(a[:,1:-1]>a[:,:-2])&(a[:,1:-1]>a[:,2:]);valley=(a[:,1:-1]<a[:,:-2])&(a[:,1:-1]<a[:,2:]);fixed=a==np.arange(1,n+1)
        for task,s,mask,label in [('peaks_on_pi',signals['peak_scores'][ids,1:n-1],signals['peak_mask'][ids,1:n-1],peak),('fixed_on_pi',signals['fixed_scores'][ids,:n],signals['fixed_mask'][ids,:n],fixed)]:
            component.append(dict(length=n,component=task,route='shared',position_micro_f1=micro_f1(mask,label),mean_predicted_count=float(mask.sum(1).mean()),mean_true_count=float(label.sum(1).mean()),position_AUROC=average(example_auc(s,label))))
        for ri,route in enumerate(ROUTES):
            s=signals['valley_scores'][ri,ids,1:n-1];mask=signals['valley_masks'][ri,ids,1:n-1]
            component.append(dict(length=n,component='valleys_on_pi',route=route,position_micro_f1=micro_f1(mask,valley),mean_predicted_count=float(mask.sum(1).mean()),mean_true_count=float(valley.sum(1).mean()),position_AUROC=average(example_auc(s,valley))))
    with (record_dir/'component_diagnostics.csv').open('w',newline='') as stream:w=csv.DictWriter(stream,fieldnames=list(component[0]));w.writeheader();w.writerows(component)
    artifacts={p.stem:dict(path=str(p),sha256=sha(p),size_bytes=p.stat().st_size) for p in dest.glob('*.npz')}
    assert artifacts['prediction_only']['sha256']==frozen_prediction_sha
    receipt=dict(status='completed',unit=unit,protocol_sha256=sha(DIAG/'protocol.json'),elapsed_seconds=time.monotonic()-started,
        predictions=artifacts['predictions'],artifacts=artifacts,matched_anchors=int(matched.sum()),test_anchors=count,
        independent_oracle_checked_on_all_test_inputs=True,prediction_only_sha_before_truth=frozen_prediction_sha,
        no_neural_forward=True,no_base_training=True,no_new_alignment=True,no_downstream_labels_in_fitting_or_thresholding=True)
    write(done,receipt);print(json.dumps(dict(event='unit_complete',unit=unit,elapsed_seconds=receipt['elapsed_seconds'],matched=int(matched.sum()))),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','historical','new','all']);args=parser.parse_args()
    if args.stage=='freeze':freeze();return
    c=config()
    with threadpool_limits(limits=2):
        try:
            if args.stage in ['historical','all']:execute(c,c['units'][0])
            if args.stage in ['new','all']:
                assert read(DIAG/'results/archdev001/completion.json')['status']=='completed'
                for u in c['units'][1:]:execute(c,u)
            write(DIAG/'run_completion.json',dict(status='completed',stage=args.stage,no_neural_forward=True,no_base_training=True,no_new_alignment=True,stopped_after_fixed_units=True))
        except Exception as error:
            write(DIAG/'failure.json',dict(status='failed',stage=args.stage,type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':main()
