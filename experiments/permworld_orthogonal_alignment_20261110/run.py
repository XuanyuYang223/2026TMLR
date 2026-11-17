"""Cache-only adaptation and fixed orthogonal-alignment reproduction."""
from pathlib import Path
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(__file__).resolve().parent
DIAG=ROOT/'diagnostics/permworld_orthogonal_alignment_20261110'
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_orthogonal_alignment')
PRIOR=ROOT/'diagnostics/permworld_peaks_probe_20261109'
PROBE_VENDOR=ROOT/'experiments/permworld_peaks_probe_20261109/vendor'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def rows(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def configure():
    if (DIAG/'protocol.json').exists():return config()
    a=read(DIAG/'cache_audit.json');assert a['status']=='passed' and a['all_standardized_files_reusable']
    assert read(DIAG/'math_design_audit.json')['status']=='passed'
    h=a['historical'];allunits=[dict(unit='archdev001',data_path=h['data_path'],data_sha256=h['data_sha256'],historical=True)]
    allunits += [dict(unit=u['unit'],root=str(Path(u['path']).parent),data_path=u['path'],data_sha256=u['sha256'],historical=False) for u in a['data_units']]
    for u in allunits:
        first=next(r for r in a['standardized_features'] if r['unit']==u['unit']);u['features']=str(Path(first['path']).parent)
    c=dict(version='permworld-orthogonal-alignment-cache-repetition/v1',created_utc=datetime.now(timezone.utc).isoformat(),
        client_date='2026-11-10',output=str(OUT),units=allunits,architectures=['gru'],encodings=['canonical_cycle','one_line'],
        lengths=list(range(10,31)),actions=['identity','complement','inverse'],shuffle_seeds=list(range(1026,1034)),
        supplied_archive_sha256=read(DIAG/'imported_archive_manifest.json')['archive_sha256'],
        feature_policy='Reuse previous schema-adapted rawFP32 features byte-for-byte. No global zscore. Missing new-unit copies may be rebuilt with original importer into a NEW output directory, never overwriting prior files. Hash mismatch stops.',
        validation_adaptation='prefix_final action1=complement,action2=inverse; peaks initial or valleys trained; arange original validation_indices; actual CP/data/parameter/source binding verified. One-line descents/recoils trained action1 only.',
        fit_policy='Unchanged alignment.py/other_specialists.py. One global raw-coordinate O(256) map per unit/phase/encoding/fit condition; fit means and equal-length cross-products use only original probe-fit validation rows. Probes/scalers/alpha selection retain original source-only split.',
        test_policy='Primary comparisons use the SAME held-out valleys complement features, complement validation mean, source mean, frozen source probe and original peak labels. Native-input rows are auxiliary. Matched and all-eligible supports both retained;21lengths equalweighted.',
        B_map=['means_only','fit_identity__test_complement','fit_inverse__test_complement',*[f'fit_shuffle_{s}__test_complement' for s in range(1026,1034)]],
        B_all='max(B_map,initial fit_complement__test_complement,oracle_count_position_prior,and forone_line both direct/procrustes descents/recoils). Within-model source/target probes are positive references,excluded.',
        main_one_line_success='trained alignedC > trained direct AND trained alignedC > B_all; reportcommonmap-onlysuccess separately. Maxima after fixed equal-length aggregation,notmean oflengthwisemaxima.',
        statistics='Four newer independent data/initialization/training bundles: rawunitvalues,mean,sampleSD,min/max,countdirections. Historicalreference separate andall5secondary. One taskpair,8shuffles and21lengths are not independent replications. No new pvalue/equivalence/CI.',
        numerical=dict(dtype='FP64 maps and ridge solves,rawFP32 cached states',strict_orthogonality_absolute=2e-12,strict_orthogonality_relative=0,
            previous_direct_baseline_absolute=1e-12,historical_supplied_AUROC_absolute=.001,
            historical_tolerance_reason='Fixed before new fits; accommodates cached GPU-origin versus historical CPU-forward FP32 states and numerical/library differences, while much smaller than the reported0.138alignmentgain.'),
        controls='Keep means-only,step0,all8individualshuffles,wrong-fit maps,ownprobes,oraclecountprior,andone-lineother specialists. No relabeling or posthoc dropping of candidates,lengths,units.',
        operation_scope='Existing inputs/activations, fileformat adaptation, ridge probes and orthogonal fitting only. No neural forward, extract_validation execution, checkpoint reinitialization, base training, extra units, F17/paper edits or automatic publication.',
        map_instrumentation='Wrap original rotation/shuffle functions without changing returned values; independently record actualR,strictorthogonality,syntheticorientation,count-preservingpermutations. Mapparameterfiles saved externally.',
        historical_gate='Reproduce supplied historical scores from existing caches first; stop on discrepancy without changing metric/tolerance. Direct/own/prior values for everyunit independentlymatch previous completedprobe results.',
        exploratory=True,prior_pilot_seen=True,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(DIAG/'protocol.json',c)
    sources=list(PACKAGE.glob('*.py'))+list((PACKAGE/'vendor').glob('*.py'))+list(PROBE_VENDOR.glob('*.py'))
    sources += [DIAG/'cache_audit.json',DIAG/'math_design_audit.json',DIAG/'imported_archive_manifest.json']
    write(DIAG/'freeze.json',dict(protocol_sha256=sha(DIAG/'protocol.json'),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},before_actual_alignment_and_probe_fits=True))
    protected={p:r['sha256'] for p,r in a['files_hashed'].items()}
    for previous in [PRIOR/'delivery_manifest.json',ROOT/'diagnostics/permworld_answer_controls_20261109/delivery_manifest.json']:
        for p,r in read(previous)['files'].items():protected[str(ROOT/p)]=r['sha256']
    for p in [ROOT/'README.md',ROOT/'CURRENT_RESEARCH.txt',ROOT/'manuscripts/representation_selectivity_20261012/manuscript.tex',ROOT/'manuscripts/representation_selectivity_20261012/manuscript.pdf',ROOT/'manuscripts/representation_selectivity_20261012/manuscript_source.zip']:
        protected[str(p)]=sha(p)
    write(DIAG/'protection_start.json',dict(scope='Explicitly named prior result/code/paper/index files and201sourcecache/data/checkpointbindings. Endpointaudit only, not allfiles orcontinuous monitoring.',files=protected,git_head=c['git_head']))
    return c


def config():
    c=read(DIAG/'protocol.json');f=read(DIAG/'freeze.json');assert sha(DIAG/'protocol.json')==f['protocol_sha256']
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,('Frozen source changed',p)
    return c


def feature_directory(c,u):
    audit=read(DIAG/'cache_audit.json');expected=[r for r in audit['standardized_features'] if r['unit']==u['unit']]
    missing=[r for r in expected if not Path(r['path']).is_file()]
    if not missing:
        for r in expected:assert sha(r['path'])==r['sha256'],('Prior feature hash mismatch',r['path'])
        return Path(u['features']),dict(reused=True,files={r['path']:r['sha256'] for r in expected})
    assert not u['historical'],'Missing historical standardized copies; no forward substitution'
    target=OUT/'reconstructed_features'/u['unit'];target.mkdir(parents=True,exist_ok=True)
    invoke([sys.executable,str(PROBE_VENDOR/'import_repetition.py'),'--unit-dir',u['root'],'--features',str(target)],OUT/'logs'/f'{u["unit"]}_rebuild.log')
    actual={}
    for r in expected:
        path=target/Path(r['path']).name
        with np.load(path,allow_pickle=False) as f:
            assert str(f['checkpoint_sha256'])==r['checkpoint_sha256']
        actual[str(path)]=sha(path)
    return target,dict(reused=False,reason='Missing schema-adapted files; original importer rebuilt into new directory',files=actual)


def adapt_validation(c,u,features):
    audit=read(DIAG/'cache_audit.json')
    sources=audit['historical']['sources'] if u['historical'] else [r for r in audit['validation_sources'] if r['unit']==u['unit']]
    dest=OUT/'validation'/u['unit'];dest.mkdir(parents=True,exist_ok=True);records=[]
    with np.load(u['data_path'],allow_pickle=False) as data:
        n=len(data['validation_lengths']);assert n==5376
        assert all((data['validation_lengths']==j).sum()==256 for j in c['lengths'])
    for r in sources:
        source=Path(r['path']);assert sha(source)==r['payload_sha256']
        if source.suffix=='.npz':
            assert sha(r['metadata_path'])==r['metadata_sha256']
            m=read(r['metadata_path']);s=m['signature']
            for key,expected in dict(data_sha256=u['data_sha256'],architecture='gru',encoding=r['encoding'],task=r['task'],phase=r['phase'],split='validation',engineering=False,checkpoint_sha256=r['checkpoint_sha256']).items():
                assert s[key]==expected,(source,key)
            with np.load(source,allow_pickle=False) as f:h=f['prefix_final'].copy()
        else:h=np.load(source,allow_pickle=False)
        assert h.shape==(n,3,256) and h.dtype==np.float32
        name=r['output_name'];reference_name=name if r['task'] in ['descents','recoils'] else f'{r["encoding"]}-'+('initial' if r['phase']=='initial' else 'valleys')+'.npz'
        with np.load(features/reference_name,allow_pickle=False) as reference:
            assert str(reference['checkpoint_sha256'])==r['checkpoint_sha256']
            assert np.array_equal(reference['validation_indices'],np.arange(n))
            if r['task'] not in ['descents','recoils']:
                assert np.array_equal(reference['validation_0_hidden'],h[:,0]),'Identity slice/order mismatch'
        arrays=dict(validation_indices=np.arange(n),checkpoint_sha256=np.asarray(r['checkpoint_sha256']))
        for action in r['actions_to_export']:arrays[f'validation_{action}_hidden']=h[:,action].copy()
        path=dest/name;np.savez(path,**arrays)
        with np.load(path,allow_pickle=False) as check:
            for action in r['actions_to_export']:assert np.array_equal(check[f'validation_{action}_hidden'],h[:,action])
        records.append(dict(output_path=str(path),output_sha256=sha(path),data_sha256=u['data_sha256'],source_path=str(source),source_sha256=r['payload_sha256'],checkpoint_sha256=r['checkpoint_sha256'],actions=r['actions_to_export'],validation_rows=n,dtype='float32',original_order_verified=True,raw_states_no_additional_normalization=True))
    assert len(records)==6
    write(dest/'adaptation.json',dict(unit=u['unit'],files=records,no_forward=True))
    return dest,records


def invoke(command,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:result=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def check_outputs(c,u,dest):
    raw=rows(dest/'summary.csv');detail=rows(dest/'by_length.csv');other=rows(dest/'other_specialists.csv')
    assert len(raw)==176 and len(detail)==176*21 and len(other)==8
    assert all(int(r['lengths'])==21 and np.isfinite(float(r['auc'])) for r in raw+other)
    prior=rows(PRIOR/'results'/u['unit']/'summary.csv');keys=['encoding','phase','support','method']
    old={(r['encoding'],r['phase'],r['support'],r['method'],r['transform'],r['label']):r for r in prior}
    error=0.;checks=0
    mapping={'direct':('transfer','complement','peak'),'within_peaks':('source_peaks','identity','peak'),'within_valleys':('target_own','complement','own_valley'),'oracle_count_position_prior':('position_prior','identity','peak')}
    for r in raw:
        if r['method'] not in mapping:continue
        oldrow=old[r['encoding'],r['phase'],'all' if r['support']=='eligible' else 'matched',*mapping[r['method']]]
        difference=abs(float(r['auc'])-float(oldrow['auc']));error=max(error,difference);checks+=1
        assert difference<=c['numerical']['previous_direct_baseline_absolute'],('Direct baseline changed',u['unit'],r,difference)
        assert int(r['anchors'])==int(oldrow['n']) and int(r['finite'])==int(oldrow['defined_n'])
    for r in other:
        if r['method']!='direct':continue
        o=old['one_line','trained','all' if r['support']=='eligible' else 'matched',r['target'],'complement','peak']
        difference=abs(float(r['auc'])-float(o['auc']));error=max(error,difference);checks+=1
        assert difference<=c['numerical']['previous_direct_baseline_absolute']
    for stage,nmaps,nshuffle in [('alignment',45,672),('other_specialists',13,168)]:
        m=read(OUT/'map_audits'/u['unit']/(stage+'.json'));assert m['status']=='completed' and m['map_calls']==nmaps and m['shuffle_call_count']==nshuffle
        assert m['maximum_orthogonality_error']<=2e-12
    return dict(direct_own_prior_baseline_checks=checks,maximum_absolute_difference=error,all21lengths=True)


def execute(c,u):
    dest=DIAG/'results'/u['unit'];done=dest/'completion.json'
    if done.exists():
        r=read(done);assert r['protocol_sha256']==sha(DIAG/'protocol.json')
        for p,d in r['result_sha256'].items():assert sha(dest/p)==d
        check_outputs(c,u,dest);return r
    start=time.monotonic();assert sha(u['data_path'])==u['data_sha256']
    features,feature_receipt=feature_directory(c,u);validation,adaptation=adapt_validation(c,u,features)
    for stage in ['alignment','other_specialists']:
        prefix=OUT/'map_audits'/u['unit']/stage
        invoke([sys.executable,PACKAGE/'instrumented.py',stage,u['unit'],prefix,'--data',u['data_path'],'--features',features,'--validation',validation,'--out',dest],OUT/'logs'/f'{u["unit"]}_{stage}.log')
    baseline=check_outputs(c,u,dest)
    if u['historical']:
        differences=[]
        for filename,keys in [('summary.csv',['encoding','phase','support','method']),('other_specialists.csv',['target','support','method'])]:
            observed={tuple(r[k] for k in keys):r for r in rows(dest/filename)};expected=rows(DIAG/'supplied_historical/results'/filename)
            assert len(observed)==len(expected)
            for r in expected:
                o=observed[tuple(r[k] for k in keys)];differences.append(abs(float(o['auc'])-float(r['auc'])))
                for field in ['lengths','anchors','finite']:
                    if field in r:assert int(o[field])==int(r[field])
        assert max(differences)<=c['numerical']['historical_supplied_AUROC_absolute'],('Historical alignment score differs',max(differences))
        write(DIAG/'historical_reproduction.json',dict(status='passed',summary_rows_checked=len(differences),maximum_AUROC_difference=max(differences),fixed_tolerance=c['numerical']['historical_supplied_AUROC_absolute'],previous_probe_baseline=baseline,no_new_forward=True))
    receipt=dict(status='completed',unit=u['unit'],protocol_sha256=sha(DIAG/'protocol.json'),elapsed_seconds=time.monotonic()-start,
        features=feature_receipt,validation_adaptation=adaptation,previous_probe_baseline=baseline,
        result_sha256={p.name:sha(p) for p in dest.glob('*.csv')},
        map_audits={stage:dict(path=str(OUT/'map_audits'/u['unit']/(stage+'.json')),sha256=sha(OUT/'map_audits'/u['unit']/(stage+'.json'))) for stage in ['alignment','other_specialists']},
        no_neural_forward=True,no_base_training=True,all21lengths=True)
    write(done,receipt);print(json.dumps(dict(unit=u['unit'],status='completed',elapsed_seconds=receipt['elapsed_seconds'],baseline=baseline)),flush=True)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','historical','new','all']);args=p.parse_args()
    if args.stage=='freeze':configure();return
    c=config()
    try:
        if args.stage in ['historical','all']:execute(c,c['units'][0])
        if args.stage in ['new','all']:
            assert read(DIAG/'historical_reproduction.json')['status']=='passed'
            for u in c['units'][1:]:execute(c,u)
        write(DIAG/'run_completion.json',dict(status='completed',stage=args.stage,no_neural_forward=True,no_base_training=True,stopped_after_fixed_units=True))
    except Exception as error:
        write(DIAG/'failure.json',dict(status='failed',stage=args.stage,type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':main()
