"""Orchestrate the unchanged supplied probes; never load a neural model."""
from pathlib import Path
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(__file__).resolve().parent
VENDOR=PACKAGE/'vendor'
DIAG=ROOT/'diagnostics/permworld_peaks_probe_20261109'
OUT=Path('/mnt/yangx-projects-f/20261109/permworld_peaks_probe')
PRIOR=ROOT/'diagnostics/permworld_repetition_20261106'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def rows(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def freeze():
    p=DIAG/'protocol.json'
    if p.exists():return config()
    audit=read(DIAG/'cache_audit.json');assert audit['status']=='passed'
    assert audit['current_task_physical_checkpoint_passed']==45
    assert read(DIAG/'math_split_audit.json')['status']=='passed'
    old=audit['historical_reproduction'];prior=read(PRIOR/'protocol.json')
    units=[dict(unit='archdev001',data_path=old['data_path'],data_sha256=old['data_sha256'],historical=True)]
    units += [dict(unit=u['unit'],root=str(Path(prior['output'])/u['unit']),
                   data_path=u['data_path'],data_sha256=u['data_sha256'],historical=False) for u in audit['units']]
    c=dict(version='supplied-peaks-valleys-cache-repetition/v1',created_utc=datetime.now(timezone.utc).isoformat(),
        output=str(OUT),units=units,encodings=['canonical_cycle','one_line'],lengths=list(range(10,31)),
        actions=['identity','complement','inverse'],alphas=[1,10,100,1000],
        supplied_archive_sha256=read(DIAG/'imported_archive_manifest.json')['archive_sha256'],
        representation='Saved final_norm state at input end before task token; FP32 stored features',
        protocol='Unchanged vendor analyze.py: one standardized multi-output ridge per length/phase/task, source-only validation tuning, original orbit exclusions and count-matched mask.',
        main_support='Unit-specific common test mask where source peaks and all three transformed valley counts coincide; same anchors for both encodings/actions.',
        aggregation='Every length required; equal1/21 mean of within-sample position AUROC. No pooling or silent NaN length omission.',
        main_contrasts=['C-I','C-inverse','C-max(I,inverse) after the fixed equal-length aggregate','trained C minus separately fitted step0 C'],
        secondary='Per-length directions, all-anchor support, own-valley labels, within-model probes, other-specialist controls, oracle-count position prior, saved count accuracies.',
        historical_reproduction_tolerance=dict(AUROC_absolute=0.0005,justification='Fixed before fits; smaller than the reported historical cycle advantage by more than an order of magnitude, accommodates CPU-forward versus saved-cache and library roundoff.',
            coverage='Exact fit/dev/test/matched counts and score inventory; original verify.py CKA tolerance retained2e-5.'),
        statistical_unit='Four newer independent data/initialization/training bundles; report raw values/mean/sampleSD/range and directions. One task pair only;21lengths/anchors/conditions do not create independent task-pair replicates. No new significance or equivalence test.',
        initial_comparison='Initial and trained phases refit separate probes. Same source probe within phase across target transforms; not a fixed-readout causal intervention across training.',
        scope='Only cached-feature import, ridge-probe fits and scalar analysis. No neural forward, base-model training, additional units, encoding changes, probes on other relations, F17 edits or metric searches.',
        result_policy='Keep all outcomes, zero/mixed effects and every technical failure; stop on historical reproducibility failure without changing scoring/tolerance.',
        user_authorized_upload_after_completion=True,exploratory=True,original_pilot_seen=True,
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write(p,c)
    sources=list(PACKAGE.glob('*.py'))+list(VENDOR.glob('*.py'))+[DIAG/'cache_audit.json',DIAG/'math_split_audit.json',DIAG/'imported_archive_manifest.json']
    write(DIAG/'freeze.json',dict(protocol_sha256=sha(p),source_sha256={str(q.relative_to(ROOT)):sha(q) for q in sources},before_actual_probe_fits=True))
    protected={str(ROOT/q):v['sha256'] for q,v in read(ROOT/'diagnostics/permworld_answer_controls_20261109/delivery_manifest.json')['files'].items()}
    for q in [ROOT/'README.md',ROOT/'CURRENT_RESEARCH.txt',ROOT/'manuscripts/representation_selectivity_20261012/manuscript.tex',ROOT/'manuscripts/representation_selectivity_20261012/manuscript.pdf',ROOT/'manuscripts/representation_selectivity_20261012/manuscript_source.zip']:
        protected[str(q)]=sha(q)
    for q in audit['physical_checkpoint_files']+old['physical_payload_checks']:
        protected[q['path']]=q['expected_sha256']
    for q in audit['cache_records']:protected[q['path']]=q['expected_payload_sha256'];protected[q['metadata_path']]=q['metadata_sha256']
    for u in units:protected[u['data_path']]=u['data_sha256']
    write(DIAG/'protection_start.json',dict(scope='Named existing result/code/manuscript files plus45physical CP,64newfeatureNPZ,metadata and five datafiles. Endpoint audit, not all-files or continuous monitoring.',files=protected,git_head=c['git_head']))
    return c


def config():
    c=read(DIAG/'protocol.json');f=read(DIAG/'freeze.json');assert sha(DIAG/'protocol.json')==f['protocol_sha256']
    for p,d in f['source_sha256'].items():assert sha(ROOT/p)==d,('Frozen probe source changed',p)
    return c


def invoke(script,args,log):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        result=subprocess.run([sys.executable,str(VENDOR/script),*map(str,args)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{script} failed; see {log}')


def historical_import(c):
    h=read(DIAG/'cache_audit.json')['historical_reproduction'];idx=read(h['index_path']);assert sha(h['index_path'])==h['index_sha256']
    table={(r['architecture'],r['encoding'],r['task'],r['condition'],r['split']):r for r in idx['records']}
    u=c['units'][0];dest=OUT/'features'/u['unit'];dest.mkdir(parents=True,exist_ok=True);used={}
    assert sha(u['data_path'])==u['data_sha256']
    with np.load(u['data_path'],allow_pickle=False) as data:sizes={s:len(data[s+'_lengths']) for s in ['validation','test']}
    for enc in c['encodings']:
        for name in ['initial','peaks','valleys','descents','recoils']:
            task,phase=('peaks','initial') if name=='initial' else (name,'trained');a={};cp=set()
            for split,n in sizes.items():
                a[split+'_indices']=np.arange(n)
                if split=='validation' and task in ['descents','recoils']:continue
                r=table['gru',enc,task,phase,split];assert r['source_data_sha256']==u['data_sha256'];cp.add(r['source_checkpoint_sha256'])
                loaded={}
                for field in ['prefix_final','prediction']:
                    desc=r['arrays'][field];p=Path(desc['path']);assert sha(p)==desc['sha256'];used[str(p)]=desc['sha256'];loaded[field]=np.load(p,mmap_mode='r',allow_pickle=False)
                assert loaded['prefix_final'].shape==(n,3,256) and loaded['prediction'].shape==(n,3)
                actions=[0] if split=='validation' or name=='peaks' else [1] if name in ['descents','recoils'] else [0,1,2]
                for action in actions:
                    a[f'{split}_{action}_hidden']=loaded['prefix_final'][:,action]
                    a[f'{split}_{action}_prediction']=loaded['prediction'][:,action]
            assert len(cp)==1;a['checkpoint_sha256']=np.asarray(cp.pop());np.savez(dest/f'{enc}-{name}.npz',**a)
    write(OUT/'historical_cache_read.json',dict(files=used,no_forward=True))
    return dest


def guard_results(path):
    rr=rows(path/'summary.csv');assert len(rr)>0 and all(int(r['finite_lengths'])==21 and np.isfinite(float(r['auc'])) for r in rr)
    cv=rows(path/'coverage.csv');assert len(cv)==21 and all(int(r['matched_n'])>0 and int(r['fit_n'])>0 and int(r['dev_n'])>0 for r in cv)
    expected={'unit02':1707,'unit03':1675,'unit04':1753,'unit05':1755,'archdev001':1757}
    assert sum(int(r['matched_n']) for r in cv)==expected[cv[0]['unit']]
    return rr


def execute(c,u):
    name=u['unit'];dest=OUT/'features'/name;result=DIAG/'results'/name;done=result/'completion.json'
    if done.exists():
        receipt=read(done);assert receipt['protocol_sha256']==sha(DIAG/'protocol.json')
        for p,d in receipt['result_sha256'].items():assert sha(result/p)==d
        guard_results(result);return receipt
    started=time.monotonic();assert sha(u['data_path'])==u['data_sha256']
    if u['historical']:historical_import(c)
    else:invoke('import_repetition.py',['--unit-dir',u['root'],'--features',dest],OUT/'logs'/f'{name}_import.log')
    invoke('analyze.py',['--data',u['data_path'],'--features',dest,'--unit',name,'--out',result],OUT/'logs'/f'{name}_analyze.log')
    guard_results(result)
    if u['historical']:
        reference=rows(DIAG/'supplied_historical/results/summary.csv');actual=rows(result/'summary.csv');keys=['encoding','phase','method','transform','label','support']
        assert rows(result/'coverage.csv')==rows(DIAG/'supplied_historical/results/coverage.csv'), 'Historical split/matched coverage differs'
        table={tuple(r[k] for k in keys):r for r in actual};differences=[]
        for r in reference:
            a=table[tuple(r[k] for k in keys)];assert int(a['n'])==int(r['n']) and int(a['defined_n'])==int(r['defined_n'])
            differences.append(abs(float(a['auc'])-float(r['auc'])))
        assert len(actual)==len(reference) and max(differences)<=c['historical_reproduction_tolerance']['AUROC_absolute'],('Historical AUROC discrepancy',max(differences))
        invoke('verify.py',['--data',u['data_path'],'--features',dest,'--published',DIAG/'supplied_historical/results/released_cka_reference.csv','--learning',result/'learning_by_length.csv','--out',result/'verification.json'],OUT/'logs'/f'{name}_verification.log')
        write(DIAG/'historical_reproduction.json',dict(status='passed',compared_summary_rows=len(reference),max_AUROC_absolute_difference=max(differences),tolerance=c['historical_reproduction_tolerance']['AUROC_absolute'],cached_features_not_new_CPU_forward=True,all21lengths=True))
    receipt=dict(status='completed',unit=name,protocol_sha256=sha(DIAG/'protocol.json'),elapsed_seconds=time.monotonic()-started,
        result_sha256={p.name:sha(p) for p in result.glob('*.csv')},imported_feature_sha256={p.name:sha(p) for p in dest.glob('*.npz')},
        fitted_only_linear_ridge_probes=True,no_neural_forward=True,no_base_training=True)
    write(done,receipt);print(json.dumps(receipt),flush=True);return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','historical','new','all']);args=parser.parse_args()
    if args.stage=='freeze':freeze();return
    c=config()
    try:
        if args.stage in ['historical','all']:execute(c,c['units'][0])
        if args.stage in ['new','all']:
            assert read(DIAG/'historical_reproduction.json')['status']=='passed'
            for u in c['units'][1:]:execute(c,u)
        write(DIAG/'run_completion.json',dict(status='completed',stage=args.stage,no_base_training=True,no_neural_forward=True))
    except Exception as error:
        write(DIAG/'failure.json',dict(status='failed',stage=args.stage,type=type(error).__name__,error=str(error)));raise


if __name__=='__main__':main()
