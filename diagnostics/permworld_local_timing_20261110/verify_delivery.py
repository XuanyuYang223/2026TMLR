"""Final delivery inventory; never fit, run a model, or rescore outcomes.

Reproduce with:
  .venv/bin/python diagnostics/permworld_local_timing_20261110/verify_delivery.py

Frozen protocol/source and audit receipts are verified. Only NEW production
CSV/NPZ payloads are freshly hashed; prior named-file protection is bound to
the completed endpoint audit rather than repeated. Later README/reproduction
and delivery inventory files are deliberately outside this helper's scope.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
DIAG=Path(__file__).resolve().parent
UNITS=('archdev001','unit02','unit03','unit04','unit05')
FORMAL_UNITS=UNITS[1:]


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


class Checks:
    def __init__(self):self.total=0;self.failures=[]
    def ok(self,test,label):
        self.total+=1
        if not bool(test):self.failures.append(label)


def csv_inventory(path):
    """Header and physical record-line inventory, without interpreting values.

    Generated fields do not contain embedded record newlines. Counts are also
    checked against the original streaming-completion counters. We do not
    present this as a new independent numerical/semantic CSV audit.
    """
    path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',newline='') as stream:header=next(csv.reader(stream))
    lines=0;last=b''
    with opener(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            lines+=chunk.count(b'\n');last=chunk[-1:]
    if last and last!=b'\n':lines+=1
    return dict(header=header,columns=len(header),data_rows=max(0,lines-1),
        count_method='Generated CSV physical record lines plus parsed header; checked against original streaming counters, not outcome rescoring')


def frozen_bindings(checks):
    protocol=read(DIAG/'protocol.json');frozen=read(DIAG/'freeze.json');digest=sha(DIAG/'protocol.json')
    checks.ok(digest==frozen['protocol_sha256'],'exact frozen production protocol')
    source_records=[]
    for relative,expected in frozen['sources'].items():
        path=ROOT/relative;actual=sha(path);checks.ok(actual==expected,'frozen production source '+relative)
        source_records.append(dict(path=str(path),sha256=actual,expected_sha256=expected,passed=actual==expected))
    checks.ok(frozen['before_new_formal_outcomes'],'freeze precedes new formal timing outcomes')
    summary=read(DIAG/'summary/completion.json');headline=read(DIAG/'headlines/completion.json')
    sf=read(DIAG/'summary_freeze.json');hf=read(DIAG/'headline_freeze.json')
    checks.ok(summary['status']=='completed_saved_timing_summary','summary completed')
    checks.ok(headline['status']=='completed_derived_timing_headlines','headlines completed')
    checks.ok(summary['protocol_sha256']==headline['protocol_sha256']==digest==sf['protocol_sha256'],'all derived protocol identities')
    checks.ok(sha(DIAG/'summarize_results.py')==summary['source_sha256']==sf['source_sha256'],'frozen summary source')
    checks.ok(sha(DIAG/'build_headline_tables.py')==headline['source_sha256']==hf['source_sha256'],'frozen headline source')
    checks.ok(sf['synthetic_verification']['status']=='passed' and hf['synthetic_verification']['status']=='passed','derived synthetic verifications')
    for label,receipt in [('summary',summary),('headline',headline)]:
        checks.ok(receipt['formal_units']==list(FORMAL_UNITS),label+' four formal units')
    checks.ok(summary['historical_engineering_reference']==headline['historical_reference']=='archdev001','historical reference kept separate')
    checks.ok(headline['main_cells']==headline['auxiliary_cells']==headline['positive_control_cells']==8,'eight registered headline cells')
    checks.ok(headline['maximum_summary_error']==0,'headline arithmetic receipt')
    for record in summary['source_tables']:
        checks.ok(record['unit'] in UNITS,'summary source unit inventory')
    for path,expected in headline['source_tables'].items():checks.ok(sha(path)==expected,'headline source-table SHA '+path)
    audit=read(DIAG/'saved_results_audit.json');protection=read(DIAG/'protection_end.json')
    checks.ok(audit['status']=='passed' and not audit['failures'],'independent saved-results audit passed')
    for path,expected in audit['provenance_snapshot'].items():checks.ok(sha(path)==expected,'independent audit dependency '+path)
    checks.ok(protection['status']=='passed' and not protection['failed_files'],'completed prior-input protection audit passed')
    checks.ok(protection['git_head_unchanged'] and not protection['end_tracked_status'],'prior end HEAD/tracked status')
    checks.ok(protection['unused40GB_token_payloads_read']==0 and protection['unused_fit_dev_payloads_rehashed']==0,'prior protection avoids unused state payloads')
    return protocol,digest,source_records,summary,headline,audit,protection


def seed_manifest(inputs,checks):
    repetition_path=ROOT/'diagnostics/permworld_repetition_20261106/protocol.json'
    registered=read(repetition_path);by_unit={u['unit']:u for u in registered['units']}
    records=[];sources={str(repetition_path):sha(repetition_path)}
    for unit in inputs['units']:
        for task,model in unit['models'].items():
            completion=Path(model['completion_path']);metadata=read(completion);sources[str(completion)]=sha(completion)
            checks.ok(metadata['checkpoint_sha256']==model['checkpoint_sha256'],'source checkpoint receipt '+unit['unit']+'/'+task)
            checks.ok(metadata['data_sha256']==unit['data_sha256'],'source data receipt '+unit['unit']+'/'+task)
            checks.ok(metadata['initial_parameter_sha256']==model['initial_parameter_sha256'],'actual initial state receipt '+unit['unit']+'/'+task)
            setting=by_unit.get(unit['unit'],{})
            init_seed=metadata.get('model_seed',setting.get('architecture_model_seeds',{}).get('gru',''))
            data_seed=metadata.get('data_seed',setting.get('data_seed',''))
            batch_seed=metadata.get('batch_seed',setting.get('batch_seed',''))
            records.append(dict(unit=unit['unit'],historical_reference=unit['historical'],task=task,
                architecture='gru',encoding='one_line',checkpoint_path=model['checkpoint_path'],
                checkpoint_sha256=model['checkpoint_sha256'],parameter_sha256=model['parameter_sha256'],
                data_path=unit['data_path'],data_sha256=unit['data_sha256'],
                initial_parameter_sha256=model['initial_parameter_sha256'],
                initial_checkpoint_sha256=model['initial_checkpoint_sha256'],
                actual_loaded_initial_state_dict_sha256=metadata.get('actual_loaded_initial_state_dict_sha256',''),
                actual_initial_identity_verified=metadata.get('actual_initial_identity_verified',''),
                initialization_seed_as_recorded=init_seed,data_seed_as_recorded=data_seed,
                batch_seed_as_recorded=batch_seed,training_seed_as_recorded=metadata.get('training_seed',''),
                dropout_seed_base_as_registered=setting.get('dropout_seed_base',''),
                evaluation_rng_seed_as_registered=setting.get('evaluation_rng_seed',''),
                evaluation_seed_note=setting.get('evaluation_anchor_source','Historical completion records data seed; no separate evaluation seed is inferred'),
                timing_sampling_seed='No new sampling or training; saved pairs reused',
                random_direction_seeds='1026,1027,1028,1029,1030,1031,1032,1033; saved arrays reused',
                seed_identity_note='Seed values do not establish weight identity; actual initial-weight hashes are retained',
                source_training_receipt=str(completion),source_training_receipt_sha256=sources[str(completion)]))
    path=DIAG/'checkpoint_seed_manifest.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return records,sources


def run(workers=2):
    started=time.perf_counter();began=datetime.now(timezone.utc).isoformat();checks=Checks()
    head_start=git('rev-parse','HEAD');tracked_start=git('status','--porcelain','--untracked-files=no')
    protocol,digest,source_records,summary,headline,audit,protection=frozen_bindings(checks)
    checks.ok(head_start==protocol['git_head']==protection['git_head_end'],'HEAD unchanged across protocol/prior protection/current start')
    checks.ok(not tracked_start,'tracked working tree clean at delivery start')
    inputs=read(DIAG/'input_audit.json');checks.ok(inputs['status']=='passed','input audit passed')
    checks.ok(sha(DIAG/'input_audit.json')==protocol['input_audit_sha256'],'exact input audit binding')
    old_models={u['unit']:u for u in inputs['units']}
    run_receipt=read(DIAG/'run_completion.json')
    checks.ok(run_receipt['status']=='completed' and run_receipt['units']==list(UNITS),'all five production units complete')
    production={};unit_rows=[];maxima=defaultdict(float);total_rows=0;total_cases=0
    for name in UNITS:
        path=DIAG/'results'/name/'completion.json';receipt=read(path)
        checks.ok(receipt['status']=='completed' and receipt['rows']==584640,'production rows '+name)
        checks.ok(receipt['protocol_sha256']==digest,'production protocol '+name)
        checks.ok(receipt['historical_reference']==(name=='archdev001'),'production historical role '+name)
        checks.ok(receipt['no_probe_fitting'] and receipt['no_new_data'] and receipt['no_base_training'],'production scope '+name)
        checks.ok(len(receipt['files'])==1,'one raw CSV per unit '+name)
        for p,expected in receipt['files'].items():production[str(Path(p).resolve())]=dict(expected=expected,kind='production_csv',unit=name)
        cases=0;task_info=[]
        for task in receipt['task_reports']:
            t=task['task'];model=old_models[name]['models'][t]
            checks.ok(task['rows']==292320,'task row inventory '+name+'/'+t)
            checks.ok(task['parameter_sha256_before']==task['parameter_sha256_after']==model['parameter_sha256'],'unchanged model parameters '+name+'/'+t)
            checks.ok(task['early_noop_argmax_mismatches']==task['rerun_late_old_argmax_mismatches']==0,'zero replay modal mismatches '+name+'/'+t)
            checks.ok(task['identical_layer1_bytes_all_common_cases'],'paired h1 byte identity '+name+'/'+t)
            checks.ok(task['no_probe_fitting'],'no refit '+name+'/'+t)
            available={(r['n'],r['i']) for r in task['cells']}
            checks.ok(available=={(n,i) for n in range(10,31) for i in range(2,n-1)},'all fixed cells '+name+'/'+t)
            for cell in task['cells']:
                checks.ok(cell['cases']==(16 if cell['i']<6 else 32),'case inventory '+name+'/'+t)
                production[str(Path(cell['path']).resolve())]=dict(expected=cell['sha256'],kind='production_npz',unit=name,task=t)
            task_cases=sum(c['cases'] for c in task['cells']);cases+=task_cases
            checks.ok(task_cases*29==task['rows'],'29 conditions per pair '+name+'/'+t)
            for key,value in task['maxima'].items():maxima[key]=max(maxima[key],float(value))
            checks.ok(task['maxima']['paired_h1']==task['maxima']['paired_checker']==0,'zero paired h1/checker difference '+name+'/'+t)
            checks.ok(task['maxima']['noedit_state']==task['maxima']['noedit_logits']==0,'exact no-edit replay '+name+'/'+t)
            for key in ['early_noedit_native_logits','late_old_logits','fullswap_native_logits']:
                checks.ok(task['maxima'][key]<=protocol['numerical']['logits_noop_old_late_and_donor_absolute'],'fixed logit tolerance '+name+'/'+t+'/'+key)
            task_info.append(dict(task=t,rows=task['rows'],cases=task_cases,cells=len(task['cells']),
                parameter_sha256=task['parameter_sha256_after'],early_noop_argmax_mismatches=0,late_old_argmax_mismatches=0,
                maxima=task['maxima'],same_h1_bytes=True))
        total_rows+=receipt['rows'];total_cases+=cases
        unit_rows.append(dict(unit=name,historical_reference=name=='archdev001',rows=receipt['rows'],cases=cases,
            elapsed_seconds=receipt['elapsed_seconds'],receipt_path=str(path),receipt_sha256=sha(path),tasks=task_info))
    checks.ok(total_rows==summary['raw_rows']==2923200,'all raw rows match summary receipt')
    checks.ok(total_cases==summary['pairs']==100800,'all pair counts match summary receipt')
    checks.ok(len(production)==3575,'five CSV plus3570 NPZ inventory')
    print(f'DELIVERY HASH {len(production)} NEW production files',flush=True)
    paths=sorted(production)
    with ThreadPoolExecutor(max_workers=workers) as pool:digests=dict(zip(paths,pool.map(sha,paths)))
    fresh=[]
    for p,descriptor in production.items():
        passed=digests[p]==descriptor['expected'];checks.ok(passed,'new production SHA '+p)
        fresh.append(dict(path=p,sha256=digests[p],size_bytes=Path(p).stat().st_size,passed=passed,
                          kind=descriptor['kind'],unit=descriptor['unit']))
    print('DELIVERY HASH COMPLETE; CSV inventory',flush=True)
    csvs={}
    for p,descriptor in production.items():
        if descriptor['kind']=='production_csv':
            record=csv_inventory(p);csvs[p]=record
            checks.ok(record['data_rows']==584640,'CSV header/row count '+descriptor['unit'])
            checks.ok({'timing','condition','first_layer_state_sha256','source_cell_sha256'}<=set(record['header']),'production CSV schema '+descriptor['unit'])
    for record in summary['source_tables']:
        checks.ok(digests[str(Path(record['path']).resolve())]==record['sha256'],'summary source raw CSV '+record['unit'])
    derived=[]
    for folder in [DIAG/'summary',DIAG/'headlines']:
        for path in sorted(folder.iterdir()):
            if not path.is_file():continue
            if path.name.endswith(('.csv','.csv.gz')):
                csvs[str(path)]=csv_inventory(path)
            derived.append(dict(path=str(path),sha256=sha(path),size_bytes=path.stat().st_size,
                                 scope='Current derived output endpoint; not a later delivery-document inventory'))
    checks.ok(csvs[str(DIAG/'summary/scored_pairs.csv.gz')]['data_rows']==summary['scored_rows']==3729600,'scored summary rows')
    checks.ok(csvs[str(DIAG/'summary/by_position.csv.gz')]['data_rows']==summary['position_rows']==1802640,'position summary rows')
    for name in ['main','auxiliary','positive_controls']:
        checks.ok(csvs[str(DIAG/'headlines'/f'{name}_eight_cells.csv')]['data_rows']==8,'eight-cell table '+name)
        checks.ok(csvs[str(DIAG/'headlines'/f'{name}_by_unit.csv')]['data_rows']==32,'four-unit raw headline table '+name)
        checks.ok(csvs[str(DIAG/'headlines'/f'{name}_historical_reference.csv')]['data_rows']==8,'historical headline table '+name)
    figure=read(DIAG/'headlines/timing_figure_receipt.json')
    checks.ok(figure['status']=='completed' and figure['no_fits'] and figure['no_forwards'],'figure derived-only receipt')
    checks.ok(sha(DIAG/'plot_headlines.py')==figure['source_code_sha256'],'figure source binding')
    for filename,expected in figure['source_tables'].items():checks.ok(sha(DIAG/'headlines'/filename)==expected,'figure table source '+filename)
    seeds,seed_sources=seed_manifest(inputs,checks)
    head_end=git('rev-parse','HEAD');tracked_end=git('status','--porcelain','--untracked-files=no')
    checks.ok(head_start==head_end and not tracked_end,'delivery end HEAD unchanged/tracked clean')
    status='passed' if not checks.failures else 'failed'
    report=dict(status=status,started_utc=began,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Frozen protocol/source and original completion receipts; fresh hashes of all new production CSVs/cellNPZs; metadata/header/row inventory; independent/protection receipts bound by SHA. No scientific effects interpreted or recalculated.',
        protocol_sha256=digest,checks=checks.total,failures=checks.failures,frozen_sources=source_records,
        units=unit_rows,unit_count=5,formal_units=list(FORMAL_UNITS),historical_reference='archdev001',
        total_raw_rows=total_rows,total_pairs=total_cases,rows_per_unit=584640,
        summary_counts={k:summary[k] for k in ['raw_rows','pairs','scored_rows','position_rows','exact_l1_hash_comparisons']},
        summary_source_tables=summary['source_tables'],headline_source_tables=headline['source_tables'],
        maxima=dict(maxima),model_parameter_hashes_equal=True,all_noop_and_late_argmax_mismatches=0,
        new_production_files=sorted(fresh,key=lambda x:x['path']),new_production_file_count=len(fresh),
        new_production_bytes_hashed=sum(x['size_bytes'] for x in fresh),
        csv_inventory=csvs,derived_files=derived,
        receipt_bindings={str(DIAG/n):sha(DIAG/n) for n in ['freeze.json','summary_freeze.json','headline_freeze.json','summary/completion.json','headlines/completion.json','saved_results_audit.json','protection_end.json','run_completion.json']},
        independent_audit_checks=audit['checks'],independent_audit_maxima=audit['maximum_differences_by_category'],
        old_protection_scope=protection['scope'],old_protected_inputs_rehashed_here=0,unused40GB_token_payloads_read=0,
        seed_manifest=dict(path=str(DIAG/'checkpoint_seed_manifest.csv'),sha256=sha(DIAG/'checkpoint_seed_manifest.csv'),rows=len(seeds),source_metadata_sha256=seed_sources),
        git_head_start=head_start,git_head_end=head_end,git_head_unchanged=head_start==head_end,
        tracked_status_start=tracked_start,tracked_status_end=tracked_end,tracked_clean=True,
        excluded=['Later README/reproduce/delivery_manifest/completion files','Global or continuous old-file verification','New outcome interpretation or new tests/forwards/fitting'],
        verifier_source_sha256=sha(__file__),elapsed_seconds=time.perf_counter()-started,
        reproduce_command='.venv/bin/python diagnostics/permworld_local_timing_20261110/verify_delivery.py',
        no_fitting=True,no_neural_forward=True,no_new_conditions=True,no_old_scientific_writes=True)
    write(DIAG/'delivery_audit.json',report)
    print(dict(status=status,checks=checks.total,failures=checks.failures[:8],rows=total_rows,
               production_files=len(fresh),production_bytes=report['new_production_bytes_hashed'],seconds=time.perf_counter()-started),flush=True)
    if checks.failures:raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=2)
    run(parser.parse_args().workers)
