"""Checkpoint-only full train/validation evaluation; never train or decode test."""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from experiments.permworld_learnability_20261010.common import (
    ROOT as SOURCE_ROOT, CONFIG as SOURCE_CONFIG, DIAG as SOURCE_DIAG,
    TASKS, TOKEN_TO_ID, batch, model, device_setup, parameter_hash, sha,
    json_write, verify as verify_source, source_version,
)

DIAG = Path(__file__).resolve().parent
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_fullset_eval')
PROTOCOL = DIAG / 'protocol.json'
METRICS = ('answer_accuracy', 'answer_ce', 'eos_accuracy', 'eos_ce', 'combined_ce')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def setup():
    assert not PROTOCOL.exists(), 'Do not overwrite an existing frozen evaluation protocol'
    source = verify_source()
    assert len(source['training_manifest']) == 16
    source_data = SOURCE_ROOT / 'worlds/learningdev001/data.npz'
    source_data_digest = sha(source_data)
    DIAG.mkdir(parents=True, exist_ok=True); ROOT.mkdir(parents=True, exist_ok=True)
    protocol = dict(protocol_version='permworld-full-train-validation-eval/v1',
        registered_utc=now(), purpose='Exploratory failure diagnosis using fixed existing8k checkpoints.',
        source_protocol_path=str(SOURCE_CONFIG), source_protocol_sha256=sha(SOURCE_CONFIG),
        source_version_sha256=source_version(),
        source_data_path=str(SOURCE_ROOT / 'worlds/learningdev001/data.npz'),
        training_manifest=source['training_manifest'], lengths=[10,11,12],
        length_weights={'10':1/3,'11':1/3,'12':1/3},
        samples_per_length={'train':2048,'validation':1024},
        decoded_arrays=[f'{s}_{name}' for s in ('train','validation') for name in
                        ('lengths','labels','one_line_input','canonical_cycle_input')],
        evaluation=dict(mode='eval plus torch.inference_mode', model_mode='eval',
            autocast=False, parameter_dtype='float32', logits_dtype='float32', TF32=False,
            precision='FP32 parameters and forward, no autocast, TF32 disabled',
            batch_size=128, sample_order='ascending source index within each length',
            actions=['identity'], answer_position='2n+5; before the true answer input',
            EOS_position='2n+6; teacher-forced on the true answer', vocabulary_size=188,
            answer_accuracy='argmax over all188 vocabulary logits, not numeric-conditional argmax',
            CE='cross-entropy over all188 logits', combined_ce='(answer_ce+eos_ce)/2',
            optimizer_constructed=False, training_forward=False, backward_called=False),
        eos_token_id=TOKEN_TO_ID['<EOS>'],
        label_task_names=list(TASKS),
        numeric_answer_token_ids={str(y):TOKEN_TO_ID[f'{y:02d}'] for y in range(31)},
        validation_reproduction_tolerances=dict(accuracy_absolute=0.,ce_absolute=1e-5,ce_relative=1e-6),
        independent_logit_ce_tolerance=dict(absolute=2e-6,relative=2e-6),
        diagnostic_learning_thresholds=dict(mean_answer_accuracy_min=.90,per_length_answer_accuracy_min=.85),
        diagnostic_threshold_interpretation='Existing learning criteria reused descriptively, not TOST or a causal conclusion.',
        expected_model_sample_pairs=147456, expected_length_cells=96,
        scope=dict(new_training=False, source_files_changed=False, test_arrays_decoded=False,
            new_CKA=False, GroupWorld=False, F17_changes=False, auto_continuation=False),
        output=str(ROOT), completion='Evaluate all16, audit, report and stop. Do not start development training.')
    protected = {str(Path(path) if Path(path).is_absolute() else WORKSPACE/ path):digest
        for path,digest in json.loads((SOURCE_DIAG / 'protected_legacy_before.json').read_text()).items()}
    old_manifest = json.loads((SOURCE_DIAG / 'artifact_manifest.json').read_text())
    for relative, digest in old_manifest['files_sha256'].items():
        path = SOURCE_DIAG / relative
        assert sha(path) == digest, ('Prior report artifact differs', path)
        protected[str(path)] = digest
    protected[str(SOURCE_DIAG / 'artifact_manifest.json')] = sha(SOURCE_DIAG / 'artifact_manifest.json')
    source_freeze = json.loads((SOURCE_DIAG / 'freeze.json').read_text())
    for relative, digest in source_freeze['source_sha256'].items():
        protected[str(WORKSPACE / relative)] = digest
    jobs = []
    for job in source['training_manifest']:
        directory = SOURCE_ROOT / 'models' / job['run_id']
        stage_path, completed_path = directory / 'stage_8000.json', directory / 'completed.json'
        stage = json.loads(stage_path.read_text()); completed = json.loads(completed_path.read_text())
        assert stage['status']=='stage_complete' and stage['step']==8000 and stage['fixed_endpoint_completed']
        assert completed['status']=='completed' and completed['step']==8000
        assert {k:v for k,v in stage.items() if k!='status'}=={k:v for k,v in completed.items() if k!='status'}
        for key in ('run_id','architecture','encoding','task'):
            assert stage[key]==job[key]
        assert stage['protocol_sha256']==sha(SOURCE_CONFIG)
        assert stage['execution_source_version_sha256']==source_version()
        assert stage['data_sha256']==source_data_digest
        assert stage['fixed_endpoint_step']==8000
        assert stage['output_directory']==str(directory)
        checkpoint = Path(stage['checkpoint_path'])
        assert checkpoint.parent==directory and checkpoint.name=='checkpoint_step_8000.pt'
        assert sha(checkpoint)==stage['checkpoint_sha256']
        initdir = SOURCE_ROOT / 'initializations' / job['architecture']
        initial_pt, initial_json = initdir/'step0.pt', initdir/'step0.json'
        assert initial_pt.is_file() and initial_json.is_file()
        meta=json.loads(initial_json.read_text())
        assert sha(initial_pt)==meta['checkpoint_sha256']==stage['initial_checkpoint_sha256']
        assert meta['parameter_sha256']==stage['initial_parameter_sha256']
        for path in (stage_path,completed_path,checkpoint,initial_pt,initial_json):
            protected[str(path)]=sha(path)
        jobs.append(dict(**job,checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                         stage_path=str(stage_path),stage_sha256=sha(stage_path)))
    data_audit=SOURCE_ROOT/'worlds/learningdev001/data_audit.json'
    data_meta=json.loads(data_audit.read_text())
    assert sha(protocol['source_data_path'])==data_meta['data_sha256']
    for path in (Path(protocol['source_data_path']),data_audit,SOURCE_CONFIG):protected[str(path)]=sha(path)
    protocol['source_data_sha256']=data_meta['data_sha256']
    protocol['checkpoint_manifest']=jobs
    json_write(PROTOCOL,protocol)
    json_write(DIAG/'protected_before.json',protected)
    diagnostic_sources=[Path(__file__).resolve(),DIAG/'report.py',DIAG/'independent_audit.py']
    assert all(path.is_file() for path in diagnostic_sources)
    json_write(DIAG/'freeze.json',dict(registered_utc=now(),protocol_sha256=sha(PROTOCOL),
        protected_manifest_sha256=sha(DIAG/'protected_before.json'),
        source_sha256={**{str(p):sha(p) for p in diagnostic_sources},
                       **{str(WORKSPACE/k):v for k,v in source_freeze['source_sha256'].items()}},
        evaluated_new_train_logits_before_freeze=False, training_forbidden=True))
    print({'setup':'passed','existing_checkpoints':len(jobs),'protected_files':len(protected)},flush=True)


def verify():
    protocol=json.loads(PROTOCOL.read_text()); frozen=json.loads((DIAG/'freeze.json').read_text())
    assert sha(PROTOCOL)==frozen['protocol_sha256']
    assert sha(DIAG/'protected_before.json')==frozen['protected_manifest_sha256']
    for path,digest in frozen['source_sha256'].items():assert sha(path)==digest,('Frozen code changed',path)
    source=verify_source()
    assert sha(SOURCE_CONFIG)==protocol['source_protocol_sha256']
    assert source_version()==protocol['source_version_sha256']
    return protocol,source


def summarize_arrays(arrays,prefix,protocol):
    lengths=arrays[prefix+'_lengths']
    answer_correct=arrays[prefix+'_predicted_answer_token']==arrays[prefix+'_gold_answer_token']
    eos_correct=arrays[prefix+'_eos_predicted_token']==protocol['eos_token_id']
    records=[]
    for n in protocol['lengths']:
        chosen=lengths==n
        count=int(chosen.sum()); assert count==protocol['samples_per_length'][prefix]
        answer_ce=math.fsum(map(float,arrays[prefix+'_answer_ce'][chosen]))/count
        eos_ce=math.fsum(map(float,arrays[prefix+'_eos_ce'][chosen]))/count
        records.append(dict(length=n,samples=count,
            answer_accuracy=int(answer_correct[chosen].sum())/count,answer_ce=answer_ce,
            eos_accuracy=int(eos_correct[chosen].sum())/count,eos_ce=eos_ce,
            combined_ce=(answer_ce+eos_ce)/2))
    summary=dict(samples=len(lengths),**{key:math.fsum(row[key] for row in records)/3 for key in METRICS})
    return records,summary


@torch.inference_mode()
def evaluate_split(net,data,job,split,device,protocol):
    assert split in ('train','validation')
    assert not net.training and all(not module.training for module in net.modules())
    assert not torch.is_grad_enabled()
    lengths=data[split+'_lengths']; count=len(lengths)
    arrays={split+'_lengths':lengths.copy(),split+'_sample_index':np.arange(count,dtype=np.int32)}
    for name in ('gold_answer_token','predicted_answer_token','eos_predicted_token'):
        arrays[split+'_'+name]=np.empty(count,dtype=np.int16)
    for name in ('answer_ce','eos_ce'):arrays[split+'_'+name]=np.empty(count,dtype=np.float64)
    for name in ('answer_logits','eos_logits'):
        arrays[split+'_'+name]=np.empty((count,188),dtype=np.float32)
    visited=np.zeros(count,dtype=np.bool_)
    for n in protocol['lengths']:
        indices=np.flatnonzero(lengths==n)
        assert len(indices)==protocol['samples_per_length'][split]
        for start in range(0,len(indices),128):
            use=indices[start:start+128]
            y=data[split+'_labels'][use,0,TASKS.index(job['task'])].astype('int64')
            ids,mask,_,_,equals=batch(data[f'{split}_{job["encoding"]}_input'][use,0],lengths[use],job['task'],y,device)
            assert torch.all(equals==2*n+5)
            rows=torch.arange(len(use),device=device)
            truth=torch.tensor([TOKEN_TO_ID[f'{int(v):02d}'] for v in y],device=device)
            with torch.autocast(device.type,enabled=False):logits=net(ids,mask)
            assert logits.dtype==torch.float32 and not net.training
            answers=logits[rows,equals]; eos=logits[rows,equals+1]
            assert torch.isfinite(answers).all() and torch.isfinite(eos).all()
            eos_truth=torch.full_like(truth,protocol['eos_token_id'])
            arrays[split+'_gold_answer_token'][use]=truth.cpu().numpy()
            arrays[split+'_predicted_answer_token'][use]=answers.argmax(-1).cpu().numpy()
            arrays[split+'_eos_predicted_token'][use]=eos.argmax(-1).cpu().numpy()
            arrays[split+'_answer_ce'][use]=F.cross_entropy(answers,truth,reduction='none').cpu().double().numpy()
            arrays[split+'_eos_ce'][use]=F.cross_entropy(eos,eos_truth,reduction='none').cpu().double().numpy()
            arrays[split+'_answer_logits'][use]=answers.cpu().numpy()
            arrays[split+'_eos_logits'][use]=eos.cpu().numpy()
            assert not visited[use].any();visited[use]=True
    assert visited.all() and all(p.grad is None for p in net.parameters())
    return arrays


def reproduce(job,stage,records,summary,protocol):
    original=stage['validation']; old_by_n={r['length']:r for r in original['per_length']}
    rows=[]
    for key,new in [(r['length'],r) for r in records]+[('equal_length',summary)]:
        old=old_by_n[key] if isinstance(key,int) else original
        for metric in METRICS:
            old_value=float(old[metric]); new_value=float(new[metric]); difference=new_value-old_value
            rules=protocol['validation_reproduction_tolerances']
            tolerance=rules['accuracy_absolute'] if 'accuracy' in metric else rules['ce_absolute']+rules['ce_relative']*abs(old_value)
            rows.append(dict(run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],
                task=job['task'],length=key,metric=metric,original=old_value,recomputed=new_value,
                signed_difference=difference,tolerance=tolerance,passed=abs(difference)<=tolerance,
                reference_path=job['stage_path']))
    return rows


def evaluate():
    protocol,source=verify()
    assert not (DIAG/'evaluation_execution.json').exists(), 'Evaluation already started; inspect its recorded state, do not silently rerun'
    assert sha(protocol['source_data_path'])==protocol['source_data_sha256']
    with np.load(protocol['source_data_path'],allow_pickle=False) as archive:
        data={name:archive[name].copy() for name in protocol['decoded_arrays']}
    assert not any(name.startswith('test_') for name in data)
    for split in ('train','validation'):
        assert sorted(map(int,np.unique(data[split+'_lengths'])))==protocol['lengths']
        assert data[split+'_labels'].shape[1]==(1 if split=='train' else 3)
    device=device_setup(); start=time.time()
    length_rows,summary_rows,reproduction,manifest=[],[],[],[]
    for job in protocol['checkpoint_manifest']:
        checkpoint=Path(job['checkpoint_path']); assert sha(checkpoint)==job['checkpoint_sha256']
        cp=torch.load(checkpoint,map_location='cpu',weights_only=False)
        stage=json.loads(Path(job['stage_path']).read_text()); assert sha(job['stage_path'])==job['stage_sha256']
        for key in ('run_id','architecture','step','protocol_sha256','execution_source_version_sha256',
                    'data_sha256','initial_parameter_sha256','initial_checkpoint_sha256','starting_rng_hash'):
            assert cp[key]==stage[key],('Checkpoint binding mismatch',key)
        assert cp['step']==8000 and cp['source_model_run']
        assert cp['data_sha256']==protocol['source_data_sha256']
        assert cp['fixed_endpoint_step']==8000
        assert cp['output_directory']==str(checkpoint.parent)
        assert cp['finite_gradient_updates']==8000
        net=model(source,job['architecture'],job['model_seed'],device)
        net.load_state_dict(cp['model'],strict=True); net.eval()
        assert all(p.dtype==torch.float32 for p in net.parameters())
        before=parameter_hash(net); arrays={}; outputs={}
        for split in ('train','validation'):
            split_arrays=evaluate_split(net,data,job,split,device,protocol)
            records,summary=summarize_arrays(split_arrays,split,protocol)
            arrays.update(split_arrays);outputs[split]=(records,summary)
            common={key:job[key] for key in ('run_id','architecture','encoding','task')}
            length_rows.extend(dict(**common,split=split,**row) for row in records)
            summary_rows.append(dict(**common,split=split,**summary))
        assert parameter_hash(net)==before and all(p.grad is None for p in net.parameters())
        out=ROOT/job['run_id'];out.mkdir(parents=True,exist_ok=True)
        prediction_path=out/'predictions.npz'
        with prediction_path.open('wb') as stream:np.savez_compressed(stream,**arrays)
        manifest.append(dict(run_id=job['run_id'],path=str(prediction_path),sha256=sha(prediction_path),
                             checkpoint_path=str(checkpoint),checkpoint_sha256=job['checkpoint_sha256'],
                             parameter_sha256_before=before,parameter_sha256_after=before))
        reproduction.extend(reproduce(job,stage,*outputs['validation'],protocol))
        json_write(DIAG/'prediction_manifest.json',manifest)
        json_write(DIAG/'evaluation_execution.json',dict(status='running',started_epoch=start,
            completed_models=len(manifest),elapsed_seconds=time.time()-start,device=str(device),
            optimizer_constructed=False,backward_called=False,parameters_changed=False,
            decoded_arrays=list(data),test_arrays_decoded=False,geometry_evaluated=False,new_training=False))
        print(dict(run_id=job['run_id'],train_accuracy=outputs['train'][1]['answer_accuracy'],
                   validation_accuracy=outputs['validation'][1]['answer_accuracy'],
                   train_answer_ce=outputs['train'][1]['answer_ce'],validation_answer_ce=outputs['validation'][1]['answer_ce']),flush=True)
        del net,cp,arrays,split_arrays
    fields=['run_id','architecture','encoding','task','split','samples',*METRICS]
    write_csv(DIAG/'metrics_summary.csv',summary_rows,fields)
    write_csv(DIAG/'metrics_by_length.csv',length_rows,fields[:5]+['length']+fields[5:])
    write_csv(DIAG/'validation_reproduction.csv',reproduction,
        ['run_id','architecture','encoding','task','length','metric','original','recomputed','signed_difference','tolerance','passed','reference_path'])
    assert len(length_rows)==96 and len(summary_rows)==32 and len(reproduction)==320
    assert sum(row['samples'] for row in summary_rows)==147456
    errors=[row for row in reproduction if not row['passed']]
    json_write(DIAG/'evaluation_execution.json',dict(status='completed' if not errors else 'completed_with_reproduction_difference',
        started_epoch=start,finished_utc=now(),completed_models=len(manifest),elapsed_seconds=time.time()-start,
        device=str(device),length_cells=len(length_rows),sample_model_pairs=147456,
        validation_reproduction_passed=not errors,validation_reproduction_failed_rows=errors,
        optimizer_constructed=False,backward_called=False,parameters_changed=False,
        decoded_arrays=list(data),test_arrays_decoded=False,geometry_evaluated=False,new_training=False))
    print({'models':len(manifest),'validation_reproduction_passed':not errors,'elapsed_seconds':time.time()-start},flush=True)


def audit():
    protocol,_=verify()
    protected=json.loads((DIAG/'protected_before.json').read_text())
    differences=[name for name,digest in protected.items() if not Path(name).is_file() or sha(name)!=digest]
    assert not differences,('Existing protected artifacts changed',differences)
    execution=json.loads((DIAG/'evaluation_execution.json').read_text())
    assert execution['completed_models']==16 and not execution['new_training']
    assert execution['sample_model_pairs']==147456
    manifests=json.loads((DIAG/'prediction_manifest.json').read_text())
    assert len(manifests)==16
    for row in manifests:
        assert sha(row['path'])==row['sha256']
        assert row['parameter_sha256_before']==row['parameter_sha256_after']
    active=[]
    for pid in Path('/proc').iterdir():
        if not pid.name.isdigit():continue
        try:args=pid.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if '-m' in args and args[args.index('-m')+1]=='experiments.permworld_learnability_20261010' and '--stage' in args:
            stage=args[args.index('--stage')+1]
            if stage in ('develop','train-one','resume'):active.append(dict(pid=int(pid.name),stage=stage))
    assert not active
    json_write(DIAG/'completion_audit.json',dict(status='passed',generated_utc=now(),
        protected_old_files=len(protected),protected_hash_differences=differences,
        checkpoints_reused=16,complete_train_and_validation_evaluated=True,source_weights_unchanged=True,
        sample_model_pairs=147456,length_cells=96,validation_reproduction_passed=execution['validation_reproduction_passed'],
        source_protocol_sha256=protocol['source_protocol_sha256'],evaluation_protocol_sha256=sha(PROTOCOL),
        train_workers=active,new_training=False,heldout_test_used=False,new_CKA=False,GroupWorld=False,F17_changed=False,
        complete_and_stopped=True))
    print({'audit':'passed','protected_old_files':len(protected),'all_weights_unchanged':True,'no_training':True},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['setup','evaluate','audit'],required=True)
    args=parser.parse_args()
    {'setup':setup,'evaluate':evaluate,'audit':audit}[args.stage]()
