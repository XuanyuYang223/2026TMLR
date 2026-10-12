"""One fixed overfit/resume/cost gate; explicit launch only, never formal models."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import TASKS, batch
from .model_runtime import (NEW_ROOT, TOKEN_TO_ID, atomic_checkpoint, deadline_guard, device_setup,
    json_write, load_saved_model, parameter_hash, payload_sha, rng_hash, sha)
from .training import run_job, schedule_ids


def array_hash(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(a.tobytes()); return h.hexdigest()


def engineering_data(raw):
    if 'train_input' in raw:
        data = {k:v for k,v in raw.items() if k.startswith(('train_', 'validation_'))}
    else:
        data = {split+'_'+key:raw[key] for split in ('train', 'validation') for key in ('input', 'lengths', 'labels')}
    for split in ('train', 'validation'):
        assert len(data[split+'_lengths']) == 32 and np.all(data[split+'_lengths'] == 6)
        assert data[split+'_labels'].shape == (32, 4)
    assert np.array_equal(data['train_input'], data['validation_input']), 'Overfit scoring must use the same fixed32 engineering samples'
    assert np.array_equal(data['train_labels'], data['validation_labels'])
    assert all(len(np.unique(data['train_labels'][:,i])) > 1 for i in range(4)), 'All engineering tasks require nonconstant labels'
    return data


def loss_mask_check(data):
    results = []
    for ti, task in enumerate(TASKS):
        gold = data['train_labels'][:, ti].astype(np.int64)
        ids, mask, target, end, eq = batch(data['train_input'], data['train_lengths'], task, gold, device='cpu')
        rows = torch.arange(32)
        assert torch.all((target != -100).sum(1) == 2)
        assert torch.equal(target[rows, eq+1], torch.as_tensor(gold))
        assert torch.all(target[rows, eq+2] == TOKEN_TO_ID['<EOS>'])
        assert torch.equal(eq, end+2) and ids.shape[1] <= 128
        generator = torch.Generator(device='cpu').manual_seed(2026101137+ti)
        z = torch.randn((32, ids.shape[1], 188), generator=generator, dtype=torch.float64)
        from .model_runtime import loss
        original = loss(z, target)
        independent = (F.cross_entropy(z[rows, eq], torch.as_tensor(gold)) +
            F.cross_entropy(z[rows, eq+1], torch.full((32,), TOKEN_TO_ID['<EOS>'], dtype=torch.long)))/2
        difference = abs(float(original-independent)); assert difference <= 1e-12
        results.append(dict(task=task, samples=32, nonconstant_answers=True, targets_per_example=2,
            answer_logit_position='eq', answer_target_position='eq+1', EOS_logit_position='eq+1', EOS_target_position='eq+2',
            double_CE_max_difference=difference, tolerance=1e-12))
    return results


def difference(a, b):
    if torch.is_tensor(a):
        assert a.shape == b.shape and a.dtype == b.dtype
        return float((a.cpu().double()-b.cpu().double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([difference(a[k], b[k]) for k in a]+[0.])
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max([difference(x,y) for x,y in zip(a,b)]+[0.])
    assert a == b
    return 0.


def resume_child(path, deadline):
    """Fresh process branch only; parent chooses the fixed16->32 check."""
    path = Path(path).resolve(); assert path.is_relative_to(NEW_ROOT/'engineering')
    inputs = torch.load(path, map_location='cpu', weights_only=False)
    return run_job(inputs['protocol'], inputs['job'], inputs['data'], inputs['schedule'], inputs['initial_ref'],
        float(deadline), inputs['output_dir'], inputs['source_binding'], engineering=True, stop_at=32, constant_lr=False)


def run(protocol, raw_engineering_data, initial_ref, absolute_gpu_deadline, *,
        formal_data, formal_schedule, source_binding, output_dir=None):
    """Called solely by the root after math/data/protection gates and GPU start."""
    started = time.monotonic(); deadline_guard(absolute_gpu_deadline)
    out = Path(output_dir or NEW_ROOT/'engineering').resolve()
    assert out.is_relative_to((NEW_ROOT/'engineering').resolve())
    out.mkdir(parents=True, exist_ok=True)
    final = out/'engineering_audit.json'
    assert not final.exists(), 'One fixed engineering invocation; inspect any prior result instead of repeating'
    data = engineering_data(raw_engineering_data)
    mask_checks = loss_mask_check(data)
    settings = protocol['engineering']; assert settings['overfit_updates'] == 500
    assert settings['overfit_learning_rate'] == protocol['training']['learning_rate'] == .0003
    rng = np.random.default_rng(int(settings.get('seed', 98000031)))
    overfit_schedule = rng.integers(0,32,size=(500,64),dtype=np.int32)
    eng_binding = dict(source_binding, data_sha256=payload_sha({k:array_hash(v) for k,v in data.items()}),
                       batch_schedule_sha256=array_hash(overfit_schedule))
    records, problems = [], []
    for task in TASKS:
        if not deadline_guard(absolute_gpu_deadline, 30.):
            problems.append('GPU absolute budget exhausted before fixed overfit task '+task); break
        job = dict(run_id='engineering_'+task, unit=initial_ref['unit'], task=task)
        receipt = run_job(protocol, job, data, overfit_schedule, initial_ref, absolute_gpu_deadline,
            out/'overfit'/task, eng_binding, engineering=True, stop_at=500, constant_lr=True)
        v = receipt.get('final_validation')
        passed = (receipt['status'] == 'engineering_endpoint' and receipt['step'] == 500 and
            receipt['finite_gradient_updates'] == 500 and receipt['maximum_gradient_norm'] > 0 and
            receipt['parameter_sha256'] != initial_ref['initial_parameter_sha256'] and v is not None and
            v['answer_accuracy'] >= settings['overfit_threshold_full_vocabulary_answer_accuracy'])
        records.append(dict(task=task, passed=passed, receipt=receipt, accuracy_is_full188=True,
                            fixed_updates=500, learning_rate=.0003, constant_lr=True, formal_warmup_used=False))
        if not passed:
            problems.append('Fixed engineering overfit failed: '+task)
            break  # No tuning, no additional iterations or replacement task.
    resume = None; estimate = None
    if not problems:
        # First32 rows of the fixed formal length/batch schedule; all three
        # lengths must be represented. This is engineering, not a source run.
        schedule = {'batch_ids': np.array(schedule_ids(formal_schedule)[:32], copy=True)}
        assert set(map(int, np.unique(formal_data['train_lengths'][schedule['batch_ids']]))) == {6,8,10}
        job = dict(run_id='engineering_resume_F1', unit=initial_ref['unit'], task='F1')
        continuous = run_job(protocol, job, formal_data, schedule, initial_ref, absolute_gpu_deadline,
            out/'resume/continuous', source_binding, engineering=True, stop_at=32)
        split = run_job(protocol, job, formal_data, schedule, initial_ref, absolute_gpu_deadline,
            out/'resume/split', source_binding, engineering=True, stop_at=16)
        if continuous['status'] != 'engineering_endpoint' or split['status'] != 'engineering_endpoint':
            problems.append('Fixed CUDA resume check interrupted by absolute budget')
        else:
            inputs_path = out/'resume/resume_inputs.pt'
            atomic_checkpoint(inputs_path, dict(protocol=protocol, job=job, data=formal_data, schedule=schedule,
                initial_ref=initial_ref, output_dir=str(out/'resume/split'), source_binding=source_binding))
            child_log = out/'resume/child.log'
            with child_log.open('w') as stream:
                child = subprocess.run([sys.executable, '-B', '-m', 'experiments.matrixgroupworld_20261011.engineering',
                    '--resume-child', str(inputs_path), '--deadline', repr(float(absolute_gpu_deadline))],
                    cwd=Path(__file__).resolve().parents[2], stdout=stream, stderr=subprocess.STDOUT,
                    timeout=max(1., absolute_gpu_deadline-time.monotonic()))
            if child.returncode != 0:
                problems.append('Fresh-process CUDA restore branch failed; inspect child.log')
            else:
                a = torch.load(out/'resume/continuous/checkpoint.pt', map_location='cpu', weights_only=False)
                b = torch.load(out/'resume/split/checkpoint.pt', map_location='cpu', weights_only=False)
                pe, oe = difference(a['model'],b['model']), difference(a['optimizer'],b['optimizer'])
                same_rng = rng_hash(a['rng']) == rng_hash(b['rng'])
                logits_error = None
                if deadline_guard(absolute_gpu_deadline, 10.):
                    dev = device_setup('cuda')
                    nets = [load_saved_model(protocol, out/'resume'/sub/'checkpoint.pt', dev) for sub in ('continuous','split')]
                    indexes = np.arange(16)
                    x, mask, _, end, eq = batch(formal_data['validation_input'][indexes], formal_data['validation_lengths'][indexes],
                                              'F1', formal_data['validation_labels'][indexes,0], device=dev)
                    with torch.inference_mode():
                        z, zz = nets[0](x,mask), nets[1](x,mask)
                        logits_error = float((z-zz).abs().max())
                        # Prefix causality and answer-position supervision are
                        # checked independently of task learning/geometry.
                        suffix = x.clone(); rows = torch.arange(16,device=dev)
                        suffix[rows,eq+1] = (suffix[rows,eq+1]+1) % 31
                        suffix[rows,eq+2] = TOKEN_TO_ID['<PAD>']
                        changed = nets[0](suffix, suffix.ne(TOKEN_TO_ID['<PAD>']))
                        causal_error = float((z[rows,eq]-changed[rows,eq]).abs().max())
                    del nets; torch.cuda.empty_cache()
                else:
                    causal_error = None
                    problems.append('Budget prevented frozen restore logits/causality check')
                passed = pe <= settings['parameter_tolerance'] and oe <= settings['optimizer_tolerance'] and same_rng and logits_error is not None and logits_error <= settings['logit_tolerance'] and causal_error is not None and causal_error <= settings['logit_tolerance']
                resume = dict(passed=passed, continuous_updates=32, split_updates=16, fresh_process_resumed_updates=16,
                    parameter_max_difference=pe, optimizer_max_difference=oe, logits_max_difference=logits_error,
                    causal_suffix_logits_max_difference=causal_error, same_final_RNG=same_rng,
                    tolerances={k:settings[k] for k in ('parameter_tolerance','optimizer_tolerance','logit_tolerance')},
                    source_endpoints_produced=0, no_tolerance_relaxation=True)
                if not passed: problems.append('Fixed CUDA continuous/fresh-process resume gate failed')
                if not problems:
                    by_n = {}
                    for n in (6,8,10):
                        times = [r['update_wall_seconds'] for r in continuous['update_timings'] if r['length'] == n and r['step'] > 3]
                        assert times
                        by_n[str(n)] = dict(updates=len(times), mean_seconds=math.fsum(times)/len(times), maximum_seconds=max(times))
                    step_estimate = math.fsum(v['mean_seconds'] for v in by_n.values())/3
                    val = continuous['final_validation']
                    validation_seconds = val['wall_seconds']; cp_seconds = continuous['last_checkpoint_wall_seconds']
                    # Conservative full logical extraction and identity train
                    # scoring, no assumed shared-initial-forward optimization.
                    extraction_examples = 8*2*8*(3*256+3*512)
                    train_eval_examples = 8*3*4096
                    inference_seconds = val['wall_seconds']*(extraction_examples+train_eval_examples)/(3*256)
                    remaining_unscaled = 160000*step_estimate + 8*20*(validation_seconds+cp_seconds) + inference_seconds
                    headroom = 1.25
                    elapsed = time.monotonic()-started
                    estimate = dict(formal_models=8, formal_updates=160000, per_n_update_seconds=by_n,
                        equal_n_mean_update_seconds=step_estimate, empirical_validation768_wall_seconds=validation_seconds,
                        empirical_terminal_checkpoint_wall_seconds=cp_seconds, full_logical_extraction_examples=extraction_examples,
                        train_identity_evaluation_examples=train_eval_examples, conservative_inference_wall_seconds=inference_seconds,
                        remaining_unscaled_wall_seconds=remaining_unscaled, headroom_multiplier=headroom,
                        engineering_elapsed_wall_seconds=elapsed, estimated_total_gpu_stage_wall_seconds=elapsed+headroom*remaining_unscaled,
                        remaining_absolute_budget_seconds=absolute_gpu_deadline-time.monotonic(),
                        estimated_remaining_within_budget=headroom*remaining_unscaled <= absolute_gpu_deadline-time.monotonic(),
                        timings_are_process_wall_not_GPU_active=True, one_fixed_recipe=True,
                        cost_is_prediction_not_guarantee=True)
    result = dict(status='passed' if not problems else 'failed', overfit_records=records, mask_checks=mask_checks,
        resume=resume, cost_estimate=estimate, problems=problems, formal_models_produced=0,
        overfit_updates_per_task_limit=500, overfit_tasks=4, overfit_constant_actual_lr=.0003,
        formal_recipe_warmup_updates=1000, formal_recipe_changed=False, fixed_CUDA_resume_training_updates=64,
        source_binding=source_binding, initial_ref=initial_ref, elapsed_seconds=time.monotonic()-started,
        absolute_gpu_deadline=absolute_gpu_deadline, geometry_read=False, analysis_labels_read=False,
        no_automatic_formal_training=True, no_tuning_or_retry=True)
    json_write(final, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-child', type=Path, required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    result = resume_child(args.resume_child, args.deadline)
    print(json.dumps(dict(status=result['status'], step=result['step']), ensure_ascii=False))
    raise SystemExit(result['status'] != 'engineering_endpoint')
