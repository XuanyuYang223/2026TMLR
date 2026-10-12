"""Deadline-bounded identity-only source fitting on new MatrixGroupWorld data."""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .core import TASKS, batch
from .model_runtime import (NEW_ROOT, TOKEN_TO_ID, atomic_checkpoint, deadline_guard,
    device_setup, initial_rng, json_write, load_initialized_model, loss, parameter_hash,
    restore_rng, rng_hash, rng_state, sha)


def optimizer(net, settings, device):
    return torch.optim.AdamW(net.parameters(), lr=settings['learning_rate'],
        weight_decay=settings['weight_decay'], betas=tuple(settings['betas']),
        eps=settings['epsilon'], fused=device.type == 'cuda')


def schedule_ids(schedule):
    return schedule if isinstance(schedule, np.ndarray) else schedule['batch_ids']


def data_contract(data, schedule, maximum):
    assert not any(k.startswith(('analysis_', 'test_')) for k in data), 'Source learning never loads analysis labels or inputs'
    for split in ('train', 'validation'):
        assert all(split+'_'+key in data for key in ('input', 'lengths', 'labels'))
        count = len(data[split+'_lengths'])
        assert data[split+'_input'].shape == (count, 128)
        assert data[split+'_labels'].shape == (count, 4)
    ids = schedule_ids(schedule)
    assert ids.ndim == 2 and ids.shape[0] >= maximum and ids.shape[1] == 64
    assert np.issubdtype(ids.dtype, np.integer)
    assert int(ids[:maximum].min()) >= 0 and int(ids[:maximum].max()) < len(data['train_lengths'])


def loss_parts(logits, target, eq):
    rows = torch.arange(len(eq), device=logits.device)
    assert torch.all((target != -100).sum(1) == 2)
    gold, eos = target[rows, eq+1], target[rows, eq+2]
    assert torch.all(gold >= 0) and torch.all(eos == TOKEN_TO_ID['<EOS>'])
    answer_loss = F.cross_entropy(logits[rows, eq], gold)
    eos_loss = F.cross_entropy(logits[rows, eq+1], eos)
    combined = loss(logits, target)  # Exact original shifted, masked objective.
    return combined, answer_loss, eos_loss


def update(net, opt, data, schedule, task, index, settings, device, constant_lr=False):
    net.train(); opt.zero_grad(set_to_none=True)
    step = index+1
    if constant_lr:
        factor = 1.
    elif step <= settings['warmup_updates']:
        factor = step/settings['warmup_updates']
    else:
        fraction = (step-settings['warmup_updates'])/(settings['updates']-settings['warmup_updates'])
        factor = settings['minimum_lr_ratio']+(1-settings['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*fraction))
    for group in opt.param_groups:
        group['lr'] = settings['learning_rate']*factor
    ids = schedule_ids(schedule)[index]
    ns = data['train_lengths'][ids]
    assert np.all(ns == ns[0]), 'Frozen source schedule has one n per optimizer update'
    task_index = TASKS.index(task)
    total, answer, eos = 0., 0., 0.
    for part in range(settings['accumulation']):
        chosen = ids[part*settings['micro_batch']:(part+1)*settings['micro_batch']]
        x, mask, target, _, eq = batch(data['train_input'][chosen], data['train_lengths'][chosen], task,
                                      data['train_labels'][chosen, task_index], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            combined, a, e = loss_parts(net(x, mask), target, eq)
            value = combined/settings['accumulation']
        assert all(torch.isfinite(v) for v in (value, a, e))
        value.backward()
        total += float(value.detach()); answer += float(a.detach())/settings['accumulation']; eos += float(e.detach())/settings['accumulation']
    norm = float(torch.nn.utils.clip_grad_norm_(net.parameters(), settings['gradient_clip'], error_if_nonfinite=True))
    assert math.isfinite(norm)
    opt.step()
    return dict(train_combined_ce=total, train_answer_ce=answer, train_eos_ce=eos,
                gradient_norm=norm, learning_rate=settings['learning_rate']*factor, length=int(ns[0]))


@torch.inference_mode()
def evaluate_model(net, data, task, device, absolute_gpu_deadline, split='validation', reserve_seconds=3.):
    """Full188 FP32 answer/EOS scoring; returns None if budget prevents completion."""
    assert split in ('train', 'validation'), 'Training validation cannot inspect final analysis'
    net.eval(); task_index = TASKS.index(task)
    ns = data[split+'_lengths']; lengths = sorted(map(int, np.unique(ns)))
    totals = {n:dict(samples=0, answer=0, eos=0, mode=0, answer_ce_sum=0., eos_ce_sum=0.) for n in lengths}
    modes = {n:int(np.argmax(np.bincount(data['train_labels'][data['train_lengths'] == n, task_index].astype(np.int64), minlength=31))) for n in lengths}
    before = time.monotonic(); inference_seconds = 0.
    for start in range(0, len(ns), 128):
        if not deadline_guard(absolute_gpu_deadline, reserve_seconds):
            return None
        use = np.arange(start, min(start+128, len(ns))); n = ns[use]
        gold_np = data[split+'_labels'][use, task_index].astype(np.int64)
        x, mask, _, _, eq = batch(data[split+'_input'][use], n, task, gold_np, device=device)
        if not deadline_guard(absolute_gpu_deadline, reserve_seconds):
            return None
        if device.type == 'cuda': torch.cuda.synchronize()
        t = time.monotonic(); z = net(x, mask)
        if device.type == 'cuda': torch.cuda.synchronize()
        inference_seconds += time.monotonic()-t
        row = torch.arange(len(use), device=device); gold = torch.as_tensor(gold_np, device=device)
        answer_logits, eos_logits = z[row, eq].float(), z[row, eq+1].float()
        correct = (answer_logits.argmax(-1) == gold).cpu().numpy()
        eos_correct = (eos_logits.argmax(-1) == TOKEN_TO_ID['<EOS>']).cpu().numpy()
        ac = F.cross_entropy(answer_logits, gold, reduction='none').cpu().double().numpy()
        ec = F.cross_entropy(eos_logits, torch.full_like(gold, TOKEN_TO_ID['<EOS>']), reduction='none').cpu().double().numpy()
        for length in np.unique(n):
            k = n == length; v = totals[int(length)]
            v['samples'] += int(k.sum()); v['answer'] += int(correct[k].sum()); v['eos'] += int(eos_correct[k].sum())
            v['mode'] += int((gold_np[k] == modes[int(length)]).sum())
            v['answer_ce_sum'] += float(ac[k].sum()); v['eos_ce_sum'] += float(ec[k].sum())
    per = []
    for n, r in totals.items():
        count = r['samples']; assert count > 0
        per.append(dict(n=n, length=n, samples=count, answer_accuracy=r['answer']/count, eos_accuracy=r['eos']/count,
            answer_ce=r['answer_ce_sum']/count, eos_ce=r['eos_ce_sum']/count,
            combined_ce=(r['answer_ce_sum']+r['eos_ce_sum'])/(2*count), training_majority_label=modes[n],
            length_majority_accuracy=r['mode']/count, accuracy_minus_length_mode=(r['answer']-r['mode'])/count))
    result = dict(samples=len(ns), lengths=lengths, per_n=per, per_length=per,
        weights='Equal fixed n; no missing-length reweighting', full_vocabulary_argmax=True, teacher_conditioned_EOS=True,
        wall_seconds=time.monotonic()-before, inference_seconds=inference_seconds, validation_complete=True)
    for field in ('answer_accuracy', 'eos_accuracy', 'answer_ce', 'eos_ce', 'combined_ce', 'length_majority_accuracy', 'accuracy_minus_length_mode'):
        result[field] = math.fsum(r[field] for r in per)/len(lengths)
    result['minimum_length_accuracy'] = min(r['answer_accuracy'] for r in per)
    return result


def run_job(protocol, job, data, schedule, initial_ref, absolute_gpu_deadline, output_dir,
            source_binding, engineering=False, stop_at=None, constant_lr=False):
    """Idempotent resume; only a complete formal20k endpoint gets completed.json."""
    invocation_started = time.monotonic()
    assert absolute_gpu_deadline is not None
    directory = Path(output_dir).resolve(); root = NEW_ROOT.resolve()
    assert directory.is_relative_to(root)
    assert directory.is_relative_to(root/'engineering') if engineering else not directory.is_relative_to(root/'engineering')
    s = protocol['training']; assert s['updates'] == 20000 and s['micro_batch'] == 16 and s['accumulation'] == 4 and s['effective_batch'] == 64
    maximum = int(stop_at or s['updates'])
    assert (0 < maximum <= 500) if engineering else maximum == 20000
    assert not constant_lr or engineering, 'Constant LR is solely the explicitly disclosed engineering overfit'
    data_contract(data, schedule, maximum)
    required = ('protocol_sha256', 'source_version_sha256', 'data_sha256', 'batch_schedule_sha256')
    assert all(k in source_binding for k in required)
    identity = dict(run_id=job['run_id'], unit=job['unit'], task=job['task'], model_seed=initial_ref['model_seed'],
        **{k:source_binding[k] for k in required}, execution_source_version_sha256=source_binding['source_version_sha256'],
        initial_parameter_sha256=initial_ref['initial_parameter_sha256'], initial_checkpoint_sha256=initial_ref['initial_checkpoint_sha256'],
        starting_rng_hash=initial_ref['starting_rng_hash'], engineering=engineering, constant_lr=constant_lr)
    assert job['unit'] == initial_ref['unit'], 'Task assigned to another unit initialization'
    checkpoint, completed = directory/'checkpoint.pt', directory/'completed.json'
    if completed.exists():
        assert not engineering
        receipt = __import__('json').loads(completed.read_text())
        assert all(receipt[k] == v for k, v in identity.items())
        assert receipt['step'] == receipt['finite_gradient_updates'] == 20000 and sha(checkpoint) == receipt['checkpoint_sha256']
        return receipt
    if not deadline_guard(absolute_gpu_deadline, 30.):
        return dict(status='interrupted_before_model_setup', **identity, step=0, formal_completed=False)
    directory.mkdir(parents=True, exist_ok=True)
    device = device_setup('cuda')
    net = load_initialized_model(protocol, initial_ref, device); opt = optimizer(net, s, device)
    restore_rng(initial_rng(initial_ref)); assert rng_hash(rng_state()) == initial_ref['starting_rng_hash']
    step, curve, elapsed, finite, low, high, peak, saved_last_update = 0, [], 0., 0, math.inf, 0., 0, None
    if checkpoint.exists():
        import json
        sidecar = json.loads((directory/'checkpoint_identity.json').read_text())
        assert all(sidecar[k] == v for k, v in identity.items())
        assert sha(checkpoint) == sidecar['checkpoint_sha256'], 'Resume checkpoint bytes changed'
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert all(cp[k] == v for k, v in identity.items()), 'Resume identity/source/data/RNG binding changed'
        assert cp['step'] == cp['finite_gradient_updates'] and 0 <= cp['step'] <= maximum
        assert cp['parameter_sha256'] == parameter_hash_from_state(cp['model']), 'Saved resume parameter digest changed'
        assert cp['rng_hash'] == rng_hash(cp['rng']), 'Saved resume stochastic state changed'
        net.load_state_dict(cp['model'], strict=True); opt.load_state_dict(cp['optimizer']); restore_rng(cp['rng'])
        step, curve, elapsed, finite = cp['step'], cp['curve'], cp['elapsed_seconds'], cp['finite_gradient_updates']
        prior_partial = directory/'partial.json'
        if prior_partial.exists():
            prior = json.loads(prior_partial.read_text())
            assert all(prior[k] == v for k, v in identity.items())
            if prior['step'] == step and prior['checkpoint_sha256'] == sidecar['checkpoint_sha256']:
                elapsed = max(elapsed, prior['elapsed_seconds'])
        low = cp['minimum_gradient_norm'] if cp['minimum_gradient_norm'] is not None else math.inf
        high, peak, saved_last_update = cp['maximum_gradient_norm'], cp['peak_cuda_allocated_bytes'], cp.get('last_update')
        assert sidecar['step'] == step and sidecar['parameter_sha256'] == cp['parameter_sha256'] and sidecar['rng_hash'] == cp['rng_hash']
    started, start_step = invocation_started, step
    torch.cuda.reset_peak_memory_stats()
    reserve = float(protocol.get('budget', {}).get('safe_checkpoint_reserve_seconds', 30.))
    timings, final_validation, interrupted_reason, last_update = [], None, '', saved_last_update

    def save_checkpoint():
        nonlocal peak
        peak = max(peak, torch.cuda.max_memory_allocated())
        state = rng_state(); ph = parameter_hash(net)
        payload = dict(model=net.state_dict(), optimizer=opt.state_dict(), rng=state, rng_hash=rng_hash(state),
            parameter_sha256=ph, step=step, curve=curve, **identity, elapsed_seconds=elapsed+time.monotonic()-started,
            finite_gradient_updates=finite, minimum_gradient_norm=low if finite else None, maximum_gradient_norm=high,
            peak_cuda_allocated_bytes=peak, invocation_start_step=start_step, invocation_updates=step-start_step,
            model_config=protocol['model'], vocabulary=TOKEN_TO_ID, update_timings=timings,
            sample_exposures=step*64, supervised_tokens=step*128, last_update=last_update)
        t = time.monotonic(); atomic_checkpoint(checkpoint, payload)
        json_write(directory/'checkpoint_identity.json', dict(**identity, step=step, checkpoint_sha256=sha(checkpoint),
            parameter_sha256=ph, rng_hash=payload['rng_hash'], elapsed_seconds=payload['elapsed_seconds']))
        return time.monotonic()-t

    if step and (step % s['validation_every'] == 0 or step == maximum) and (not curve or curve[-1]['step'] != step):
        # Recover a pending validation at its actual saved weights before
        # advancing. Otherwise a budget interruption could lose that point.
        final_validation = evaluate_model(net, data, job['task'], device, absolute_gpu_deadline, reserve_seconds=5.)
        if final_validation is None:
            interrupted_reason = 'saved_step_validation_incomplete_due_to_absolute_gpu_deadline'
        else:
            assert last_update is not None
            curve.append(dict(step=step, **last_update, validation=final_validation, resumed_validation=True))

    for index in range(step, maximum):
        if interrupted_reason:
            break
        if not deadline_guard(absolute_gpu_deadline, reserve):
            interrupted_reason = 'absolute_gpu_deadline_checkpoint_reserve'; break
        torch.cuda.synchronize(); before = time.monotonic()
        last_update = update(net, opt, data, schedule, job['task'], index, s, device, constant_lr)
        torch.cuda.synchronize(); duration = time.monotonic()-before
        step = index+1; finite += 1; low = min(low, last_update['gradient_norm']); high = max(high, last_update['gradient_norm'])
        if engineering: timings.append(dict(step=step, length=last_update['length'], update_wall_seconds=duration))
        if step % s['validation_every'] == 0 or step == maximum:
            final_validation = evaluate_model(net, data, job['task'], device, absolute_gpu_deadline, reserve_seconds=5.)
            if final_validation is None:
                interrupted_reason = 'validation_incomplete_due_to_absolute_gpu_deadline'; break
            record = dict(step=step, **last_update, validation=final_validation)
            curve.append(record); json_write(directory/f'stage_{step}.json', dict(**identity, **record))
            print(dict(run_id=job['run_id'], step=step, answer_accuracy=final_validation['answer_accuracy'],
                       answer_ce=final_validation['answer_ce'], eos_ce=final_validation['eos_ce']), flush=True)
        if step % s['checkpoint_every'] == 0 and step < maximum:
            save_checkpoint()
    if step == maximum and final_validation is None and not interrupted_reason:
        # A final saved state may have interrupted immediately before final
        # validation on an earlier invocation; do not silently complete it.
        final_validation = evaluate_model(net, data, job['task'], device, absolute_gpu_deadline, reserve_seconds=5.)
        if final_validation is None:
            interrupted_reason = 'endpoint_validation_incomplete_due_to_absolute_gpu_deadline'
        elif not curve or curve[-1]['step'] != step:
            assert last_update is not None
            curve.append(dict(step=step, **last_update, validation=final_validation, resumed_endpoint_validation=True))
    checkpoint_seconds = save_checkpoint()
    formal_complete = not engineering and step == 20000 and finite == 20000 and final_validation is not None and not interrupted_reason
    ph = parameter_hash(net)
    receipt = dict(**identity, status='completed' if formal_complete else 'engineering_endpoint' if engineering and step == maximum and not interrupted_reason else 'interrupted',
        step=step, fixed_endpoint_updates=20000, formal_completed=formal_complete, checkpoint_sha256=sha(checkpoint),
        parameter_sha256=ph, final_parameter_sha256=ph, actual_initial_identity_verified=True, finite_gradient_updates=finite,
        minimum_gradient_norm=low if finite else None, maximum_gradient_norm=high, curve=curve,
        final_validation=final_validation, validation_complete=final_validation is not None,
        interrupted_reason=interrupted_reason, elapsed_seconds=elapsed+time.monotonic()-started,
        invocation_start_step=start_step, invocation_updates=step-start_step, peak_cuda_allocated_bytes=peak,
        parameters=sum(p.numel() for p in net.parameters()), sample_exposures=step*64, exposures=step*64,
        supervised_tokens=step*128, update_timings=timings, last_checkpoint_wall_seconds=checkpoint_seconds,
        performance_is_not_a_formal_stopping_rule=True)
    if formal_complete:
        assert ph != initial_ref['initial_parameter_sha256'] and high > 0
        assert [r['step'] for r in curve] == list(range(1000, 20001, 1000))
        json_write(completed, receipt)
    else:
        json_write(directory/'partial.json', receipt)
    del net, opt
    torch.cuda.empty_cache()
    return receipt


def parameter_hash_from_state(state):
    # Same sorted state-dictionary digest as the frozen existing helper.
    import hashlib
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        for part in (name, str(a.dtype), str(a.shape)): h.update(part.encode())
        h.update(a.tobytes())
    return h.hexdigest()
