"""Validation-only fitting with fixed global schedules and safe budget stops.

This driver writes only to the new development root. It never evaluates the
reserved test split or calls a representation metric. Output overrides are for
the independently restarted engineering checks, not pretraining source models.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import (
    CONFIG, ROOT, DEV_TASKS, ENCODINGS, TASKS, TOKEN_TO_ID,
    atomic_checkpoint, batch, device_setup, initial, json_write, load_data,
    loss, model, parameter_hash, restore_rng, rng_hash, rng_state, sha,
    source_version, verify,
)


class ValidationBudgetExhausted(RuntimeError):
    """Do not turn an incomplete identity validation into a complete score."""


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _budget(config_digest):
    """Read the orchestrator's immutable deadline before each new update."""
    path = ROOT / 'gpu_budget.json'
    if not path.is_file():
        raise RuntimeError('GPU budget has not been established; refuse unbounded fitting')
    budget = json.loads(path.read_text())
    assert budget['protocol_sha256'] == config_digest, 'Budget belongs to another protocol'
    deadline = float(budget['deadline_epoch'])
    cutoff = float(budget['stop_updates_epoch'])
    assert math.isfinite(deadline) and 0 < cutoff <= deadline, 'Invalid fixed GPU deadline/cutoff'
    return budget, time.time() >= cutoff


def learning_rate(settings, step):
    """The same full-8k warmup/cosine schedule on every invocation."""
    final_step = int(settings['updates'])
    warmup = int(settings['warmup_updates'])
    assert 0 <= warmup < final_step and 1 <= step <= final_step
    if warmup and step <= warmup:
        factor = step / warmup
    else:
        factor = float(settings['minimum_lr_ratio']) + (
            1. - float(settings['minimum_lr_ratio'])) * .5 * (
                1. + math.cos(math.pi * (step - warmup) / (final_step - warmup)))
    return float(settings['learning_rate']) * factor


@torch.inference_mode()
def accuracy(net, data, encoding, task, device, split='validation', *, deadline_epoch=None):
    """Evaluate all identity examples in train or validation, never test.

    Answer CE uses all vocabulary logits. Numeric accuracy uses the conditional
    argmax over numeric answer tokens 0..30. EOS is teacher-forced at the known
    answer token. Combined CE equals the frozen answer-plus-EOS causal loss.
    Means are weighted by samples; the development validation lengths have equal
    counts, so its overall accuracy is also the equal-length mean.
    """
    assert split in ('train', 'validation'), 'Reserved test cannot be evaluated by this driver'
    assert encoding in ENCODINGS and task in DEV_TASKS
    lengths = np.asarray(data[split + '_lengths'])
    assert lengths.ndim == 1 and len(lengths) > 0
    expected_mode = net.training
    net.eval()
    records = []
    numeric_ids = torch.tensor([TOKEN_TO_ID[f'{n:02d}'] for n in range(31)], device=device)
    try:
        for length in sorted(map(int, np.unique(lengths))):
            chosen = np.flatnonzero(lengths == length)
            answers_correct = numeric_correct = eos_correct = 0
            answer_sum = eos_sum = 0.
            for start in range(0, len(chosen), 128):
                if deadline_epoch is not None and deadline_epoch-time.time() < 5.:
                    raise ValidationBudgetExhausted('Insufficient validation-batch margin; do not start another forward')
                use = chosen[start:start + 128]
                values = data[split + '_labels'][use, 0, TASKS.index(task)].astype('int64')
                assert np.all((0 <= values) & (values <= 30))
                ids, mask, _, _, equals = batch(
                    data[f'{split}_{encoding}_input'][use, 0], lengths[use],
                    task, values, device)
                if deadline_epoch is not None and deadline_epoch-time.time() < 5.:
                    raise ValidationBudgetExhausted('Insufficient margin before validation forward')
                logits = net(ids, mask)
                rows = torch.arange(len(use), device=device)
                truth = torch.tensor([TOKEN_TO_ID[f'{int(v):02d}'] for v in values], device=device)
                numeric_truth = torch.as_tensor(values, device=device)
                answer_logits = logits[rows, equals].float()
                eos_logits = logits[rows, equals + 1].float()
                eos_truth = torch.full_like(truth, TOKEN_TO_ID['<EOS>'])
                answers_correct += int((answer_logits.argmax(-1) == truth).sum())
                numeric_correct += int((answer_logits[:, numeric_ids].argmax(-1) == numeric_truth).sum())
                eos_correct += int((eos_logits.argmax(-1) == eos_truth).sum())
                answer_sum += float(F.cross_entropy(answer_logits, truth, reduction='sum'))
                eos_sum += float(F.cross_entropy(eos_logits, eos_truth, reduction='sum'))
            samples = len(chosen)
            record = dict(
                length=length, samples=samples,
                full_vocab_accuracy=answers_correct / samples,
                answer_accuracy=answers_correct / samples,
                numeric_accuracy=numeric_correct / samples,
                eos_accuracy=eos_correct / samples,
                answer_ce=answer_sum / samples,
                eos_ce=eos_sum / samples,
                combined_ce=(answer_sum + eos_sum) / (2 * samples),
            )
            assert all(math.isfinite(v) for v in record.values())
            records.append(record)
    finally:
        net.train(expected_mode)
    samples = sum(r['samples'] for r in records)
    def aggregate(field):
        return sum(r['samples'] * r[field] for r in records) / samples
    return dict(
        split=split, action='identity', samples=samples, per_length=records,
        overall_full_vocab=aggregate('full_vocab_accuracy'),
        overall_numeric=aggregate('numeric_accuracy'),
        overall_eos=aggregate('eos_accuracy'),
        answer_accuracy=aggregate('full_vocab_accuracy'),
        numeric_accuracy=aggregate('numeric_accuracy'),
        eos_accuracy=aggregate('eos_accuracy'),
        answer_ce=aggregate('answer_ce'), eos_ce=aggregate('eos_ce'),
        combined_ce=aggregate('combined_ce'),
        eos_teacher_forced=True, test_evaluated=False, geometry_evaluated=False,
    )


def _write_validation(path, job, step, validation):
    rows = [dict(run_id=job['run_id'], architecture=job['architecture'],
                 encoding=job['encoding'], task=job['task'], step=step,
                 split='validation', action='identity', **record)
            for record in validation['per_length']]
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _copy_checkpoint(source, target):
    """Keep a stage's complete state even after the latest checkpoint advances."""
    temporary = target.with_name(target.name + '.tmp')
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def run_job(run_id, stop_at, output=None):
    config = verify()
    matches = [j for j in config['training_manifest'] if j['run_id'] == run_id]
    assert len(matches) == 1, 'Run must be present exactly once in the fixed manifest'
    job = matches[0]
    assert not isinstance(stop_at, bool)
    requested = int(stop_at)
    settings = config['training']
    final_step = int(settings['updates'])
    assert 0 < requested <= final_step
    assert int(settings['effective_batch']) == int(settings['micro_batch']) * int(settings['accumulation'])
    protocol_digest, version = sha(CONFIG), source_version()
    budget, exhausted_on_start = _budget(protocol_digest)
    out = (ROOT / 'models' / run_id) if output is None else Path(output)
    out = out.resolve()
    assert out.is_relative_to(ROOT.resolve()), 'Output override must remain in the new development root'
    if output is not None:
        assert out.is_relative_to((ROOT / 'engineering').resolve()), 'Overrides are for new engineering checks only'
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / 'checkpoint.pt'
    stage_target = out / f'stage_{requested}.json'
    data, schedule, audit = load_data(config)
    if stage_target.exists():
        record = json.loads(stage_target.read_text())
        assert record['run_id'] == run_id and record['protocol_sha256'] == protocol_digest
        assert record['execution_source_version_sha256'] == version
        assert record['data_sha256'] == audit['data_sha256']
        if record['status'] == 'stage_complete':
            assert record['step'] == requested
            assert sha(record['checkpoint_path']) == record['checkpoint_sha256']
            print({'already_complete': run_id, 'step': requested}, flush=True)
            return record
    saved, meta = initial(config, job['architecture'])
    # Do not start a new CUDA model invocation past the updates cutoff. Its state
    # can still be saved safely on CPU without pretending validation completed.
    device = device_setup(cpu=exhausted_on_start)
    _synchronize(device)
    invocation_start_epoch = time.time()
    invocation_monotonic = time.monotonic()
    net = model(config, job['architecture'], job['model_seed'], device)
    net.load_state_dict(saved['model'])
    assert parameter_hash(net) == meta['parameter_sha256']
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=settings['learning_rate'],
        weight_decay=settings['weight_decay'], betas=tuple(settings['betas']),
        eps=settings['epsilon'], fused=device.type == 'cuda')
    stochastic_seed = int(job.get('stochastic_seed', config['units'][0]['dropout_seed_base']))
    random.seed(stochastic_seed)
    np.random.seed(stochastic_seed)
    torch.manual_seed(stochastic_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(stochastic_seed)
    starting_rng = rng_state()
    starting_rng_digest = rng_hash(starting_rng)
    rngfile = ROOT / 'initializations' / job['architecture'] / f'start_rng_{job["task"]}.pt'
    if rngfile.exists():
        prior = torch.load(rngfile, map_location='cpu', weights_only=False)
        assert prior['rng_hash'] == starting_rng_digest
        assert prior['parameter_sha256'] == meta['parameter_sha256']
        assert prior['protocol_sha256'] == protocol_digest
    else:
        atomic_checkpoint(rngfile, dict(
            rng=starting_rng, rng_hash=starting_rng_digest,
            parameter_sha256=meta['parameter_sha256'], protocol_sha256=protocol_digest,
            stochastic_seed=stochastic_seed))
    step, finite_updates, curve = 0, 0, []
    minimum_norm, maximum_norm = math.inf, 0.
    prior_elapsed = 0.
    if checkpoint.exists():
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert cp['run_id'] == run_id and cp['output_directory'] == str(out)
        assert cp['protocol_sha256'] == protocol_digest
        assert cp['execution_source_version_sha256'] == version
        assert cp['data_sha256'] == audit['data_sha256']
        assert cp['initial_parameter_sha256'] == meta['parameter_sha256']
        assert cp['initial_checkpoint_sha256'] == meta['checkpoint_sha256']
        assert cp['starting_rng_hash'] == starting_rng_digest
        assert cp['fixed_endpoint_step'] == final_step
        net.load_state_dict(cp['model'])
        optimizer.load_state_dict(cp['optimizer'])
        restore_rng(cp['rng'])
        step, finite_updates, curve = int(cp['step']), int(cp['finite_gradient_updates']), cp['curve']
        minimum_norm = math.inf if cp['minimum_gradient_norm'] is None else float(cp['minimum_gradient_norm'])
        maximum_norm = float(cp['maximum_gradient_norm'])
        prior_elapsed = float(cp['elapsed_seconds'])
        previous_stage = out / f'stage_{step}.json'
        if previous_stage.exists():
            previous_record = json.loads(previous_stage.read_text())
            assert previous_record['run_id'] == run_id and previous_record['protocol_sha256'] == protocol_digest
            prior_elapsed = float(previous_record['elapsed_seconds'])
    assert step <= requested, 'Cannot rewind a newer checkpoint to an earlier requested stage'
    assert finite_updates == step, 'A counted update must have finite checked gradients'
    invocation_start_step = step
    parameters = sum(p.numel() for p in net.parameters())
    latest_validation = None
    latest_validation_step = None
    exhausted = exhausted_on_start

    def cpu_state(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {k: cpu_state(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cpu_state(v) for v in value]
        if isinstance(value, tuple):
            return tuple(cpu_state(v) for v in value)
        return value

    def validate_current():
        current_budget, _ = _budget(protocol_digest)
        deadline = float(current_budget['deadline_epoch'])
        if exhausted_on_start or deadline - time.time() < 5.:
            return None
        try:
            return accuracy(net, data, job['encoding'], job['task'], device,
                            deadline_epoch=deadline)
        except ValidationBudgetExhausted:
            return None

    def persist():
        _synchronize(device)
        atomic_checkpoint(checkpoint, dict(
            model=cpu_state(net.state_dict()), optimizer=cpu_state(optimizer.state_dict()), rng=rng_state(),
            step=step, curve=curve, run_id=run_id, output_directory=str(out),
            architecture=job['architecture'], protocol_sha256=protocol_digest,
            execution_source_version_sha256=version, data_sha256=audit['data_sha256'],
            initial_parameter_sha256=meta['parameter_sha256'],
            initial_checkpoint_sha256=meta['checkpoint_sha256'],
            starting_rng_hash=starting_rng_digest, fixed_endpoint_step=final_step,
            finite_gradient_updates=finite_updates,
            minimum_gradient_norm=None if math.isinf(minimum_norm) else minimum_norm,
            maximum_gradient_norm=maximum_norm,
            elapsed_seconds=prior_elapsed + time.monotonic() - invocation_monotonic,
            invocation_start_step=invocation_start_step, invocation_updates=step-invocation_start_step,
            learning_rate=optimizer.param_groups[0]['lr'], source_model_run=output is None,
        ))

    while step < requested:
        budget, expired = _budget(protocol_digest)
        if expired:
            exhausted = True
            break
        next_step = step + 1
        lr = learning_rate(settings, next_step)
        for group in optimizer.param_groups:
            group['lr'] = lr
        net.train()
        optimizer.zero_grad(set_to_none=True)
        value_total = 0.
        for part in range(int(settings['accumulation'])):
            width = int(settings['micro_batch'])
            use = schedule[step, part*width:(part+1)*width]
            ids, mask, target, _, _ = batch(
                data[f'train_{job["encoding"]}_input'][use, 0], data['train_lengths'][use],
                job['task'], data['train_labels'][use, 0, TASKS.index(job['task'])], device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                value = loss(net(ids, mask), target) / int(settings['accumulation'])
            assert torch.isfinite(value), 'Nonfinite training loss'
            value.backward()
            value_total += float(value.detach())
        norm = float(torch.nn.utils.clip_grad_norm_(
            net.parameters(), settings['gradient_clip'], error_if_nonfinite=True))
        assert math.isfinite(norm)
        optimizer.step()
        step = next_step
        finite_updates += 1
        minimum_norm, maximum_norm = min(minimum_norm, norm), max(maximum_norm, norm)
        if step % int(settings['validation_every']) == 0 or step == requested:
            latest_validation = validate_current()
            latest_validation_step = step if latest_validation is not None else None
            if latest_validation is not None:
                assert [r['length'] for r in latest_validation['per_length']] == config['lengths']
            record = dict(step=step, train_loss_last_update=value_total, learning_rate=lr,
                          validation=latest_validation,
                          validation_missing_due_to_budget=latest_validation is None)
            curve.append(record)
            print({'run': run_id, 'step': step,
                   'answer_accuracy': None if latest_validation is None else latest_validation['answer_accuracy'],
                   'answer_ce': None if latest_validation is None else latest_validation['answer_ce'],
                   'learning_rate': lr}, flush=True)
        if step % int(settings['checkpoint_every']) == 0:
            persist()
    # A budget stop is not a completed requested stage, even if the preceding
    # checkpoint exists. Finish one clearly marked validation and full save.
    if latest_validation_step != step:
        latest_validation = validate_current()
        latest_validation_step = step if latest_validation is not None else None
        if latest_validation is not None:
            assert [r['length'] for r in latest_validation['per_length']] == config['lengths']
        if not curve or curve[-1]['step'] != step:
            curve.append(dict(step=step, validation=latest_validation, budget_cleanup=exhausted,
                              validation_missing_due_to_budget=latest_validation is None))
    if latest_validation is not None:
        assert latest_validation['action'] == 'identity' and latest_validation['split'] == 'validation'
    persist()
    immutable_checkpoint = out / f'checkpoint_step_{step}.pt'
    _copy_checkpoint(checkpoint, immutable_checkpoint)
    if latest_validation is not None:
        _write_validation(out / f'validation_stage_{step}.csv', job, step, latest_validation)
    if step > 0:
        assert parameter_hash(net) != meta['parameter_sha256'], 'No parameter change after fitting'
    immutable_digest = sha(immutable_checkpoint)
    _synchronize(device)
    end_epoch = time.time()
    invocation_elapsed = time.monotonic() - invocation_monotonic
    status = 'stage_complete' if step == requested and latest_validation is not None else 'budget_exhausted'
    result = dict(
        **job, status=status, step=step, requested_stop_step=requested,
        fixed_endpoint_step=final_step,
        fixed_endpoint_updates_completed=step == final_step,
        fixed_endpoint_completed=step == final_step and latest_validation is not None,
        validation_missing_due_to_budget=latest_validation is None,
        validation=latest_validation, curve=curve, parameters=parameters,
        exposures=step*int(settings['effective_batch']),
        supervised_tokens=2*step*int(settings['effective_batch']),
        finite_gradient_updates=finite_updates,
        minimum_gradient_norm=None if math.isinf(minimum_norm) else minimum_norm,
        maximum_gradient_norm=maximum_norm,
        protocol_sha256=protocol_digest, execution_source_version_sha256=version,
        data_sha256=audit['data_sha256'], initial_parameter_sha256=meta['parameter_sha256'],
        initial_checkpoint_sha256=meta['checkpoint_sha256'], starting_rng_hash=starting_rng_digest,
        actual_initial_identity_verified=True, checkpoint_path=str(immutable_checkpoint),
        checkpoint_sha256=immutable_digest, output_directory=str(out),
        invocation_start_step=invocation_start_step, invocation_updates=step-invocation_start_step,
        invocation_start_epoch=invocation_start_epoch, invocation_end_epoch=end_epoch,
        invocation_elapsed_seconds=invocation_elapsed, elapsed_seconds=prior_elapsed+invocation_elapsed,
        budget_deadline_epoch=float(budget['deadline_epoch']),
        budget_stop_updates_epoch=float(budget['stop_updates_epoch']),
        cleanup_after_deadline_seconds=max(0., end_epoch-max(invocation_start_epoch,float(budget['deadline_epoch']))),
        budget_exhausted=status == 'budget_exhausted', validation_device=str(device),
        source_model_run=output is None, formal_training_started=False,
        test_evaluated=False, geometry_evaluated=False,
    )
    json_write(out / f'stage_{step}.json', result)
    if step == final_step and latest_validation is not None and output is None:
        json_write(out / 'completed.json', dict(result, status='completed'))
    print({'run': run_id, 'status': status, 'step': step, 'requested_stop_step': requested,
           'invocation_elapsed_seconds': invocation_elapsed}, flush=True)
    return result
