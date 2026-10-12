"""Engineering validation of new configuration, never selected by geometry."""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .common import *
from experiments.permworld_encoding.core import encode, decode, labels
from experiments.permworld_encoding_archdev_20261009.training import difference


def metric_fixture():
    """Known-logit, unequal-length toy checks evaluation independently of fitting."""
    from .training import accuracy, learning_rate
    class KnownLogits(torch.nn.Module):
        def forward(self, ids, mask):
            out = torch.zeros(*ids.shape, len(TOKEN_TO_ID))
            out[:, :, TOKEN_TO_ID['01']] = 2.
            for row, n in enumerate(ids[:, 2].tolist()):
                out[row, 2*n+6, TOKEN_TO_ID['<EOS>']] = 3. if n == 10 else 1.
            return out
    lengths = np.array([10, 10, 12])
    inputs = np.full((3, 1, 64), TOKEN_TO_ID['<PAD>'], dtype=np.uint16)
    for i, n in enumerate(lengths):
        prefix = encode(tuple(range(1, int(n)+1)), 'one_line')
        inputs[i, 0, :len(prefix)] = prefix
    targets = np.zeros((3, 1, len(TASKS)), dtype=np.uint8)
    targets[:, 0, TASKS.index('descents')] = [1, 0, 1]
    data = dict(validation_lengths=lengths, validation_labels=targets,
                validation_one_line_input=inputs)
    net = KnownLogits().train()
    got = accuracy(net, data, 'one_line', 'descents', torch.device('cpu'))
    assert net.training
    assert got['overall_full_vocab'] == got['overall_numeric'] == got['overall_eos'] == 2/3
    expected_answer = math.log(math.exp(2)+187)-4/3
    expected_eos = (2*(math.log(math.exp(3)+math.exp(2)+186)-3)
                    +math.log(math.exp(1)+math.exp(2)+186)-1)/3
    assert abs(got['answer_ce']-expected_answer) < 1e-6
    assert abs(got['eos_ce']-expected_eos) < 1e-6
    assert abs(got['combined_ce']-.5*(expected_answer+expected_eos)) < 1e-6
    try:
        accuracy(net, data, 'one_line', 'descents', torch.device('cpu'), 'test')
    except AssertionError:
        pass
    else:
        raise AssertionError('Reserved test was not rejected')
    settings = dict(updates=8000, warmup_updates=400, minimum_lr_ratio=.1, learning_rate=.001)
    assert learning_rate(settings, 400) == .001
    assert abs(learning_rate(settings, 8000)-.0001) < 1e-16
    full = [learning_rate(settings, s) for s in (1, 400, 2000, 4000, 8000)]
    partitioned = [learning_rate(settings, s) for part in ((1,400,2000),(4000,8000)) for s in part]
    assert full == partitioned
    return dict(status='passed', known_logits_unequal_length_weighting=True,
        answer_and_EOS_CE_independently_checked=True, full_and_numeric_accuracy=True,
        model_training_mode_restored=True, test_rejected=True,
        global_LR_schedule_endpoints_and_partition_checked=True, no_training=True)


def cpu_check():
    config = verify(); data, schedule, audit = load_data(config)
    records = []
    torch.set_num_threads(2)
    selected = np.concatenate([np.flatnonzero(data['train_lengths'] == n)[:4] for n in config['lengths']])
    assert schedule.shape == (8000, 64)
    for n in config['lengths']:
        assert int(np.sum(data['train_lengths'] == n)) == 2048
        assert int(np.sum(data['validation_lengths'] == n)) == 1024
    for row in selected:
        n = int(data['train_lengths'][row]); p = tuple(map(int, data['train_permutations'][row, :n]))
        for encoding in ENCODINGS:
            prefix = data[f'train_{encoding}_input'][row, 0, :2*n+4]
            assert decode(prefix, encoding) == p
            assert tuple(prefix) == encode(p, encoding)
        assert tuple(data['train_labels'][row, 0]) == labels(p)
    for architecture in ARCHITECTURES:
        saved, meta = initial(config, architecture)
        net = model(config, architecture, config['architecture_model_seeds'][architecture], 'cpu')
        net.load_state_dict(saved['model']); net.eval()
        assert parameter_hash(net) == meta['parameter_sha256']
        for encoding in ENCODINGS:
            for task in DEV_TASKS:
                ns = data['train_lengths'][selected]
                y = data['train_labels'][selected, 0, TASKS.index(task)]
                ids, mask, target, end, eq = batch(data[f'train_{encoding}_input'][selected, 0], ns, task, y)
                rows = torch.arange(len(selected)); truth = torch.as_tensor(y.astype('int64'))
                assert torch.all((target != -100).sum(1) == 2)
                toy = torch.randn(*ids.shape, len(TOKEN_TO_ID), requires_grad=True)
                independent = .5 * (F.cross_entropy(toy[rows, eq], truth) +
                    F.cross_entropy(toy[rows, eq+1], torch.full_like(truth, TOKEN_TO_ID['<EOS>'])))
                actual = loss(toy, target)
                # Independent reductions may differ by a few FP32 ulps. Check
                # the exact same logits in float64 before bounding FP32 rounding.
                toy64 = toy.detach().double()
                independent64 = .5 * (F.cross_entropy(toy64[rows, eq], truth) +
                    F.cross_entropy(toy64[rows, eq+1], torch.full_like(truth, TOKEN_TO_ID['<EOS>'])))
                actual64 = loss(toy64, target)
                loss64_error = abs(float(independent64-actual64))
                loss32_error = abs(float((independent-actual).detach()))
                rounding_limit = 8 * torch.finfo(torch.float32).eps * max(1., abs(float(actual.detach())))
                assert loss64_error < 1e-12
                assert loss32_error <= rounding_limit
                actual.backward()
                expected = torch.zeros_like(target, dtype=torch.bool)
                expected[rows, eq] = True; expected[rows, eq+1] = True
                assert torch.equal(toy.grad.abs().sum(-1) > 0, expected)
                prompt, pmask, _, _, peq = batch(data[f'train_{encoding}_input'][selected, 0], ns, task)
                with torch.inference_mode():
                    full = net(ids, mask)[rows, eq]
                    short = net(prompt, pmask)[rows, peq]
                causal = float((full-short).abs().max())
                assert causal <= 2e-5
                records.append(dict(architecture=architecture, encoding=encoding, task=task,
                    independently_checked_shift_and_loss_mask=True, causal_max_difference=causal,
                    float64_independent_loss_error=loss64_error,
                    float32_reduction_difference=loss32_error, float32_rounding_bound=rounding_limit))
    json_write(ROOT / 'cpu_check.json', dict(status='passed', protocol_sha256=sha(CONFIG),
        records=records, metric_fixture=metric_fixture(), no_training=True, exact_encoding_roundtrips=True, actual_initial_identity=True,
        mathematical_labels_equal_across_encodings=True, geometry_not_used=True))


def child(job, step, out):
    subprocess.run([sys.executable, '-m', 'experiments.permworld_learnability_20261010',
        '--stage', 'train-one', '--run-id', job['run_id'], '--stop-step', str(step),
        '--output', str(out)], cwd=WORKSPACE, check=True)


def resume_check():
    config = verify(); records = []
    def budget_stop():
        json_write(ROOT / 'resume_check.json', dict(status='budget_exhausted', records=records,
            protocol_sha256=sha(CONFIG), source_models_not_started=True,
            geometry_not_used=True, no_false_resume_pass=True))
    for architecture in ARCHITECTURES:
        for encoding in ENCODINGS:
            job = next(j for j in config['training_manifest'] if
                j['architecture'] == architecture and j['encoding'] == encoding and j['task'] == 'descents')
            out = ROOT / 'engineering/resume' / architecture / encoding
            out.mkdir(parents=True, exist_ok=True)
            result = out / 'result.json'
            if result.exists():
                item = json.loads(result.read_text()); assert item['status'] == 'passed'
                assert item['protocol_sha256'] == sha(CONFIG)
                records.append(item); continue
            budget = json.loads((ROOT / 'gpu_budget.json').read_text())
            import time
            if time.time() >= budget['stop_updates_epoch']:
                budget_stop(); return
            continuous = out / 'continuous'; split = out / 'split'
            assert not (continuous / 'checkpoint.pt').exists(), 'Continuous reference cannot be silently resumed'
            child(job, 32, continuous)
            if torch.load(continuous / 'checkpoint.pt', map_location='cpu', weights_only=False)['step'] != 32:
                budget_stop(); return
            child(job, 16, split)
            if torch.load(split / 'checkpoint.pt', map_location='cpu', weights_only=False)['step'] != 16:
                budget_stop(); return
            child(job, 32, split)
            if torch.load(split / 'checkpoint.pt', map_location='cpu', weights_only=False)['step'] != 32:
                budget_stop(); return
            a = torch.load(continuous / 'checkpoint.pt', map_location='cpu', weights_only=False)
            b = torch.load(split / 'checkpoint.pt', map_location='cpu', weights_only=False)
            assert a['step'] == b['step'] == 32
            weights = difference(a['model'], b['model']); moments = difference(a['optimizer'], b['optimizer'])
            assert weights <= 1e-5 and moments <= 1e-5
            assert rng_hash(a['rng']) == rng_hash(b['rng'])
            item = dict(status='passed', architecture=architecture, encoding=encoding,
                protocol_sha256=sha(CONFIG), continuous_updates=32, resumed_after=16,
                parameter_max_difference=weights, optimizer_max_difference=moments,
                RNG_identical=True, independent_process_restart=True,
                continuous_sha256=sha(continuous / 'checkpoint.pt'), split_sha256=sha(split / 'checkpoint.pt'))
            json_write(result, item); records.append(item)
    json_write(ROOT / 'resume_check.json', dict(status='passed', records=records,
        source_models_not_preupdated=True, engineering_updates_counted_in_wall_budget=True,
        geometry_not_used=True))
