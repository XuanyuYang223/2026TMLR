"""Small CUDA RNG-resume engineering fixture; no PermWorld source/data use.

One process runs updates 1..3, saves all ordinary checkpoint state, and continues
through update 6. A fresh Python process reloads update 3 and performs 4..6.
This is a software diagnostic, excluded from every source-model experiment.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
STORAGE = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/gru_cuda_engineering_fixture')
MODEL_SEED = 98741001
DATA_SEED = 98741002
DROPOUT_SEED = 98741003


class FixtureGRU(nn.Module):
    def __init__(self, layout):
        super().__init__()
        self.layout = layout
        self.token_embedding = nn.Embedding(188, 256)
        self.position_embedding = nn.Embedding(128, 256)
        self.embedding_dropout = nn.Dropout(.1)
        if layout == 'native':
            self.gru = nn.GRU(256, 256, 4, batch_first=True, dropout=.1)
        else:
            self.layers = nn.ModuleList([nn.GRU(256, 256, 1, batch_first=True, dropout=0.) for _ in range(4)])
            self.between = nn.Dropout(.1)
        self.final_norm = nn.LayerNorm(256)
        self.lm_head = nn.Linear(256, 188, bias=False)
        nn.init.normal_(self.token_embedding.weight, std=.02)
        nn.init.normal_(self.position_embedding.weight, std=.02)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, ids):
        h = self.embedding_dropout(self.token_embedding(ids) +
                                   self.position_embedding(torch.arange(ids.shape[1], device=ids.device))[None])
        if self.layout == 'native':
            h, _ = self.gru(h)
        else:
            for i, gru in enumerate(self.layers):
                h, _ = gru(h)
                if i < 3:
                    h = self.between(h)
        return self.lm_head(self.final_norm(h))


def rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def rng_digest(state):
    h = hashlib.sha256()
    h.update(repr(state['python']).encode())
    h.update(state['numpy'][0].encode()); h.update(state['numpy'][1].tobytes())
    h.update(repr(state['numpy'][2:]).encode()); h.update(state['torch'].numpy().tobytes())
    for value in state['cuda']:
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def setup(layout):
    assert torch.cuda.is_available(), 'CUDA fixture requires CUDA; no CPU result disguised as CUDA.'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    net = FixtureGRU(layout).cuda()
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0003, weight_decay=.01,
                                 betas=(.9, .999), eps=1e-8, fused=True)
    generator = torch.Generator().manual_seed(DATA_SEED)
    ids = torch.randint(0, 31, (64, 24), generator=generator)
    answers = torch.randint(0, 4, (64,), generator=generator)
    ids[:, 22] = answers; ids[:, 23] = 187
    targets = torch.full_like(ids, -100); targets[:, 22] = answers; targets[:, 23] = 187
    torch.manual_seed(DROPOUT_SEED); torch.cuda.manual_seed_all(DROPOUT_SEED)
    return net, optimizer, ids.cuda(), targets.cuda()


def update(net, optimizer, ids, targets):
    net.train(); optimizer.zero_grad(set_to_none=True)
    total = 0.
    for k in range(4):
        sl = slice(16 * k, 16 * (k + 1))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = net(ids[sl])
            value = F.cross_entropy(logits[:, :-1].reshape(-1, 188), targets[sl, 1:].reshape(-1), ignore_index=-100) / 4
        assert torch.isfinite(value)
        value.backward(); total += float(value.detach())
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
    assert norm > 0
    optimizer.step()
    return total


def checkpoint(path, net, optimizer, step, losses):
    torch.cuda.synchronize()
    torch.save(dict(model=net.state_dict(), optimizer=optimizer.state_dict(), rng=rng(),
                    step=step, losses=losses, layout=net.layout, fixture_only=True), path)


def delta(a, b):
    if torch.is_tensor(a):
        assert a.shape == b.shape and a.dtype == b.dtype
        return float((a.double() - b.double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return max([delta(a[key], b[key]) for key in a] + [0.])
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        return max([delta(x, y) for x, y in zip(a, b)] + [0.])
    assert a == b
    return 0.


def child(path):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    net, optimizer, ids, targets = setup(saved['layout'])
    net.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); restore(saved['rng'])
    losses = []
    for _ in range(3):
        losses.append(update(net, optimizer, ids, targets))
    checkpoint(path.parent / 'resumed6.pt', net, optimizer, 6, losses)


def run(layout):
    start = time.monotonic()
    out = STORAGE / layout
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'initial3.pt').exists(), 'Fixture already exists; keep its evidence and explicitly choose a new storage path.'
    net, optimizer, ids, targets = setup(layout)
    first = [update(net, optimizer, ids, targets) for _ in range(3)]
    checkpoint(out / 'initial3.pt', net, optimizer, 3, first)
    continued = [update(net, optimizer, ids, targets) for _ in range(3)]
    checkpoint(out / 'continuous6.pt', net, optimizer, 6, continued)
    # Release parent's allocator before the independent fresh process.
    del net, optimizer, ids, targets; torch.cuda.empty_cache()
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(out / 'initial3.pt')], check=True)
    a = torch.load(out / 'continuous6.pt', map_location='cpu', weights_only=False)
    b = torch.load(out / 'resumed6.pt', map_location='cpu', weights_only=False)
    weight = delta(a['model'], b['model']); moments = delta(a['optimizer'], b['optimizer'])
    same_rng = rng_digest(a['rng']) == rng_digest(b['rng'])
    loss_error = max(abs(x - y) for x, y in zip(a['losses'], b['losses']))
    status = 'passed' if weight <= 1e-5 and moments <= 1e-5 and same_rng else 'failed'
    result = dict(status=status, layout=layout, software_engineering_fixture=True,
                  PermWorld_source_models=0, experimental_data_read=False, formal_training_started=False,
                  model_seed=MODEL_SEED, synthetic_data_seed=DATA_SEED, stochastic_seed=DROPOUT_SEED,
                  width=256, layers=4, native_interlayer_dropout=.1 if layout == 'native' else 0.,
                  explicit_torch_interlayer_dropout=.1 if layout == 'explicit' else 0.,
                  embedding_dropout=.1, updates_before_save=3, updates_after_save=3,
                  micro_batch=16, accumulation=4, sequence_length=24,
                  precision='FP32 parameters; BF16 CUDA autocast; TF32 disabled',
                  optimizer='fused AdamW lr3e-4 betas(.9,.999) eps1e-8 weight_decay.01; gradient clip1',
                  parameter_max_abs=weight, optimizer_max_abs=moments, RNG_identity=same_rng,
                  resumed_loss_max_abs=loss_error, continuous_losses=a['losses'], resumed_losses=b['losses'],
                  parameter_tolerance=1e-5, optimizer_tolerance=1e-5, RNG_exact_required=True,
                  torch=torch.__version__, torch_git_commit=torch.version.git_version,
                  cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  fixture_checkpoint_directory=str(out), elapsed_seconds=time.monotonic() - start)
    target = HERE / f'gru_cuda_resume_{layout}.json'
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layout', choices=('native', 'explicit'), default='native')
    parser.add_argument('--child', type=Path)
    args = parser.parse_args()
    if args.child:
        child(args.child)
    else:
        run(args.layout)
