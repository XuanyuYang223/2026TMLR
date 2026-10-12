"""New MatrixGroupWorld runtime; reuse old classes without changing old globals."""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from experiments.permworld_encoding.core import CausalTransformer, TOKEN_TO_ID as OLD_VOCAB, loss, parameter_hash
from experiments.longrun_attention import accelerate

TOKEN_TO_ID = dict(OLD_VOCAB)
MODEL_CFG = dict(vocab_size=188, max_seq_len=128, d_model=256, layers=4,
                 dropout=.1, heads=8, mlp_ratio=4, tie_embeddings=True)
NEW_ROOT = Path('/mnt/yangx-projects-f/20261011/matrixgroupworld')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def payload_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: tree_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [tree_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(tree_cpu(v) for v in value)
    return copy.deepcopy(value)


def atomic_checkpoint(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(NEW_ROOT.resolve()), 'Only the new MatrixGroupWorld root is writable'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    # Materialize CPU state before serialization, including on interruption.
    torch.save(tree_cpu(value), temporary)
    os.replace(temporary, path)


def device_setup(device='cuda'):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    result = torch.device(device)
    assert result.type != 'cuda' or torch.cuda.is_available(), 'No implicit CPU substitute for source training'
    return result


def deadline_guard(deadline, reserve=0.):
    assert deadline is not None and np.isfinite(deadline), 'An absolute monotonic GPU deadline is mandatory'
    return time.monotonic() + float(reserve) < float(deadline)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        assert torch.cuda.is_available() and len(state['cuda']) == torch.cuda.device_count()
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def rng_hash(state):
    h = hashlib.sha256(repr(state['python']).encode())
    h.update(state['numpy'][0].encode())
    h.update(state['numpy'][1].tobytes())
    h.update(repr(state['numpy'][2:]).encode())
    h.update(state['torch'].cpu().numpy().tobytes())
    for value in state['cuda']:
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def build_model(protocol, seed, device='cpu', reseed=True):
    config = dict(protocol.get('model', MODEL_CFG))
    assert config == MODEL_CFG, ('Fixed Transformer configuration changed', config)
    assert len(TOKEN_TO_ID) == 188 and TOKEN_TO_ID == OLD_VOCAB
    if reseed:
        random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    net = CausalTransformer(vocab_size=config['vocab_size'], max_seq_len=config['max_seq_len'],
        d_model=config['d_model'], layers=config['layers'], dropout=config['dropout'], n_heads=config['heads'],
        mlp_ratio=config['mlp_ratio'], tie_embeddings=config['tie_embeddings'])
    accelerate(net)  # Patches only this new model instance's attention methods.
    assert net.lm_head.weight is net.token_embedding.weight
    assert sum(p.numel() for p in net.parameters()) == 3240448
    return net.to(device)


def create_initializations(root, units, protocol):
    """Save actual post-constructor weights and RNG without protocol-hash cycles."""
    root = Path(root).resolve()
    assert root == NEW_ROOT.resolve()
    device_setup('cpu')
    results = {}
    constructor = Path(__file__).resolve().parents[2]/'external/neurips/src/neurips_permutations/models.py'
    adapter = Path(__file__).resolve().parents[2]/'experiments/longrun_attention.py'
    for unit in units:
        name, seed = unit['unit'], int(unit['model_seed'])
        spec = dict(unit=name, model_seed=seed, model=MODEL_CFG, vocabulary=TOKEN_TO_ID,
                    constructor_source_sha256=sha(constructor), attention_adapter_source_sha256=sha(adapter))
        spec_hash = payload_sha(spec)
        directory = root/'initializations'/name
        cp, rp, meta = directory/'step0.pt', directory/'starting_rng.pt', directory/'identity.json'
        if meta.exists():
            ref = json.loads(meta.read_text())
            assert ref['init_spec_sha256'] == spec_hash
            assert sha(cp) == ref['initial_checkpoint_sha256'] and sha(rp) == ref['starting_rng_sha256']
            saved = torch.load(cp, map_location='cpu', weights_only=False)
            net = build_model(protocol, seed)
            net.load_state_dict(saved['model'], strict=True)
            assert parameter_hash(net) == ref['initial_parameter_sha256']
            results[name] = ref
            continue
        assert not cp.exists() and not rp.exists(), 'Partial initial inventory needs explicit engineering review'
        net = build_model(protocol, seed)
        ph, state = parameter_hash(net), rng_state()
        saved = dict(model=net.state_dict(), rng=state, parameter_sha256=ph, model_config=MODEL_CFG,
            vocabulary=TOKEN_TO_ID, unit=name, model_seed=seed, init_spec_sha256=spec_hash,
            constructor_source_sha256=spec['constructor_source_sha256'], attention_adapter_source_sha256=spec['attention_adapter_source_sha256'],
            actual_step0=True, stage='Post constructor/weight tying; no optimizer or forward; CPU parameters',
            torch_version=str(torch.__version__), numpy_version=np.__version__)
        atomic_checkpoint(cp, saved)
        atomic_checkpoint(rp, dict(rng=state, rng_hash=rng_hash(state), unit=name, model_seed=seed,
            initial_parameter_sha256=ph, init_spec_sha256=spec_hash,
            stage='Same actual post-constructor stochastic state, restored after model/optimizer setup for every task'))
        ref = dict(unit=name, model_seed=seed, init_spec_sha256=spec_hash, initial_checkpoint_path=str(cp),
            initial_checkpoint_sha256=sha(cp), initial_parameter_sha256=ph, starting_rng_path=str(rp),
            starting_rng_sha256=sha(rp), starting_rng_hash=rng_hash(state), actual_initial_identity_verified=True,
            vocabulary_sha256=payload_sha(TOKEN_TO_ID), parameters=3240448, model_config=MODEL_CFG)
        json_write(meta, ref); results[name] = ref
        del net
    assert len(results) == len(units)
    assert len({v['initial_parameter_sha256'] for v in results.values()}) == len(units), 'Independent unit initializations accidentally equal'
    return results


def load_initialized_model(protocol, ref, device='cpu'):
    assert sha(ref['initial_checkpoint_path']) == ref['initial_checkpoint_sha256']
    saved = torch.load(ref['initial_checkpoint_path'], map_location='cpu', weights_only=False)
    assert saved['actual_step0'] and saved['model_config'] == MODEL_CFG and saved['vocabulary'] == TOKEN_TO_ID
    assert saved['init_spec_sha256'] == ref['init_spec_sha256'] and saved['parameter_sha256'] == ref['initial_parameter_sha256']
    net = build_model(protocol, ref['model_seed'], device)
    net.load_state_dict(saved['model'], strict=True)
    assert parameter_hash(net) == ref['initial_parameter_sha256']
    return net


def load_saved_model(protocol, checkpoint_path, device='cpu'):
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    net = build_model(protocol, int(saved.get('model_seed', 0)), device)
    net.load_state_dict(saved['model'], strict=True)
    net.eval()
    return net


def initial_rng(ref):
    assert sha(ref['starting_rng_path']) == ref['starting_rng_sha256']
    saved = torch.load(ref['starting_rng_path'], map_location='cpu', weights_only=False)
    assert saved['initial_parameter_sha256'] == ref['initial_parameter_sha256']
    assert rng_hash(saved['rng']) == saved['rng_hash'] == ref['starting_rng_hash']
    return saved['rng']
