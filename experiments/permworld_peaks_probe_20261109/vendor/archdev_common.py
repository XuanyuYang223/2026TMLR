"""Independent development protocol; reuse frozen mathematics without mutation."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from experiments.permworld_encoding.core import (
    ACTIONS, ENCODINGS, RELATIONS, TASKS, WORKSPACE, TOKEN_TO_ID,
    batch, decode, encode, json_write, loss, parameter_hash, sha,
)
from experiments.permworld_encoding.train import atomic_checkpoint, rng_hash, rng_state, restore_rng
from experiments.longrun_attention import accelerate

SOURCE = Path(__file__).parent
ROOT = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
CONFIG = ROOT/'protocol.json'
DEV_TASKS = ('descents', 'recoils', 'peaks', 'valleys')
ARCHITECTURES = ('transformer', 'gru')
METRICS_PATH = WORKSPACE/'diagnostics/permworld_scope_inventory_20261009/metrics.py'
spec = importlib.util.spec_from_file_location('permworld_archdev_fixed_metrics', METRICS_PATH)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


class CausalGRU(nn.Module):
    """Four-layer causal GRU with the same token/padding/readout interface.

    Width and depth match, parameter count does not. Padding occurs only after
    each contiguous valid prefix and cannot affect an earlier recurrent state.
    """
    def __init__(self, config):
        super().__init__()
        self.config = SimpleNamespace(**config)
        d = config['d_model']
        self.token_embedding = nn.Embedding(len(TOKEN_TO_ID), d)
        self.position_embedding = nn.Embedding(config['max_seq_len'], d)
        self.embedding_dropout = nn.Dropout(config['dropout'])
        # Native multilayer cuDNN dropout has backend-owned RNG state outside
        # torch CPU/CUDA checkpoint RNG. Explicit torch.Dropout makes restart
        # reproducible while preserving four GRU cells and the same dropout rate.
        self.recurrent_layers = nn.ModuleList([
            nn.GRU(d, d, 1, batch_first=True, dropout=0.) for _ in range(config['layers'])])
        self.interlayer_dropout = nn.ModuleList([
            nn.Dropout(config['dropout']) for _ in range(config['layers']-1)])
        self.final_norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, len(TOKEN_TO_ID), bias=False)
        nn.init.normal_(self.token_embedding.weight, std=.02)
        nn.init.normal_(self.position_embedding.weight, std=.02)
        if config['tie_embeddings']:
            self.lm_head.weight = self.token_embedding.weight

    def forward(self, ids, mask=None):
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        positions = torch.arange(ids.shape[1], device=ids.device)
        h = self.token_embedding(ids) + self.position_embedding(positions)[None]
        h = self.embedding_dropout(h) * mask[..., None]
        for i, recurrent in enumerate(self.recurrent_layers):
            h, _ = recurrent(h)
            if i < len(self.interlayer_dropout):
                h = self.interlayer_dropout[i](h)
        h = self.final_norm(h) * mask[..., None]
        return self.lm_head(h)


def model(config, architecture, seed, device='cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if architecture == 'transformer':
        from experiments.permworld_encoding.core import make_model
        result = accelerate(make_model(config, seed))
    elif architecture == 'gru':
        result = CausalGRU(config['model'])
    else:
        raise ValueError(architecture)
    return result.to(device)


def device_setup(cpu=False):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    return torch.device('cpu' if cpu or not torch.cuda.is_available() else 'cuda')


def verify():
    config = json.loads(CONFIG.read_text())
    frozen = json.loads((ROOT/'freeze_manifest.json').read_text())
    assert sha(CONFIG) == frozen['protocol_sha256'], 'Development protocol changed after freeze'
    for p, h in frozen['source_sha256'].items():
        assert sha(WORKSPACE/p) == h, f'Development dependency changed: {p}'
    return config


def setup():
    if CONFIG.exists():
        return verify()
    ROOT.mkdir(parents=True, exist_ok=True)
    base = json.loads((WORKSPACE/'results/permworld_encoding_plan_20261009/protocol.json').read_text())
    unit = dict(unit='archdev001', model_seed=95100001, data_seed=95200001,
                batch_seed=95300001, dropout_seed_base=95400000)
    config = {
        'protocol_version': 'permworld-encoding-architecture-development/v1',
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Engineering and exploratory development, excluded from confirmation.',
        'output': str(ROOT/'worlds'), 'freeze_manifest': str(ROOT/'freeze_manifest.json'),
        'units': [unit], 'lengths': base['lengths'], 'data': base['data'],
        'historical_exclusions': base['historical_exclusions'],
        'model': base['model'], 'training': base['training'],
        'tasks': list(DEV_TASKS), 'architectures': list(ARCHITECTURES),
        'encodings': list(ENCODINGS),
        'relations': [r for r in RELATIONS if r['left'] in DEV_TASKS and r['right'] in DEV_TASKS],
        'architecture_model_seeds': {'transformer': 95100001, 'gru': 95100002},
        'architecture_control': '256-wide four-layer models; Transformer has eight heads. Parameter counts differ and are reported; no capacity-matched claim.',
        'initialization_control': 'One actual saved step-0 per architecture, cloned across all tasks and both encodings. Paired encodings share per-task starting RNG. Cross-architecture parameter identity is not claimed.',
        'evaluation': {
            'primary_state': 'final_norm at input end before task token',
            'fit_split': 'validation, orbit disjoint from source train and geometry test',
            'test_split': 'test, fixed512 anchors per length',
            'controls': ['raw_within_length', 'answer_strata'],
            'answer_strata': 'Joint [n,left identity answer,right identity/complement/inverse answers]; independently keep strata>=3 and subtract within-stratum means in fit/test. All conditions share masks.',
            'length_weights': {str(n): 1/len(base['lengths']) for n in base['lengths']},
            'metrics': ['linear_cka', 'knn_overlap', 'procrustes'],
            'sensitivity': ['debiased_cka'], 'knn_k': 10, 'procrustes_rank_cap': 8,
            'undefined_policy': 'Keep undefined cells; do not reweight incomplete length aggregates.',
            'baseline': 'Numeric0..30 conditional probabilities plus static true-answer onehot/scalar. Static gold has no fabricated initial or training delta.',
            'reported_components': ['B0', 'Bt', 'change_correct', 'change_wrong', 'Delta', 'Gamma0', 'Gammat', 'GammaDelta'],
        },
        'engineering': {
            'overfit_updates': 1000, 'overfit_samples': 48, 'overfit_answer_classes': 4,
            'overfit_answer_accuracy_min': .95, 'overfit_eos_accuracy_min': .99,
            'resume_from': 1000, 'resume_until': 2000,
            'parameter_tolerance': 1e-5, 'optimizer_tolerance': 1e-5, 'logit_tolerance': 1e-4,
            'source_validation_gain_over_initial_min': .10,
            'no_geometry_sign_or_pvalue_gate': True,
        },
        'inference': 'One exploratory data-world/source unit per architecture, no confirmation p-values or claims of statistical power.',
        'completion_rule': 'Each stage is explicit. Stop after this development report; never launch formal manifest, further seeds, GroupWorld or F17 work.',
        'new_world_exclusion': 'Historical input index excludes all old source inputs. Existing development and prior encoding preflight worlds additionally excluded as fixed archives.',
    }
    previous = WORKSPACE/'results/permworld_encoding_preflight_20261009/development/dev001/data.npz'
    if previous.exists():
        config['historical_exclusions']['archives'].append({'path': str(previous), 'sha256': sha(previous)})
    config['training_manifest'] = [dict(
        run_id=f'archdev001_{a}_{e}_{t}', unit=unit['unit'], architecture=a, encoding=e, task=t,
        model_seed=config['architecture_model_seeds'][a], data_seed=unit['data_seed'],
        batch_seed=unit['batch_seed'], updates=config['training']['updates'])
        for a in ARCHITECTURES for e in ENCODINGS for t in DEV_TASKS]
    json_write(CONFIG, config)
    dependencies = [*SOURCE.glob('*.py'),
        WORKSPACE/'experiments/permworld_encoding/core.py', WORKSPACE/'experiments/permworld_encoding/data.py',
        WORKSPACE/'experiments/permworld_encoding/train.py', WORKSPACE/'experiments/longrun_attention.py',
        WORKSPACE/'experiments/permutation_audit.py', METRICS_PATH,
        *[WORKSPACE/p for p in json.loads((WORKSPACE/'results/permworld_encoding_plan_20261009/freeze_manifest.json').read_text())['code_sha256']
          if p.startswith('external/neurips/')]]
    hashes = {str(p.relative_to(WORKSPACE)): sha(p) for p in dependencies}
    json_write(ROOT/'freeze_manifest.json', {
        'protocol_sha256': sha(CONFIG), 'config_sha256': sha(CONFIG),
        'source_sha256': hashes, 'code_sha256': hashes,
        'formal_training_started': False, 'new_source_training_started': False,
    })
    json_write(ROOT/'training_manifest.json', config['training_manifest'])
    return config


def world(config):
    return Path(config['output'])/config['units'][0]['unit']


def initial(config, architecture):
    directory = ROOT/'initializations'/architecture
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'step0.pt'
    metadata = directory/'step0.json'
    if path.exists():
        meta = json.loads(metadata.read_text())
        assert sha(path) == meta['checkpoint_sha256'] and meta['protocol_sha256'] == sha(CONFIG)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        return saved, meta
    seed = config['architecture_model_seeds'][architecture]
    net = model(config, architecture, seed)
    saved = dict(model=copy.deepcopy(net.state_dict()), rng=rng_state(), architecture=architecture,
        vocabulary=dict(TOKEN_TO_ID), model_config=config['model'], model_seed=seed,
        protocol_sha256=sha(CONFIG), parameter_sha256=parameter_hash(net),
        stage='actual_constructor_output_before_optimizer_or_training_forward',
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        source_sha256=json.loads((ROOT/'freeze_manifest.json').read_text())['source_sha256'])
    atomic_checkpoint(path, saved)
    meta = dict(checkpoint_sha256=sha(path), parameter_sha256=saved['parameter_sha256'],
        protocol_sha256=sha(CONFIG), architecture=architecture, actual_initial_identity_verified=True,
        parameters=sum(p.numel() for p in net.parameters()))
    json_write(metadata, meta)
    return saved, meta
