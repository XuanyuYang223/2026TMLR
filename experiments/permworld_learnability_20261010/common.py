"""New independent roots; historical frozen code and results are read-only."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.permworld_encoding.core import (
    WORKSPACE, TASKS, ENCODINGS, TOKEN_TO_ID, batch, loss, parameter_hash,
    sha, json_write,
)
from experiments.permworld_encoding.train import (
    atomic_checkpoint, rng_state, restore_rng, rng_hash,
)
from experiments.permworld_encoding_archdev_20261009.common import (
    model, device_setup,
)

SOURCE = Path(__file__).resolve().parent
DIAG = WORKSPACE / 'diagnostics/permworld_learnability_20261010'
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_learnability')
CONFIG = DIAG / 'protocol.json'
DEV_TASKS = ('descents', 'recoils', 'peaks', 'valleys')
ARCHITECTURES = ('transformer', 'gru')


def verify():
    config = json.loads(CONFIG.read_text())
    frozen = json.loads((DIAG / 'freeze.json').read_text())
    assert sha(CONFIG) == frozen['protocol_sha256'], 'New protocol changed after freeze'
    for relative, digest in frozen['source_sha256'].items():
        assert sha(WORKSPACE / relative) == digest, ('Frozen dependency changed', relative)
    assert config['tasks'] == list(DEV_TASKS)
    assert config['lengths'] == [10, 11, 12]
    assert config['training']['updates'] == 8000
    assert config['scope']['geometry_used_for_selection'] is False
    return config


def source_version():
    hashes = json.loads((DIAG / 'freeze.json').read_text())['source_sha256']
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def initial(config, architecture):
    directory = ROOT / 'initializations' / architecture
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'step0.pt'
    metadata = directory / 'step0.json'
    if path.exists():
        meta = json.loads(metadata.read_text())
        assert sha(path) == meta['checkpoint_sha256']
        assert meta['protocol_sha256'] == sha(CONFIG)
        saved = torch.load(path, map_location='cpu', weights_only=False)
        assert saved['parameter_sha256'] == meta['parameter_sha256']
        return saved, meta
    seed = config['architecture_model_seeds'][architecture]
    net = model(config, architecture, seed, 'cpu')
    saved = dict(model=copy.deepcopy(net.state_dict()), rng=rng_state(),
                 architecture=architecture, vocabulary=dict(TOKEN_TO_ID),
                 model_config=config['model'], model_seed=seed,
                 protocol_sha256=sha(CONFIG), parameter_sha256=parameter_hash(net),
                 stage='actual_constructor_output_before_training_or_optimizer',
                 torch_version=str(torch.__version__), source_version_sha256=source_version())
    atomic_checkpoint(path, saved)
    meta = dict(checkpoint_sha256=sha(path), parameter_sha256=saved['parameter_sha256'],
                protocol_sha256=sha(CONFIG), architecture=architecture,
                actual_initial_identity_verified=True, model_seed=seed,
                parameters=sum(p.numel() for p in net.parameters()))
    json_write(metadata, meta)
    return saved, meta


def load_data(config):
    directory = ROOT / 'worlds' / config['units'][0]['unit']
    audit = json.loads((directory / 'data_audit.json').read_text())
    assert sha(directory / 'data.npz') == audit['data_sha256']
    assert sha(directory / 'batch_indices.npy') == audit['batch_indices_sha256']
    with np.load(directory / 'data.npz', allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files
                if name.startswith(('train_', 'validation_'))}
    schedule = np.load(directory / 'batch_indices.npy', mmap_mode='r')
    assert schedule.shape == (8000, 64)
    return data, schedule, audit
