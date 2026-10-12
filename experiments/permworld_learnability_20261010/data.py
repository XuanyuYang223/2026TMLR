"""Fresh n10..12 world, excluding all prior development and source inputs."""
import json
from pathlib import Path

import numpy as np

from .common import ROOT, DIAG, CONFIG, verify, sha, json_write, initial, ARCHITECTURES
from experiments.permworld_encoding.core import orbit, fingerprint
from experiments.permworld_encoding.data import generate_world, contains


def prepare():
    config = verify()
    ROOT.mkdir(parents=True, exist_ok=True)
    previous = Path(config['history']['index_path'])
    assert sha(previous) == config['history']['index_sha256']
    archive = Path(config['history']['last_development_archive'])
    assert sha(archive) == config['history']['last_development_archive_sha256']
    target = ROOT / 'worlds' / 'historical_input_hashes.npy'
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = target.with_suffix('.json')
    if not target.exists():
        extras = []
        counts = {}
        with np.load(archive, allow_pickle=False) as data:
            for split in ('train', 'validation', 'test'):
                permutations = data[split + '_permutations']
                lengths = data[split + '_lengths']
                counts[split] = len(lengths)
                for row, length in zip(permutations, lengths):
                    p = tuple(map(int, row[:int(length)]))
                    members = orbit(p)
                    assert len(set(members)) == 8
                    extras.extend(fingerprint(q) for q in members)
        original = np.load(previous)
        combined = np.unique(np.concatenate((original, np.asarray(extras, dtype='V16'))))
        np.save(target, combined)
        assert contains(combined, extras).all()
        json_write(metadata, dict(sha256=sha(target), previous_index_sha256=sha(previous),
            appended_archive_sha256=sha(archive), prior_development_anchors=sum(counts.values()),
            prior_development_split_counts=counts, appended_orbit_members=len(extras),
            all_appended_orbit_members_verified=True, unique_inputs=len(combined),
            policy='Reject full candidate orbits if any member is historical; no historical inputs admitted.'))
    meta = json.loads(metadata.read_text())
    assert sha(target) == meta['sha256']
    assert meta['previous_index_sha256'] == sha(previous)
    assert meta['appended_archive_sha256'] == sha(archive)
    unit = config['units'][0]
    generate_world(config, unit, np.load(target, mmap_mode='r'), meta,
                   ROOT / 'worlds' / unit['unit'])
    directory = ROOT / 'worlds' / unit['unit']
    with np.load(directory / 'data.npz', allow_pickle=False) as data:
        keys = []
        for split in ('train', 'validation', 'test'):
            current = set()
            for row, n in zip(data[split + '_permutations'], data[split + '_lengths']):
                p = tuple(map(int, row[:int(n)]))
                current.add(min(fingerprint(q) for q in orbit(p)))
            assert len(current) == len(data[split + '_lengths'])
            assert not any(current & prior for prior in keys)
            keys.append(current)
    for architecture in ARCHITECTURES:
        initial(config, architecture)
    audit = json.loads((directory / 'data_audit.json').read_text())
    json_write(ROOT / 'data_ready.json', dict(status='passed', protocol_sha256=sha(CONFIG),
        data_sha256=audit['data_sha256'], prior_source_and_development_excluded=True,
        within_world_orbit_disjoint=True, paired_encodings_shared_data_and_batches=True,
        new_actual_initializations_saved=True, training_performed_by_this_stage=False,
        test_reserved_not_evaluated=True))
