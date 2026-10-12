"""Write a single bounded learnability plan before new results are observed."""
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from .common import ROOT, DIAG, CONFIG, WORKSPACE, SOURCE, DEV_TASKS, ARCHITECTURES
from experiments.permworld_encoding.core import sha, json_write, ENCODINGS


def setup():
    if CONFIG.exists():
        from .common import verify
        return verify()
    oldroot = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
    old = json.loads((oldroot / 'protocol.json').read_text())
    oldfreeze = json.loads((oldroot / 'freeze_manifest.json').read_text())
    for relative, digest in oldfreeze['source_sha256'].items():
        assert sha(WORKSPACE / relative) == digest
    model = copy.deepcopy(old['model']); model['dropout'] = 0.
    training = copy.deepcopy(old['training'])
    training.update(updates=8000, micro_batch=64, accumulation=1, effective_batch=64,
        learning_rate=.001, warmup_updates=400, validation_every=1000,
        checkpoint_every=1000,
        length_schedule='Uniform random permutations of the fixed n10,11,12 window; shared saved schedule.',
        batch_schedule='64 distinct train anchors per update, shared by all16 development models.',
        endpoint='Fixed8k endpoint, staged2k/4k/8k; no performance-dependent checkpoint selection.',
        execution='Serial jobs, global2h active-GPU wall-clock cap; safe partial checkpoints remain explicit.')
    unit = dict(unit='learningdev001', model_seed=96100001, data_seed=96200001,
                batch_seed=96300001, dropout_seed_base=96400000)
    config = dict(protocol_version='permworld-validation-only-learnability/v1',
        registered_utc=datetime.now(timezone.utc).isoformat(),
        client_date='2026-10-10', client_timezone='America/Los_Angeles',
        purpose='Exploratory learning development. Does not confirm any geometry hypothesis.',
        output=str(ROOT / 'worlds'), units=[unit], lengths=[10, 11, 12],
        tasks=list(DEV_TASKS), encodings=list(ENCODINGS), architectures=list(ARCHITECTURES),
        model=model, training=training,
        architecture_model_seeds={'transformer':96100001, 'gru':96100002},
        data=dict(examples_per_length={'train':2048, 'validation':1024, 'test':512},
            maximum_attempts_per_world=2000000,
            split_rule='Fresh seeded uniform permutations; reject historical inputs and within-world orbit overlap. No answer balancing.',
            training_actions=['identity'], validation_selection_actions=['identity'],
            test_policy='Reserved, not read by learning selector or this development driver.'),
        history=dict(index_path=str(oldroot / 'worlds/historical_input_hashes.npy'),
            index_sha256=sha(oldroot / 'worlds/historical_input_hashes.npy'),
            last_development_archive=str(oldroot / 'worlds/archdev001/data.npz'),
            last_development_archive_sha256=sha(oldroot / 'worlds/archdev001/data.npz'),
            rule='Append every last development orbit member to the verified complete historical index.'),
        development_choice=dict(candidate_count=1,
            length_choice='Contiguous first three old lengths chosen to reduce joint length/sequence burden, not because they previously succeeded.',
            changed_factors=['length window10..30 to10..12','train examples/length4096 to2048',
                            'learning rate3e-4 to1e-3','dropout.1 to0','microbatch16x4 to64x1',
                            'fixed updates20k to8k'],
            interpretation='Combined training-development change; cannot attribute any improvement to one factor.',
            no_CKA_or_Delta_configuration_selection=True),
        learning_gate=dict(primary='Identity validation full-vocabulary answer accuracy.',
            per_model_equal_length_accuracy_min=.90, per_length_accuracy_min=.85,
            per_model_gain_over_training_length_majority_min_pp=20.,
            paired_encoding_accuracy_gap_max_pp=5.,
            paired_encoding_per_length_accuracy_gap_max_pp=5.,
            per_model_eos_accuracy_min=.99,
            losses='Validation answer CE and combined answer+EOS CE fully reported; no extra loss threshold.',
            joint_acceptance='Every four tasks and both encodings within an architecture pass; report architectures separately and overall16 acceptance.',
            checkpoints='Only the completed fixed8k endpoint qualifies. Partial2k/4k checkpoints never substitute.'),
        budget=dict(active_gpu_wall_seconds=7200,
            finalization_reserve_seconds=30,
            starts='Immediately before first GPU engineering invocation; data/CPU-only preparation excluded.',
            deadline_rule='Stop updates30s before the absolute2h deadline; reserve that time for validation/checkpoint cleanup. No new GPU validation after deadline. Explicitly missing measurements cannot qualify.',
            additional_candidates=0, budget_extension=False),
        scope=dict(geometry_used_for_selection=False, geometry_evaluated_in_development=False,
            test_used_for_selection=False, F17_changes=False, old_artifact_changes=False,
            GroupWorld_started=False,
            continuation='Report completed development. If a setting qualifies, freeze its independent verification manifest; do not call this development result confirmation.'),
        initialization='One new actual saved step0 per architecture, cloned across four tasks and both encodings; no cross-architecture weight identity.',
        inference='One exploratory world/initializer per architecture; lengths/models do not count as independent worlds; no confirmation p-values.')
    config['training_manifest'] = [dict(run_id=f'{unit["unit"]}_{a}_{e}_{t}',
        unit=unit['unit'], architecture=a, encoding=e, task=t,
        model_seed=config['architecture_model_seeds'][a], data_seed=unit['data_seed'],
        batch_seed=unit['batch_seed'], updates=8000,
        stochastic_seed=unit['dropout_seed_base']+1000*ARCHITECTURES.index(a)+DEV_TASKS.index(t))
        for a in ARCHITECTURES for t in DEV_TASKS for e in ENCODINGS]
    DIAG.mkdir(parents=True, exist_ok=True); ROOT.mkdir(parents=True, exist_ok=True)
    protected = json.loads((WORKSPACE / 'diagnostics/permworld_layers_20261009/protected_f17_before.json').read_text())
    for path, digest in protected.items():
        assert sha(WORKSPACE / path) == digest
    for path in [*oldroot.rglob('*.json'), *oldroot.glob('*.csv'), oldroot / 'report.txt']:
        if path.is_file():
            protected[str(path)] = sha(path)
    for path in (WORKSPACE / 'diagnostics/permworld_layers_20261009').glob('*'):
        if path.is_file() and path.suffix in ('.csv', '.json', '.html', '.txt', '.py'):
            protected[str(path)] = sha(path)
    json_write(DIAG / 'protected_legacy_before.json', protected)
    json_write(CONFIG, config)
    hashes = dict(oldfreeze['source_sha256'])
    hashes.update({str(p.relative_to(WORKSPACE)):sha(p) for p in SOURCE.glob('*.py')})
    json_write(DIAG / 'freeze.json', dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(CONFIG), source_sha256=hashes,
        development_results_observed_before_freeze=False,
        historical_geometric_results_not_used_for_selection=True))
    json_write(ROOT / 'protocol.json', config)
    json_write(ROOT / 'training_manifest.json', config['training_manifest'])
    return config
