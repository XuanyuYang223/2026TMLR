"""Read-only inventory and bounded cost estimates for the eight-relation extension.

CPU checkpoint/hash audit only: no model, optimizer, forward, gradient, training,
data regeneration, deletion, movement, or modifications of original artifacts.
All new tasks bind to an actually saved per-architecture descents RNG stream.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import tomllib
import zipfile

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[3]
DIRECTORY = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = WORKSPACE / 'diagnostics/permworld_controlled_comparison_20261010'
SOURCE_ROOT = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations')
ARCHITECTURES = ('transformer', 'gru')
ENCODINGS = ('one_line', 'canonical_cycle')
OLD_TASKS = ('descents', 'recoils', 'peaks', 'valleys')
FIELDS = ('run_id', 'unit', 'architecture', 'encoding', 'task', 'model_seed',
          'data_seed', 'batch_seed', 'updates', 'status', 'checkpoint_path',
          'checkpoint_sha256', 'parameter_sha256', 'initial_checkpoint_path',
          'initial_checkpoint_sha256', 'initial_parameter_sha256',
          'starting_rng_path', 'starting_rng_checkpoint_sha256',
          'starting_rng_hash', 'starting_rng_source_task', 'starting_rng_policy',
          'data_path', 'data_sha256', 'batch_schedule_path',
          'batch_schedule_sha256', 'batch_lengths_path', 'batch_lengths_sha256',
          'vocabulary_sha256', 'original_protocol_sha256',
          'original_execution_source_version_sha256', 'exposures',
          'supervised_tokens', 'parameters', 'reuse_evidence', 'missing_reason')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        for part in (name, str(a.dtype), str(a.shape)):
            h.update(part.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def rng_hash(state):
    h = hashlib.sha256()
    h.update(repr(state['python']).encode())
    h.update(state['numpy'][0].encode())
    h.update(state['numpy'][1].tobytes())
    h.update(repr(state['numpy'][2:]).encode())
    h.update(state['torch'].cpu().numpy().tobytes())
    for x in state['cuda']:
        h.update(x.cpu().numpy().tobytes())
    return h.hexdigest()


def json_read(path):
    return json.loads(Path(path).read_text())


def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def csv_read(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def array_headers(path):
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, order, dtype = reader(f)
                out[name.removesuffix('.npy')] = dict(shape=list(shape), dtype=str(dtype), fortran=order)
    return out


def import_closure(paths):
    """Static local import evidence; dynamic dependencies remain explicit bindings."""
    pending = list(paths)
    seen = set()
    edges = []
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file() or path.suffix != '.py':
            continue
        seen.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            modules = [x.name for x in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
            for module in modules:
                target = WORKSPACE.joinpath(*module.split('.')).with_suffix('.py')
                if not target.is_file():
                    target = WORKSPACE.joinpath(*module.split('.'))/'__init__.py'
                if target.is_file():
                    edges.append(dict(source=str(path), dependency=str(target), evidence='static Python import'))
                    pending.append(target)
    return seen, edges


def run():
    started = time.monotonic()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    registered = {}
    def register(path, expected=None):
        value = sha(path)
        assert expected is None or value == expected, ('Changed dependency', str(path), value, expected)
        registered[str(path)] = value
        return value
    source_path = SOURCE_DIRECTORY/'inventory/sources.json'
    source = json_read(source_path)
    register(source_path)
    protocol = json_read(SOURCE_ROOT/'protocol.json')
    register(SOURCE_ROOT/'protocol.json', source['protocol_sha256'])
    assert protocol['training']['updates'] == 20000 and protocol['lengths'] == list(range(10, 31))
    assert protocol['tasks'] == list(OLD_TASKS) and len(source['checkpoint_inventory']) == 16
    freeze = json_read(SOURCE_ROOT/'freeze_manifest.json')
    register(SOURCE_ROOT/'freeze_manifest.json')
    for relative, expected in freeze['source_sha256'].items():
        register(WORKSPACE/relative, expected)
    assert canonical(freeze['source_sha256']) == source['execution_source_version_sha256']
    relations_path = WORKSPACE/'external/neurips/configs/property_task_geometry.toml'
    relations = tomllib.loads(relations_path.read_text())['relation_pairs']
    register(relations_path)
    assert len(relations) == 8
    tasks = sorted({r[s] for r in relations for s in ('left', 'right')})
    assert len(tasks) == 16
    old_lookup = {(j['architecture'], j['encoding'], j['task']): j for j in source['checkpoint_inventory']}
    initials, streams = {}, {}
    vocabulary = None
    for architecture in ARCHITECTURES:
        init_path = SOURCE_ROOT/'initializations'/architecture/'step0.pt'
        meta_path = init_path.with_suffix('.json')
        meta = json_read(meta_path)
        register(meta_path)
        register(init_path, meta['checkpoint_sha256'])
        cp = torch.load(init_path, map_location='cpu', weights_only=False)
        assert state_hash(cp['model']) == meta['parameter_sha256'] == cp['parameter_sha256']
        assert cp['protocol_sha256'] == source['protocol_sha256'] and cp['model_config'] == protocol['model']
        assert meta['actual_initial_identity_verified'] is True
        if vocabulary is None:
            vocabulary = cp['vocabulary']
        assert cp['vocabulary'] == vocabulary and canonical(vocabulary) == source['vocabulary_sha256']
        assert len(vocabulary) == 188 and all(vocabulary[f'{n:02d}'] == n for n in range(31))
        initials[architecture] = dict(**meta, checkpoint_path=str(init_path),
            saved_stage=cp['stage'], initial_parameter_sha256=meta['parameter_sha256'],
            original_saved_protocol_sha256=cp['protocol_sha256'], vocabulary_sha256=canonical(vocabulary))
        if architecture == 'gru':
            repair_path = SOURCE_ROOT/'explicit_dropout_repair_log.json'
            repair = json_read(repair_path)
            register(repair_path)
            original_path = init_path.parent/'step0_before_explicit_dropout_fix.pt'
            register(original_path, meta['original_initial_checkpoint_sha256'])
            original = torch.load(original_path, map_location='cpu', weights_only=False)
            mapping = repair['parameter_mapping']
            assert set(mapping) == set(original['model']) and set(mapping.values()) == set(cp['model'])
            assert all(torch.equal(original['model'][old], cp['model'][new]) for old, new in mapping.items())
            assert state_hash({old: cp['model'][new] for old, new in mapping.items()}) == meta['canonical_parameter_sha256']
            initials[architecture]['migration_tensors_verified'] = len(mapping)
            del original
        del cp
        for task in OLD_TASKS:
            rng_path = init_path.parent/f'start_rng_{task}.pt'
            register(rng_path)
            saved = torch.load(rng_path, map_location='cpu', weights_only=False)
            assert rng_hash(saved['rng']) == saved['rng_hash']
            assert saved['parameter_sha256'] == meta['parameter_sha256']
            streams[architecture, task] = dict(path=str(rng_path), sha256=registered[str(rng_path)],
                rng_hash=saved['rng_hash'], source_task=task,
                actual_saved_python_numpy_cpu_cuda_states_verified=True)
            del saved
    data_audit_path = SOURCE_ROOT/'worlds/archdev001/data_audit.json'
    audit = json_read(data_audit_path)
    register(data_audit_path)
    register(source['data_path'], audit['data_sha256'])
    register(source['batch_schedule_path'], audit['batch_indices_sha256'])
    register(source['batch_lengths_path'], audit['batch_lengths_sha256'])
    headers = array_headers(source['data_path'])
    assert headers['train_labels']['shape'] == [86016, 1, 16]
    assert headers['validation_labels']['shape'] == [5376, 3, 16]
    assert headers['test_labels']['shape'] == [10752, 3, 16]
    rng_policy = 'For each newly introduced task, bind the existing actual saved descents start RNG of the same architecture; paired encodings share that exact state. No old same-task RNG exists. All new tasks within an architecture share this stream (coupled). Original four tasks retain their original task-specific streams.'
    gaps, missing, inventory, costs = [], [], [], []
    for architecture in ARCHITECTURES:
        for encoding in ENCODINGS:
            for task in tasks:
                old = old_lookup.get((architecture, encoding, task))
                init = initials[architecture]
                stream = streams[architecture, task if old else 'descents']
                run_id = old['run_id'] if old else f'eightrel001_{architecture}_{encoding}_{task}'
                params = 3240448 if architecture == 'transformer' else 1660416
                rec = dict(run_id=run_id, unit='archdev001', architecture=architecture,
                    encoding=encoding, task=task, model_seed=protocol['architecture_model_seeds'][architecture],
                    data_seed=95200001, batch_seed=95300001, updates=20000,
                    status='completed_exact_reuse' if old else 'missing_authorized',
                    checkpoint_path=old['checkpoint_path'] if old else str(ROOT/'models'/run_id/'checkpoint.pt'),
                    checkpoint_sha256=old['checkpoint_sha256'] if old else '',
                    parameter_sha256=old['parameter_sha256'] if old else '',
                    initial_checkpoint_path=init['checkpoint_path'],
                    initial_checkpoint_sha256=init['checkpoint_sha256'],
                    initial_parameter_sha256=init['parameter_sha256'],
                    starting_rng_path=stream['path'], starting_rng_checkpoint_sha256=stream['sha256'],
                    starting_rng_hash=stream['rng_hash'], starting_rng_source_task=stream['source_task'],
                    starting_rng_policy='Retain original same-task actual saved RNG.' if old else rng_policy,
                    data_path=source['data_path'], data_sha256=source['data_sha256'],
                    batch_schedule_path=source['batch_schedule_path'], batch_schedule_sha256=source['batch_schedule_sha256'],
                    batch_lengths_path=source['batch_lengths_path'], batch_lengths_sha256=source['batch_lengths_sha256'],
                    vocabulary_sha256=source['vocabulary_sha256'], original_protocol_sha256=source['protocol_sha256'],
                    original_execution_source_version_sha256=source['execution_source_version_sha256'],
                    exposures=1280000, supervised_tokens=2560000, parameters=params,
                    reuse_evidence='Live container, parameter, initialization, RNG, data, batch and original frozen execution hashes verified.' if old else '',
                    missing_reason='' if old else 'No completed20k endpoint with exact original world, actual init, batches, vocabulary and architecture/encoding in the audited source registry.')
                if old:
                    completed = json_read(old['completed_path'])
                    register(old['completed_path'], old['completed_sha256'])
                    register(old['checkpoint_path'], old['checkpoint_sha256'])
                    cp = torch.load(old['checkpoint_path'], map_location='cpu', weights_only=False)
                    assert cp['step'] == completed['step'] == 20000
                    assert cp['data_sha256'] == source['data_sha256'] and cp['initial_parameter_sha256'] == init['parameter_sha256']
                    assert cp['starting_rng_hash'] == stream['rng_hash'] == old['starting_rng_hash']
                    assert cp['execution_source_version_sha256'] == source['execution_source_version_sha256']
                    assert state_hash(cp['model']) == old['parameter_sha256']
                    assert completed['exposures'] == 1280000 and completed['supervised_tokens'] == 2560000
                    inventory.append(old)
                    costs.append(dict(run_id=run_id, architecture=architecture, encoding=encoding,
                        task=task, steps=20000, model_process_elapsed_seconds=completed['elapsed_seconds'],
                        checkpoint_bytes=Path(old['checkpoint_path']).stat().st_size,
                        parameters=params, completion_path=old['completed_path'],
                        completion_sha256=registered[old['completed_path']],
                        timing_scope='Cumulative model training process wall, including validation/checkpoint IO and possible concurrent GPU contention. Not device-kernel time or total experiment wall.'))
                    del cp
                else:
                    assert not Path(rec['checkpoint_path']).exists(), 'Unexpected new endpoint: rerun inventory before overwriting manifest.'
                    missing.append(rec)
                gaps.append(rec)
    assert len(gaps) == 64 and len(inventory) == 16 and len(missing) == 48
    historical_path = WORKSPACE/'diagnostics/permworld_scope_inventory_20261009/training_inventory.csv'
    register(historical_path)
    history = csv_read(historical_path)
    rejections = []
    for r in history:
        if r['single_multi'] == 'single' and set(r['tasks'].split('|')) & set(tasks):
            reason = []
            if r['data_sha256'] != source['data_sha256']:
                reason.append('different or unverified data world/batch schedule')
            if r['original_initial_weight_identity_verified'] != 'True':
                reason.append('actual original initial weight identity not verified')
            if r['seed'] not in ('95100001', '95100002'):
                reason.append('different initialization seed coordinate')
            if r['training_steps'] != '20000':
                reason.append('different or incomplete update endpoint')
            assert reason, ('Unexpected possibly matching historical model', r['checkpoint'])
            rejections.append(dict(cohort=r['cohort'], task=r['tasks'], seed=r['seed'],
                checkpoint=r['checkpoint'], data_sha256=r['data_sha256'], training_steps=r['training_steps'],
                original_initial_weight_identity_verified=r['original_initial_weight_identity_verified'],
                status='historical_not_exact_match', reason='; '.join(reason)))
    csv_write(DIRECTORY/'model_gap_manifest.csv', gaps, FIELDS)
    csv_write(DIRECTORY/'training_manifest.csv', missing, FIELDS)
    csv_write(OUTPUT/'cost_source_runs.csv', costs)
    csv_write(OUTPUT/'historical_rejections.csv', rejections)
    estimate_rows = []
    for architecture in ARCHITECTURES:
        for encoding in ENCODINGS:
            ref = [r for r in costs if r['architecture']==architecture and r['encoding']==encoding]
            times = [r['model_process_elapsed_seconds'] for r in ref]
            estimate_rows.append(dict(resource='new_source_training_process_wall', architecture=architecture,
                encoding=encoding, missing_models=12, reference_models=4,
                per_model_seconds_mean=statistics.mean(times), per_model_seconds_median=statistics.median(times),
                per_model_seconds_min=min(times), per_model_seconds_max=max(times),
                estimated_seconds_mean=12*statistics.mean(times), estimated_seconds_min=12*min(times),
                estimated_seconds_max=12*max(times),
                estimated_bytes=12*max(r['checkpoint_bytes'] for r in ref),
                note='Serial workload extrapolation from actual fixed20k n10..30 logs; historical jobs overlapped so contention remains. Sum is an accumulated process-wall estimate, not actual GPU hours, device time, or promised elapsed completion.'))
    actual_cost = json_read(SOURCE_DIRECTORY/'cost_estimate.json')
    register(SOURCE_DIRECTORY/'cost_estimate.json')
    estimate_rows.append(dict(resource='incremental_multilayer_extraction_pipeline_wall', architecture='both', encoding='both',
        missing_models=48, reference_models=16, estimated_seconds_mean=actual_cost['actual_gpu_extraction_pipeline_wall_seconds']*96/40,
        estimated_bytes=math.ceil(actual_cost['actual_new_cache_bytes']*96/40),
        note='96 additional trained physical splits (48 trained x validation/test), versus previous40; prefix hidden states can reuse existing eight initial splits and prior32 trained splits where hashes match. Dense cache/IO estimate, not CUDA kernel time. Task-specific initial logits are NOT covered by shared initial prefix hidden states; see separate readout row.'))
    estimate_rows.append(dict(resource='additional_task_specific_initial_readout_forward_wall', architecture='both', encoding='both',
        missing_models=48, reference_models=16,
        estimated_seconds_mean=actual_cost['actual_gpu_extraction_pipeline_wall_seconds']*96/40,
        estimated_bytes=48*(5376+10752)*3*(31*4+4+4),
        note='Conservative full-forward equivalent for48 new task-specific initial readouts x2 splits, without storing duplicate shared initial hidden states. Initial output probabilities cannot be reused merely from identical pre-task hidden states. Numeric probabilities+prediction+answerCE storage arithmetic; exact evaluator strategy/IO may reduce cost.'))
    estimate_rows.append(dict(resource='all_eight_relations_CPU_metric_pipeline_wall', architecture='both', encoding='both',
        missing_models=48, reference_models=16, estimated_seconds_mean=actual_cost['actual_cpu_metric_pipeline_wall_seconds']*4,
        note='Eight versus two relations, fixed metric/length coverage. Linear extrapolation includes hashes/cacheIO/grouping; strata sizes may differ. If prior two relation tables can be exactly reused, additional workload roughly3x previous. Not CPU-core hours.'))
    for architecture, count in [('transformer',3240448),('gru',1660416)]:
        estimate_rows.append(dict(resource='parameter_gradient_Adam_FP32_bytes_lower_bound',architecture=architecture,
            encoding='both', estimated_bytes=16*count,
            note='4 bytes each for parameter, gradient, first moment, second moment; excludes activations, attention/cuDNN workspaces, embedding logits, optimizer temporary buffers, allocator and CUDA context. Not a VRAM peak estimate. No original20k peak-memory measurement was saved.'))
    csv_write(DIRECTORY/'cost_estimate.csv', estimate_rows)
    disk = shutil.disk_usage(ROOT.parent)
    source_json = dict(status='passed_read_only_inventory', generated_utc=datetime.now(timezone.utc).isoformat(),
        primary_root=str(SOURCE_ROOT), extension_root=str(ROOT), reference_inventory=str(source_path),
        protocol_path=str(SOURCE_ROOT/'protocol.json'), protocol_sha256=source['protocol_sha256'],
        original_execution_source_version_sha256=source['execution_source_version_sha256'],
        model_config=protocol['model'], training_config=protocol['training'],
        tasks=tasks, relations=relations, architecture_model_seeds=protocol['architecture_model_seeds'],
        vocabulary=vocabulary, vocabulary_sha256=source['vocabulary_sha256'],
        data_path=source['data_path'], data_sha256=source['data_sha256'], data_array_headers=headers,
        data_audit=audit, batch_schedule_path=source['batch_schedule_path'],
        batch_schedule_sha256=source['batch_schedule_sha256'], batch_lengths_path=source['batch_lengths_path'],
        batch_lengths_sha256=source['batch_lengths_sha256'],
        checkpoint_inventory=inventory, initializations=initials,
        saved_starting_rngs=[dict(architecture=a,task=t,**r) for (a,t),r in streams.items()],
        new_task_rng_policy=rng_policy, model_gap_manifest=gaps, training_manifest=missing,
        historical_candidates_rejected=len(rejections), historical_registry_rows=len(history),
        historical_search_scope='Audited scope inventory plus exact first16 current source inventory; not a claim about unknown unregistered models elsewhere.',
        cost_estimates=estimate_rows, original_20k_cost_sources=costs,
        peak_VRAM_measurement_available=False,
        storage=dict(mount=str(ROOT.parent), total_bytes=disk.total, free_bytes=disk.free,
            reused_world_bytes=Path(source['data_path']).stat().st_size, world_not_copied=True,
            projected_final_checkpoint_bytes=sum(r['estimated_bytes'] for r in estimate_rows if r['resource']=='new_source_training_process_wall'),
            projected_incremental_feature_bytes=next(r['estimated_bytes'] for r in estimate_rows if r['resource']=='incremental_multilayer_extraction_pipeline_wall')),
        source_sha256=registered, inventory_script_sha256=sha(__file__),
        original_stale_prose=['old protocol says all32 models although actual first manifest is16', 'old protocol says serial although disjoint historical workers ran concurrently; timing interpreted accordingly'],
        scope=dict(training_started=False, optimizer_constructed=False, forward=False,
            original_data_decoded=False, new_data_generated=False, original_artifacts_modified=False,
            F17_modified=False, files_deleted=False, files_moved=False))
    json_write(OUTPUT/'sources.json', source_json)
    # Logical archive only. The frozen current dependency set, not directory names,
    # distinguishes current code/data from independent historical work.
    dependency = {Path(x) if Path(x).is_absolute() else WORKSPACE/x for x in freeze['source_sha256']}
    dependency.update(Path(x) for x in registered)
    dependency.update(Path(r['checkpoint_path']) for r in inventory)
    dependency.update(Path(r['initial_checkpoint_path']) for r in gaps)
    new_source = WORKSPACE/'experiments/permworld_eight_relations_20261010'
    graph_source = WORKSPACE/'experiments/graphworld_pilot_20261010'
    graph_directory = WORKSPACE/'diagnostics/graphworld_pilot_20261010'
    active_roots = (new_source, DIRECTORY, graph_source, graph_directory)
    active_py = [p for directory in active_roots if directory.exists() for p in directory.rglob('*.py')]
    closure, edges = import_closure(active_py+list(dependency))
    dependency.update(closure)
    archive = []
    def archive_row(path, category, use, reason, rebuild, action='Keep original path; logical archival only'):
        archive.append(dict(path=str(path),category=category,use=use,
            dependency_evidence=reason,retain_reason=reason,rebuildable=rebuild,
            suggested_action=action,exists=path.exists(),protected_action='Do not delete, move, or modify earlier documents/published artifacts'))
    for path in sorted(dependency):
        if any(path.is_relative_to(root) for root in active_roots):
            archive_row(path,'1_current_PermWorld_graph_research','Actual source code for the new extension/graph pilot/unified analysis',
                        'Existing current source file, inventoried by local import closure; its owner freezes code before execution.',
                        'Rebuild from the current version; preserve source hashes and protocol')
        else:
            archive_row(path,'2_legacy_dependencies_of_current_research','Actual frozen source/weights/data/vocabulary/RNG state/audit or import dependencies',
                        'Bound by first20k source inventory/hash manifest or current local static import closure. Dependencies can overlap independent projects; keep original path.',
                        'Source/derived tables can be rebuilt from version history; genuine weights/RNG/unique data must not be assumed reconstructible')
    for path in [DIRECTORY, WORKSPACE/'diagnostics/permworld_controlled_comparison_20261010',
                 WORKSPACE/'experiments/permworld_eight_relations_20261010',
                 graph_directory, graph_source]:
        archive_row(path,'1_current_PermWorld_graph_research','Current inventory, extension, unified-comparison, and graph-pilot entrypoint',
                    'New user-authorized task plus controlled comparison protocol reference; graph paths independently read from its current common.py.',
                    'Reports can be rebuilt from retained inputs and frozen scripts; source endpoints must not be replaced by retraining')
    # Discover named source/result units and classify outside the explicit current
    # dependency graph. This is an index, not authorization to remove anything.
    for base in ('experiments','results','diagnostics','paper','external'):
        directory=WORKSPACE/base
        if not directory.exists():continue
        for path in sorted(directory.iterdir()):
            if path.name=='__pycache__' or path in dependency or path==DIRECTORY or path==SOURCE_DIRECTORY:
                continue
            covered=any(x.is_relative_to(path) for x in dependency)
            if path in active_roots:
                archive_row(path,'1_current_PermWorld_graph_research','New extension/graph implementation and analysis',
                            'Current source/root binding; keep source code and corresponding protocol before execution.',
                            'Source can be rebuilt from version history; retain fixed artifacts and audits')
            elif covered:
                archive_row(path,'2_legacy_dependencies_of_current_research','Contains earlier files referenced by current reproduction; the directory may also contain independent research',
                            'At least one exact bound dependency lies below this path; do not move the directory as a whole.',
                            'Mixed; inspect file by file, do not delete wholesale')
            else:
                archive_row(path,'3_independent_legacy_research','Outside the currently declared frozen dependency closure; retain historical/negative results and original paths',
                            'Not referenced by the currently enumerated fixed source bindings/static import closure. Dynamic references and final extension closure require owner review; this is provisional, never a deletion decision.',
                            'Unknown/mixed; protect genuine weights, data, code history, results, F17, and published artifacts')
    for path in sorted(WORKSPACE.glob('*')):
        if path.name in ('experiments','results','diagnostics','paper','external','archive_index.csv','cleanup_candidates.csv','CURRENT_RESEARCH.txt','.git','.venv'):
            continue
        if path.is_dir():
            archive_row(path,'3_independent_legacy_research','Existing projects or workspace infrastructure/mounts/release history, with no authorization to change them',
                        'Keep workspace infrastructure, package/release/unique evidence intact; do not infer independence from directory name alone.',
                        'Unknown; not handled this round')
    cleanup=[]
    for path in sorted({*WORKSPACE.glob('experiments/**/__pycache__'),*WORKSPACE.glob('diagnostics/**/__pycache__'),*WORKSPACE.glob('.pytest_cache')}):
        files=[x for x in path.rglob('*') if x.is_file()]
        if any(x in dependency for x in files):continue
        archive_row(path,'4_regenerable_temporary_files','Python bytecode/pytest cache',
                    'Known interpreter/test-generated files; no exact current dependency binding points into this cache. Scientific artifacts and logs excluded.',
                    'Yes; rebuild from the corresponding source/tests','Cleanup candidate only; not executed this round')
        cleanup.append(dict(path=str(path),kind='Python bytecode/test cache',
            bytes=sum(x.stat().st_size for x in files),rebuildable=True,
            dependency_status='not an exact bound scientific artifact',
            retain_reason='Do not execute deletion in this task; retain until owner checks active process use.',
            suggested_action='Propose cleanup candidates; do not delete',deleted=False,moved=False))
    csv_write(WORKSPACE/'archive_index.csv',archive)
    csv_write(WORKSPACE/'cleanup_candidates.csv',cleanup)
    json_write(OUTPUT/'archive_dependency_edges.json',dict(static_edges=edges,
        exact_declared_dependencies=[str(x) for x in sorted(dependency)],
        dynamic_dependency_caveat='New extension/graph source paths may still be under implementation. Reconcile this logical index after final freeze without changing protected artifacts.',
        files_deleted=False,files_moved=False))
    training_seconds=sum(r['estimated_seconds_mean'] for r in estimate_rows if r['resource']=='new_source_training_process_wall')
    lines=[
        'Current research entry (2026-10-10; draft at inventory/protocol preparation stage)',
        'Question: do mathematical correspondences change cross-task representational similarity, how does this depend on encoding, model configuration, layer, and metric, and is it observed in a graph pilot?',
        '',
        'Completed: diagnostics/permworld_controlled_comparison_20261010/reader_report.html. The first 16 models use fixed 20k endpoints, two encodings × two configurations, two relations, and genuinely saved step-0 weights. The earlier 82-model multilayer analysis is historical reference only; later 8k learnability models are excluded from the primary queue.',
        'Current extension: diagnostics/permworld_eight_relations_20261010/. Of 64 intended source models, 16 are reused and 48 remain; the same world, batches, vocabulary, genuine step-0 weights, and 20k recipe are fixed.',
        'Gaps: model_gap_manifest.csv; missing-only training list: training_manifest.csv; actual cost: cost_estimate.csv; machine-readable provenance index: inventory/sources.json.',
        'Inventory review command (CPU, no training): .venv/bin/python diagnostics/permworld_eight_relations_20261010/inventory/inventory.py --run',
        'Training/engineering/analysis commands: supplied by the extension owner after freezing the new protocol, pointing only to the new missing-model list. Do not execute an old one-command large-training entrypoint. This draft does not invent an unfrozen CLI.',
        f'Extrapolating the first round’s actual 20k logs gives approximately {training_seconds/3600:.2f} cumulative model-process hours for 48 models. Historical concurrent contention affects timing; this is neither actual GPU hours nor required wall time this round. The original 20k runs did not save peak GPU memory; new engineering measurements must be listed separately.',
        'Large-artifact location: '+str(ROOT)+'; do not duplicate the world or delete unique earlier weights/raw data to free space.',
        'Graph pilot: diagnostics/graphworld_pilot_20261010/; experiments/graphworld_pilot_20261010/; large artifacts at /mnt/yangx-projects-f/20261010/graphworld_pilot. One reference Transformer/row-order adjacency encoding, four source tasks, four dependent relations, and at most four new endpoints; compute separately for 6/7/8 vertices, then use fixed weight 1/3. No automatic expansion of graph tasks, group families, encoding/architecture grids, or confirmatory seeds.',
        'Result order: original correct/wrong/identity similarities → trained Bt pairing preference → auxiliary B0,dc/dw,Delta decomposition; retain all negative results, weak source tasks, and NaNs.',
        'Historical/independent projects: see archive_index.csv. Category 2 contains earlier code/data/checkpoints required for current reproduction and must not be moved wholesale. Category 3 retains original paths and evidence; F17 paper/results/releases remain unchanged.',
        'Temporary files: see cleanup_candidates.csv. This round only proposes Python-bytecode/test-cache cleanup candidates; nothing was deleted or moved, and Git history was unchanged.',
        'Stopping boundary: stop after the unified eight-relation comparison and graph-pilot reports; do not expand budgets based on geometry or task performance, and do not publish automatically.',
    ]
    (WORKSPACE/'CURRENT_RESEARCH.txt').write_text('\n'.join(lines)+'\n')
    elapsed=time.monotonic()-started
    summary=dict(status='passed', elapsed_seconds=elapsed, reusable_models=16, missing_models=48,
        target_source_models=64, relations=8, relation_configuration_conditions=32,
        historical_rejected_candidates=len(rejections), cumulative_process_wall_estimate_hours=training_seconds/3600,
        checkpoint_bytes_estimate=source_json['storage']['projected_final_checkpoint_bytes'],
        feature_bytes_estimate=source_json['storage']['projected_incremental_feature_bytes'],
        disk_free_bytes=disk.free, archive_rows=len(archive),cleanup_candidate_rows=len(cleanup),
        inventory_script_sha256=sha(__file__), no_original_artifact_writes=True,
        training_started=False, forward=False, files_deleted=False,files_moved=False)
    json_write(OUTPUT/'inventory_audit.json',summary)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',action='store_true',required=True)
    parser.parse_args()
    print(json.dumps(run(),ensure_ascii=False,indent=2))
