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
    def archive_row(path, category, use, reason, rebuild, action='保留原路径；仅逻辑归档'):
        archive.append(dict(path=str(path),category=category,use=use,
            dependency_evidence=reason,retain_reason=reason,rebuildable=rebuild,
            suggested_action=action,exists=path.exists(),protected_action='不删除、不移动、不改旧文档/发布物'))
    for path in sorted(dependency):
        if any(path.is_relative_to(root) for root in active_roots):
            archive_row(path,'1_当前PermWorld图论主线','新扩展/图论/统一分析实际源码',
                        'Existing current source file, inventoried by local import closure; its owner freezes code before execution.',
                        '由当前版本重建；保留源哈希和协议')
        else:
            archive_row(path,'2_当前主线旧依赖','实际冻结源码/权重/数据/词表/随机状态/审计或导入依赖',
                        'Bound by first20k source inventory/hash manifest or current local static import closure. Dependencies can overlap independent projects; keep original path.',
                        '源码/派生表可从版本重建；真实权重/RNG/唯一数据不得默认为可重建')
    for path in [DIRECTORY, WORKSPACE/'diagnostics/permworld_controlled_comparison_20261010',
                 WORKSPACE/'experiments/permworld_eight_relations_20261010',
                 graph_directory, graph_source]:
        archive_row(path,'1_当前PermWorld图论主线','当前库存、扩展、统一比较和图论小试入口',
                    'New user-authorized task plus controlled comparison protocol reference; graph paths independently read from its current common.py.',
                    '报告可由保留输入和冻结脚本重建；源终点不可用重新训练代替')
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
                archive_row(path,'1_当前PermWorld图论主线','新扩展/图论实现及分析',
                            'Current source/root binding; keep source code and corresponding protocol before execution.',
                            '源码可从版本重建，保留固定产物与审计')
            elif covered:
                archive_row(path,'2_当前主线旧依赖','含当前复现引用的旧文件；目录内可能兼有独立研究',
                            'At least one exact bound dependency lies below this path; do not move the directory as a whole.',
                            '混合；逐文件核对，不能整体删除')
            else:
                archive_row(path,'3_独立旧研究','未进入当前声明的冻结依赖闭包；保留历史/负结果与原路径',
                            'Not referenced by the currently enumerated fixed source bindings/static import closure. Dynamic references and final extension closure require owner review; this is provisional, never a deletion decision.',
                            '未知/混合；保护真实权重、数据、代码历史、结果、F17及发布物')
    for path in sorted(WORKSPACE.glob('*')):
        if path.name in ('experiments','results','diagnostics','paper','external','archive_index.csv','cleanup_candidates.csv','CURRENT_RESEARCH.txt','.git','.venv'):
            continue
        if path.is_dir():
            archive_row(path,'3_独立旧研究','原有项目或工作区基础设施/挂载/发布历史，未授权更改',
                        'Keep workspace infrastructure, package/release/unique evidence intact; do not infer independence from directory name alone.',
                        '未知；本轮不处理')
    cleanup=[]
    for path in sorted({*WORKSPACE.glob('experiments/**/__pycache__'),*WORKSPACE.glob('diagnostics/**/__pycache__'),*WORKSPACE.glob('.pytest_cache')}):
        files=[x for x in path.rglob('*') if x.is_file()]
        if any(x in dependency for x in files):continue
        archive_row(path,'4_可再生临时文件','Python字节码/pytest缓存',
                    'Known interpreter/test-generated files; no exact current dependency binding points into this cache. Scientific artifacts and logs excluded.',
                    '是；由对应源码/测试重建','仅清理候选；本轮未执行')
        cleanup.append(dict(path=str(path),kind='Python bytecode/test cache',
            bytes=sum(x.stat().st_size for x in files),rebuildable=True,
            dependency_status='not an exact bound scientific artifact',
            retain_reason='Do not execute deletion in this task; retain until owner checks active process use.',
            suggested_action='提出清理候选，不删除',deleted=False,moved=False))
    csv_write(WORKSPACE/'archive_index.csv',archive)
    csv_write(WORKSPACE/'cleanup_candidates.csv',cleanup)
    json_write(OUTPUT/'archive_dependency_edges.json',dict(static_edges=edges,
        exact_declared_dependencies=[str(x) for x in sorted(dependency)],
        dynamic_dependency_caveat='New extension/graph source paths may still be under implementation. Reconcile this logical index after final freeze without changing protected artifacts.',
        files_deleted=False,files_moved=False))
    training_seconds=sum(r['estimated_seconds_mean'] for r in estimate_rows if r['resource']=='new_source_training_process_wall')
    lines=[
        '当前研究入口（2026-10-10；库存和协议准备阶段草案）',
        '问题：数学对应是否改变跨任务表征相似性，这个观察受编码、模型配置、层和指标影响，能否在图论小试中观察到？',
        '',
        '已完成：diagnostics/permworld_controlled_comparison_20261010/reader_report.html。首轮16模型固定20k、两编码×两配置、两关系、真实保存step0。旧多层82模型只作历史参考；后续8k可学习性模型不并入主队列。',
        '当前扩展：diagnostics/permworld_eight_relations_20261010/。64个源模型目标中复用16个，剩余48个；固定同一world、batch、词表、真实step0与20k配方。',
        '缺口：model_gap_manifest.csv；只训练缺失项：training_manifest.csv；真实成本：cost_estimate.csv；来源机器索引：inventory/sources.json。',
        '库存复核命令（CPU，不启动训练）：.venv/bin/python diagnostics/permworld_eight_relations_20261010/inventory/inventory.py --run',
        '训练/工程/分析命令：由扩展负责人在新协议冻结后提供，只指向新缺失清单；不要执行旧一键大训练入口。本草案不编造尚未冻结的CLI。',
        f'以首轮真实20k日志外推48模型约{training_seconds/3600:.2f}累计模型进程小时；历史并行争用影响计时，不能称实际GPU小时或本轮必需壁钟。原始20k未保存显存峰值，新工程测量应另列。',
        '大产物位置：'+str(ROOT)+'；不复制世界、不删除唯一旧权重/原始数据腾空间。',
        '图论：diagnostics/graphworld_pilot_20261010/；experiments/graphworld_pilot_20261010/；大产物/mnt/yangx-projects-f/20261010/graphworld_pilot。一个参考Transformer/行序邻接编码、四源任务、四耦合关系，4个新终点上限；按6/7/8顶点分别算后固定1/3。不自动扩展图任务、群族、编码架构网格或确认种子。',
        '结果入口：原始正确/错误/恒等相似度 → 训练后Bt配对偏好 → B0、dc/dw及Delta辅助分解；负结果、弱源任务和NaN完整保留。',
        '历史/独立项目：见archive_index.csv。分类2含当前复现必需的旧代码/数据/检查点，不能整体搬走；分类3仍保留原路径与证据，F17论文/结果/release保持原样。',
        '临时文件：见cleanup_candidates.csv；本轮仅提出Python字节码/测试缓存候选，未删除、未移动、未改Git历史。',
        '停止边界：完成八关系统一比较和图论小试报告后停止；不按几何或任务成绩增加预算，不自动发布。',
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
