"""Read-only provenance inventory of the first 16 fixed-20k specialists.

Loads saved weight states on CPU to hash them, never constructs a model or an
optimizer, never extracts activations or decodes evaluation input/label arrays.
Existing accuracy/baseline CSVs are inventoried as existing results, not rerun.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import torch


WORKSPACE = Path(__file__).resolve().parents[3]
ROOT = Path('/mnt/yangx-projects-f/20261009/permworld_new_artifacts/encoding_archdev')
EXPORT = WORKSPACE / 'diagnostics/permworld_layers_20261009/encoding_development'
OUTPUT = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, v in sorted(state.items()):
        a = v.detach().cpu().contiguous().numpy()
        for part in (name, str(a.dtype), str(a.shape)):
            h.update(part.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rng_hash(state):
    h = hashlib.sha256(); h.update(repr(state['python']).encode())
    h.update(state['numpy'][0].encode()); h.update(state['numpy'][1].tobytes())
    h.update(repr(state['numpy'][2:]).encode()); h.update(state['torch'].numpy().tobytes())
    for value in state['cuda']:
        h.update(value.cpu().numpy().tobytes())
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def npz_headers(path):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            with archive.open(name) as f:
                version = np.lib.format.read_magic(f)
                assert version in ((1, 0), (2, 0)), ('Unexpected NPY header version', version)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(f)
            result[name.removesuffix('.npy')] = dict(shape=list(shape), dtype=str(dtype), fortran=fortran)
    return result


def run():
    start = time.monotonic(); inputs = {}
    def register(path, expected=None):
        actual = sha(path)
        assert expected is None or expected == actual, ('Source hash mismatch', str(path))
        inputs[str(path)] = actual
        return actual
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    protocol_sha = register(ROOT / 'protocol.json')
    assert protocol['tasks'] == ['descents', 'recoils', 'peaks', 'valleys']
    assert protocol['lengths'] == list(range(10, 31)) and protocol['training']['updates'] == 20000
    assert protocol['model']['dropout'] == .1
    manifest = json.loads((ROOT / 'training_manifest.json').read_text())
    register(ROOT / 'training_manifest.json')
    assert len(manifest) == 16 and manifest == protocol['training_manifest']
    assert len({j['run_id'] for j in manifest}) == 16
    freeze = json.loads((ROOT / 'freeze_manifest.json').read_text())
    register(ROOT / 'freeze_manifest.json')
    assert freeze['protocol_sha256'] == protocol_sha
    current_source = {}
    for relative, expected in freeze['source_sha256'].items():
        current_source[relative] = register(WORKSPACE / relative, expected)
    source_version = canonical(current_source)
    world = ROOT / 'worlds/archdev001'
    data_audit = json.loads((world / 'data_audit.json').read_text())
    register(world / 'data_audit.json')
    register(world / 'data.npz', data_audit['data_sha256'])
    register(world / 'batch_indices.npy', data_audit['batch_indices_sha256'])
    register(world / 'batch_lengths.npy', data_audit['batch_lengths_sha256'])
    # Headers are shape/type metadata, not decoded permutation/label arrays.
    data_headers = npz_headers(world / 'data.npz')
    split_counts = {row['split']: row['anchors'] for row in data_audit['split_audit']}
    assert split_counts == dict(train=86016, validation=5376, test=10752)
    initials = {}
    vocabulary = None
    for architecture in protocol['architectures']:
        directory = ROOT / 'initializations' / architecture
        meta = json.loads((directory / 'step0.json').read_text())
        register(directory / 'step0.json')
        register(directory / 'step0.pt', meta['checkpoint_sha256'])
        cp = torch.load(directory / 'step0.pt', map_location='cpu', weights_only=False)
        assert state_hash(cp['model']) == meta['parameter_sha256'] == cp['parameter_sha256']
        assert cp['protocol_sha256'] == protocol_sha and meta['protocol_sha256'] == protocol_sha
        assert cp['model_config'] == protocol['model']
        assert meta['actual_initial_identity_verified'] is True
        if vocabulary is None:
            vocabulary = cp['vocabulary']
        assert vocabulary == cp['vocabulary'] and len(vocabulary) == 188
        assert all(vocabulary[f'{i:02d}'] == i for i in range(31))
        initial = dict(**meta, checkpoint_path=str(directory / 'step0.pt'),
            model_seed=cp['model_seed'], model_config=cp['model_config'],
            saved_stage=cp['stage'], vocabulary_sha256=canonical(vocabulary),
            initialization_identity_evidence='saved_actual_initial_weights_tensor_and_container_hash_verified',
            saved_initial_constructor_source_sha256=cp['source_sha256'])
        if architecture == 'gru':
            original_path = directory / 'step0_before_explicit_dropout_fix.pt'
            register(original_path, meta['original_initial_checkpoint_sha256'])
            original = torch.load(original_path, map_location='cpu', weights_only=False)
            repair = json.loads((ROOT / 'explicit_dropout_repair_log.json').read_text())
            register(ROOT / 'explicit_dropout_repair_log.json')
            mapping = repair['parameter_mapping']
            assert set(mapping) == set(original['model']) and set(mapping.values()) == set(cp['model'])
            assert all(torch.equal(original['model'][old], cp['model'][new]) for old, new in mapping.items())
            assert state_hash({old: cp['model'][new] for old, new in mapping.items()}) == meta['canonical_parameter_sha256']
            initial['migration_tensors_independently_verified'] = len(mapping)
            initial['original_checkpoint_path'] = str(original_path)
            initial['migration_is_not_constructor_fresh_initialization'] = True
            del original
        initials[architecture] = initial
        del cp
    write_json(OUTPUT / 'vocabulary.json', dict(vocabulary=vocabulary,
        vocabulary_sha256=canonical(vocabulary), numeric_answer_token_ids=list(range(31)),
        note='Read from both actual saved step0 containers and compared exactly.'))
    learning = read_csv(ROOT / 'source_learning.csv'); register(ROOT / 'source_learning.csv')
    assert len(learning) == 16
    by_run = {r['run_id']: r for r in learning}
    baseline_summary = read_csv(EXPORT / 'model_minus_length_baseline_summary_supplemental.csv')
    register(EXPORT / 'model_minus_length_baseline_summary_supplemental.csv')
    baseline_by_key = {(r['architecture'], r['encoding'], r['task'], r['condition'], r['action']): r for r in baseline_summary}
    accuracy_rows = read_csv(ROOT / 'task_accuracy_length.csv'); register(ROOT / 'task_accuracy_length.csv')
    assert len(accuracy_rows) == 2016
    register(EXPORT / 'length_baseline_accuracy_supplemental.csv')
    models, caches, completed_metadata, starts = [], [], {}, {}
    for job in manifest:
        completion_path = ROOT / 'models' / job['run_id'] / 'completed.json'
        completion = json.loads(completion_path.read_text()); register(completion_path)
        completed_metadata[job['run_id']] = completion
        path = completion_path.parent / 'checkpoint.pt'
        register(path, completion['checkpoint_sha256'])
        cp = torch.load(path, map_location='cpu', weights_only=False)
        assert cp['step'] == completion['step'] == job['updates'] == 20000
        assert cp['run_id'] == completion['run_id'] == job['run_id']
        assert cp['protocol_sha256'] == completion['protocol_sha256'] == protocol_sha
        assert cp['execution_source_version_sha256'] == completion['execution_source_version_sha256'] == source_version
        assert cp['data_sha256'] == completion['data_sha256'] == data_audit['data_sha256']
        initial = initials[job['architecture']]
        assert cp['initial_parameter_sha256'] == completion['initial_parameter_sha256'] == initial['parameter_sha256']
        assert cp['initial_checkpoint_sha256'] == initial['checkpoint_sha256']
        rng_key = job['architecture'], job['task']
        if rng_key not in starts:
            rng_path = ROOT / 'initializations' / job['architecture'] / f'start_rng_{job["task"]}.pt'
            register(rng_path)
            start_state = torch.load(rng_path, map_location='cpu', weights_only=False)
            assert rng_hash(start_state['rng']) == start_state['rng_hash']
            assert start_state['parameter_sha256'] == initial['parameter_sha256']
            starts[rng_key] = dict(path=str(rng_path), sha256=inputs[str(rng_path)],
                                  rng_hash=start_state['rng_hash'])
            del start_state
        assert cp['starting_rng_hash'] == starts[rng_key]['rng_hash']
        final_parameter = state_hash(cp['model'])
        assert final_parameter != initial['parameter_sha256']
        source = by_run[job['run_id']]
        assert abs(float(source['final_validation_accuracy']) - completion['curve'][-1]['answer_accuracy']) < 1e-12
        assert completion['curve'][-1]['samples'] == 5376
        baseline = baseline_by_key[job['architecture'], job['encoding'], job['task'], 'trained', 'identity']
        record = dict(**job, checkpoint_path=str(path), checkpoint_sha256=completion['checkpoint_sha256'],
            final_parameter_sha256=final_parameter, completed_metadata_path=str(completion_path),
            step0_path=initial['checkpoint_path'], step0_sha256=initial['checkpoint_sha256'],
            initial_parameter_sha256=initial['parameter_sha256'], vocabulary_sha256=canonical(vocabulary),
            starting_rng_hash=starts[rng_key]['rng_hash'], starting_rng_path=starts[rng_key]['path'],
            starting_rng_checkpoint_sha256=starts[rng_key]['sha256'],
            initialization_identity_evidence=initial['initialization_identity_evidence'],
            parameters=completion['parameters'], protocol_path=str(ROOT / 'protocol.json'),
            protocol_sha256=protocol_sha, execution_source_version_sha256=source_version,
            data_path=str(world / 'data.npz'), data_sha256=data_audit['data_sha256'],
            batch_schedule_path=str(world / 'batch_indices.npy'), batch_schedule_sha256=data_audit['batch_indices_sha256'],
            training_anchors=86016, exposures=completion['exposures'], supervised_tokens=completion['supervised_tokens'],
            final_validation_samples=5376, final_validation_answer_accuracy=float(source['final_validation_accuracy']),
            final_validation_eos_accuracy=completion['curve'][-1]['eos_accuracy'],
            existing_test_identity_answer_accuracy=float(baseline['full_vocab_accuracy']),
            existing_test_identity_numeric_accuracy=float(baseline['numeric_accuracy']),
            existing_test_identity_training_length_majority_accuracy=float(baseline['length_majority_accuracy']),
            existing_test_identity_gain_over_length_majority_pp=float(baseline['full_vocab_gain_over_length_majority_pp']),
            answer_ce_available=False, eos_ce_available=False, combined_fullset_ce_available=False,
            ce_missing_reason='Only full-vocab argmax predictions and numeric0..30 conditional probabilities saved; full-vocab logits/EOS CE not cached.',
            endpoint_rule='Fixed final20000; no best checkpoint selection',
            baseline_scope='Old frozen10752test anchors, identity action; baseline train-only per-length mode, fixed1/21; existing results reused, not recomputed.',
            source_accuracy_scope='Original validation5376 anchors, identity action, full-vocabulary argmax; final20k.',
            match_status='strict_within_architecture_encoding_and_budget_controls;cross_architecture_configuration_comparison')
        assert record['exposures'] == 1280000 and record['supervised_tokens'] == 2560000
        models.append(record)
        for condition in ('initial', 'trained'):
            for split in ('validation', 'test'):
                cache_path = ROOT / 'evaluation_cache' / f'{job["architecture"]}_{job["encoding"]}_{job["task"]}_{condition}_{split}.npz'
                metadata_path = cache_path.with_suffix('.json')
                metadata = json.loads(metadata_path.read_text()); register(metadata_path)
                register(cache_path, metadata['sha256'])
                signature = metadata['signature']
                assert signature['protocol_sha256'] == protocol_sha and signature['data_sha256'] == data_audit['data_sha256']
                assert signature['split'] == split
                expected_cp = initial['checkpoint_sha256'] if condition == 'initial' else completion['checkpoint_sha256']
                assert signature['checkpoint_sha256'] == expected_cp
                headers = npz_headers(cache_path)
                assert headers['prefix_final']['shape'] == [split_counts[split], 3, 256]
                assert headers['output_prob']['shape'] == [split_counts[split], 3, 31]
                assert headers['prediction']['shape'] == [split_counts[split], 3]
                caches.append(dict(run_id=job['run_id'], architecture=job['architecture'],
                    encoding=job['encoding'], task=job['task'], condition=condition, split=split,
                    cache_path=str(cache_path), cache_sha256=metadata['sha256'], bytes=cache_path.stat().st_size,
                    cache_metadata_path=str(metadata_path), signature=signature, array_headers=headers,
                    extraction='eval final_norm at prefix end before task token',
                    output_probability_definition='Conditional softmax restricted to valid numeric tokens0..30, not full-vocabulary probability',
                    new_cache_array_payload_read=False))
        del cp
    # Check pair control references, without inferring same initial geometry.
    for architecture in protocol['architectures']:
        for task in protocol['tasks']:
            pair = [m for m in models if m['architecture'] == architecture and m['task'] == task]
            assert len(pair) == 2 and all(pair[0][field] == pair[1][field] for field in (
                'initial_parameter_sha256', 'step0_sha256', 'vocabulary_sha256', 'data_sha256',
                'batch_schedule_sha256', 'starting_rng_hash', 'exposures', 'supervised_tokens', 'updates'))
    exported_matches = []
    for relative in ('protocol.json', 'training_manifest.json', 'source_learning.csv', 'task_accuracy_length.csv',
                     'worlds/archdev001/data_audit.json', 'freeze_manifest.json'):
        copied = EXPORT / 'artifacts' / relative
        assert register(copied) == inputs[str(ROOT / relative)]
        exported_matches.append(relative)
    write_csv(OUTPUT / 'model_inventory.csv', models)
    write_json(OUTPUT / 'model_inventory.json', models)
    write_json(OUTPUT / 'cache_inventory.json', caches)
    write_json(OUTPUT / 'sources.json', dict(
        status='passed', primary_root=str(ROOT), export_root=str(EXPORT),
        protocol_path=str(ROOT / 'protocol.json'), protocol_sha256=protocol_sha,
        protocol=protocol, model_config=protocol['model'], training_config=protocol['training'],
        training_manifest_path=str(ROOT / 'training_manifest.json'),
        training_manifest_sha256=inputs[str(ROOT / 'training_manifest.json')],
        source_freeze_path=str(ROOT / 'freeze_manifest.json'), execution_source_version_sha256=source_version,
        vocabulary=vocabulary, vocabulary_sha256=canonical(vocabulary),
        data_path=str(world / 'data.npz'), data_sha256=data_audit['data_sha256'], data_audit=data_audit,
        batch_schedule_path=str(world / 'batch_indices.npy'), batch_schedule_sha256=data_audit['batch_indices_sha256'],
        batch_lengths_path=str(world / 'batch_lengths.npy'), batch_lengths_sha256=data_audit['batch_lengths_sha256'],
        checkpoint_inventory=[dict(run_id=m['run_id'], architecture=m['architecture'], encoding=m['encoding'],
            task=m['task'], model_seed=m['model_seed'], checkpoint_path=m['checkpoint_path'],
            checkpoint_sha256=m['checkpoint_sha256'], parameter_sha256=m['final_parameter_sha256'],
            initial_checkpoint_path=m['step0_path'], initial_checkpoint_sha256=m['step0_sha256'],
            initial_parameter_sha256=m['initial_parameter_sha256'], vocabulary_sha256=m['vocabulary_sha256'],
            completed_path=m['completed_metadata_path'], completed_sha256=inputs[m['completed_metadata_path']],
            init_evidence=m['initialization_identity_evidence'], parameters=m['parameters'],
            exposures=m['exposures'], supervised_tokens=m['supervised_tokens'], updates=m['updates'],
            starting_rng_hash=m['starting_rng_hash'], starting_rng_path=m['starting_rng_path'],
            starting_rng_checkpoint_sha256=m['starting_rng_checkpoint_sha256']) for m in models],
        caches=[dict(run_id=c['run_id'], architecture=c['architecture'], encoding=c['encoding'], task=c['task'],
            condition=c['condition'], split=c['split'], path=c['cache_path'], sha256=c['cache_sha256'],
            metadata_path=c['cache_metadata_path'], signature=c['signature'], array_headers=c['array_headers']) for c in caches],
        initializations=initials, completed_metadata=completed_metadata,
        input_sha256=inputs, scope='First16/final20k/n10..30 only; no new test data decoding or fitting.'))
    write_json(OUTPUT / 'source_index.json', dict(
        status='passed', primary_cohort='First archdev00116 models/final20k/n10..30 only',
        protocol=protocol, protocol_path=str(ROOT / 'protocol.json'), protocol_sha256=protocol_sha,
        source_freeze_path=str(ROOT / 'freeze_manifest.json'), source_version_sha256=source_version,
        source_code_sha256=current_source, initializations=initials,
        data_audit=data_audit, data_header_metadata=data_headers, vocabulary_sha256=canonical(vocabulary),
        final_model_rows=16, existing_cache_files=64, existing_cache_bytes=sum(c['bytes'] for c in caches),
        artifact_export_matches=exported_matches, input_sha256=inputs,
        source_accuracy_files=[str(ROOT / 'source_learning.csv'), str(ROOT / 'task_accuracy_length.csv')],
        baseline_files=[str(EXPORT / 'length_baseline_accuracy_supplemental.csv'),
                        str(EXPORT / 'model_minus_length_baseline_summary_supplemental.csv')],
        evaluation_code_path=str(WORKSPACE / 'experiments/permworld_encoding_archdev_20261009/evaluation.py'),
        metrics_code_path=str(WORKSPACE / 'diagnostics/permworld_scope_inventory_20261009/metrics.py'),
        missing=['Full-vocabulary answerCE on frozen validation/test inputs', 'EOS CE on same frozen inputs',
                 'Intermediate transformer block1..4/GRU layer1..4 activations',
                 'Per-length validation accuracy CSV (existing curves have mixed-length validation only; caches have predictions)'],
        caveats=['Capacity3240448 vs1660416 is unmatched across architectures.',
                 'Same actual weight tensors do not imply equal initialization geometry across encodings.',
                 'GRU actual initial weights were key-migrated with21tensor bitwise preservation before any source optimization.',
                 'Original protocol batch text saysall32 but actual frozen development manifest is16.',
                 'Protocol serial execution prose is stale: engineering and disjoint parallel workers are separately logged; summed per-model time is not GPU elapsed hours.',
                 'Only one source world and one initialization per architecture; no confirmation/power inference.'],
        scope=dict(second_round_8k_read=False, training_started=False, optimizer_constructed=False,
                   model_constructed=False, new_test_label_or_input_payload_read=False,
                   new_test_forward=False, new_cache_activation_payload_read=False,
                   checkpoint_states_loaded_CPU_for_hash=True, old_artifacts_modified=False),
        inventory_elapsed_seconds=time.monotonic()-start))
    lines = ['首轮20k PermWorld 16模型来源与匹配审计',
        '状态：PASS；仅库存核查，没有训练或新测试评价。',
        '唯一队列：archdev001，下降/逆下降/峰/谷×一行/规范轮换×Transformer/GRU；全部20000更新，n10..30。',
        '模型、原始配置、step0、最终权重、词表与数据哈希以model_inventory.csv/json和source_index.json为准，不根据标题推测。',
        '固定项：同一底层world、标签、orbit split、batch_indices、有效batch64、1.28M样本曝光、2.56M答案+EOS监督token、20k固定末点；两编码每架构共享实际step0和词表188。',
        'GRU为4个单层GRU和3个显式dropout(.1)，实际初始权重迁移21tensor逐值保存；原PT与绑定记录均存在。Transformer为256d、4层、8头；参数3240448，GRU1660416，跨架构仅配置比较。',
        '已有源任务validationaccuracy覆盖全部16模型（5376身份输入）；已有test任务accuracy覆盖16×2condition×21长度×3action=2016单元。此处只复用旧CSV，不新读test标签/输入或做新前向。',
        '长度众数已有训练标签固定的补充结果；identity测试baseline和模型相对差已加入model_inventory。',
        '缺失答案CE/EOS CE/全数据combined CE：不能把末批loss或数字0..30条件概率当作这些量；需要按新协议用既有检查点与冻结评估输入前向补齐。',
        '已有64个 final_norm缓存共'+str(round(sum(c['bytes'] for c in caches)/2**30,3))+'GiB，全部SHA/signature/NPYheader核验通过；不用重复最终层前向。中间4层尚未保存。',
        '新增中间层工作量参考：16个trained＋4个可去重的architecture×encoding真实init，validation5376＋test10752 anchors，3actions，共967680 prefix实例；128batch、按长度分组约7560forward batches。GPU耗时必须实测小批后外推，本库存不虚构GPU小时。',
        '库存CPU实测耗时：'+str(round(time.monotonic()-start,3))+'秒（包含2个step0、16个CP参数哈希和64个cache容器哈希；不算GPU前向）。',
        '原batch文字all32与serial措辞继承旧规划，真实清单16、实际disjoint并行日志需作为成本来源；不影响已核验的样本/预算匹配，但不要拿模型elapsed之和当实际GPU小时。',
        '', '实际16清单与已有accuracy（val身份/旧test身份/baseline）：']
    for m in models:
        lines.append(f'{m["run_id"]}: val={m["final_validation_answer_accuracy"]*100:.3f}%,旧test={m["existing_test_identity_answer_accuracy"]*100:.3f}%,长度baseline={m["existing_test_identity_training_length_majority_accuracy"]*100:.3f}%,gain={m["existing_test_identity_gain_over_length_majority_pp"]:.3f}pp,CP={m["checkpoint_path"]}。')
    lines += ['', '本任务没有读取第二轮8k文件，不修改旧代码/结果/F17；只在新inventory子目录落盘。']
    (OUTPUT / 'match_audit.txt').write_text('\n'.join(lines) + '\n')
    return dict(status='passed', model_count=16, cache_count=64,
                inventory_elapsed_seconds=time.monotonic()-start, output=str(OUTPUT),
                missing_answer_ce=True, missing_intermediate_layers=True)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2, ensure_ascii=False))
