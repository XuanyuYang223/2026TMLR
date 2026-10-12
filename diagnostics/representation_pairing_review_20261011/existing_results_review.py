"""Independent arithmetic review of saved tables; standard library only.

No models, tensors, activation arrays, optimizers, metric implementations or
training modules are imported. This recomputes fixed-weight differences of
already saved similarity values, not any similarity metric.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parent / 'permworld_eight_relations_20261010'
LENGTHS = tuple(range(10, 31))
WEIGHT = 1 / 21
KEYS = ('architecture', 'encoding', 'relation', 'control', 'branch', 'layer', 'metric')
RAW = ('initial_correct', 'initial_wrong', 'initial_identity',
       'trained_correct', 'trained_wrong', 'trained_identity')
DIFFERENCES = {'B0': ('initial_correct', 'initial_wrong'),
               'Bt': ('trained_correct', 'trained_wrong'),
               'dc': ('trained_correct', 'initial_correct'),
               'dw': ('trained_wrong', 'initial_wrong'),
               'di': ('trained_identity', 'initial_identity')}
FIELDS = RAW + tuple(DIFFERENCES) + ('Delta',)
METRICS = ('linear_cka', 'knn_overlap', 'procrustes')
TOL = 1e-12


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def num(value):
    if value in ('', None):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def minus(a, b):
    return None if a is None or b is None else a - b


def mean(values):
    require(len(values) == 21, 'Fixed twenty-one length groups required')
    return None if any(v is None for v in values) else math.fsum(values) / 21


def sign(value):
    return '?' if value is None else '+' if value > 0 else '-' if value < 0 else '0'


def write_csv(name, values):
    require(values, 'Refuse an empty output: ' + name)
    fields = list(dict.fromkeys(k for row in values for k in row))
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(values)


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def run():
    begun = time.monotonic()
    require((OUT / 'protocol.json').is_file(), 'Root review protocol must exist before actual review')
    old = read(SOURCE / 'protocol.json')
    audit = read(SOURCE / 'metric_review/final_reader_audit.json')
    require(audit['status'] == 'passed', 'Original table audit is not passed')
    bindings = {}

    def bind(path, expected=None):
        path = Path(path); actual = sha(path)
        require(expected is None or actual == expected, 'Source identity changed: ' + str(path))
        bindings[str(path.resolve())] = actual
        return path

    bind(OUT / 'protocol.json'); bind(SOURCE / 'protocol.json')
    bind(SOURCE / 'metric_review/final_reader_audit.json')
    for name in ('per_length_results.csv', 'contrast_summary.csv',
                 'source_learning.csv', 'source_learning_summary.csv'):
        bind(SOURCE / name, audit['input_sha256'][name])

    relations = {r['pair_id']: r for r in old['relations']}
    require(len(relations) == 8 and old['lengths'] == list(LENGTHS), 'Original fixed scope changed')
    summaries = {}
    for r in rows(SOURCE / 'contrast_summary.csv'):
        if r['contrast_type'] == 'condition_summary':
            key = tuple(r[k] for k in KEYS)
            require(key not in summaries, 'Duplicate original summary')
            summaries[key] = r
    groups = defaultdict(dict); arithmetic_checks = 0; max_difference = 0.0
    for saved in rows(SOURCE / 'per_length_results.csv'):
        key = tuple(saved[k] for k in KEYS); n = int(saved['length'])
        require(n in LENGTHS and n not in groups[key], 'Invalid/duplicate length cell')
        values = {field: num(saved[field]) for field in RAW}
        for field, (left, right) in DIFFERENCES.items():
            values[field] = minus(values[left], values[right])
        values['Delta'] = minus(values['Bt'], values['B0'])
        for field in FIELDS:
            reference = num(saved[field]); actual = values[field]
            require((reference is None) == (actual is None), 'Saved undefined identity differs')
            if actual is not None:
                difference = abs(actual - reference); max_difference = max(max_difference, difference)
                require(difference <= TOL, 'Saved length arithmetic differs: ' + str(key) + '/' + field)
                arithmetic_checks += 1
        groups[key][n] = dict(length=n, **values, nan_reason=saved['nan_reason'])

    calculated = {}; boundaries = []; main_lengths = []; summary_checks = 0
    for key, length_rows in groups.items():
        require(tuple(sorted(length_rows)) == LENGTHS, 'Missing length is never silently reweighted')
        record = dict(zip(KEYS, key)); record['lengths'] = json.dumps(LENGTHS)
        record['length_count'] = 21; record['weight'] = 'fixed1/21; any undefined => undefined'
        for field in FIELDS:
            values = [length_rows[n][field] for n in LENGTHS]
            record[field] = mean(values)
            record[field + '_length_signs'] = ''.join(sign(v) for v in values)
            absent = [n for n in LENGTHS if length_rows[n][field] is None]
            record[field + '_undefined_lengths'] = json.dumps(absent)
            reference = num(summaries[key][field]); actual = record[field]
            require((reference is None) == (actual is None), 'Original aggregate undefined identity differs')
            if actual is not None:
                difference = abs(actual - reference); max_difference = max(max_difference, difference)
                require(difference <= TOL, 'Original fixed-weight aggregate differs')
                summary_checks += 1
            if absent:
                boundaries.append({k: record[k] for k in KEYS} | dict(
                    component=field, undefined_lengths=json.dumps(absent), undefined_count=len(absent),
                    aggregate_defined=False, weights_renormalized=False,
                    saved_reason_by_length=json.dumps({n: json.loads(length_rows[n]['nan_reason'] or '{}')
                                                       for n in absent}, sort_keys=True)))
        calculated[key] = record
        if key[3:] == ('answer_strata', 'hidden', 'final_norm', 'linear_cka'):
            main_lengths.extend(dict(zip(KEYS, key)) | {k:v for k,v in length_rows[n].items()
                                if k != 'nan_reason'} for n in LENGTHS)
    primary = [r for r in calculated.values() if tuple(r[k] for k in KEYS[3:]) ==
               ('answer_strata', 'hidden', 'final_norm', 'linear_cka')]
    require(len(primary) == 32 and len(main_lengths) == 672, 'Incomplete32 main grid')

    learning_groups = defaultdict(dict)
    for r in rows(SOURCE / 'source_learning.csv'):
        if r['condition'] == 'trained' and r['action'] == 'identity':
            key = (r['architecture'], r['encoding'], r['task'], r['split'])
            n = int(r['length']); require(n not in learning_groups[key], 'Duplicate source-learning length')
            learning_groups[key][n] = r
    learning = {}
    for key, r in learning_groups.items():
        require(tuple(sorted(r)) == LENGTHS, 'Source learning has incomplete lengths')
        learning[key] = {field: mean([num(r[n][field]) for n in LENGTHS]) for field in
                        ('answer_accuracy', 'answer_ce', 'length_majority_accuracy', 'accuracy_minus_length_mode')}
        learning[key]['minimum_length_accuracy'] = min(num(r[n]['answer_accuracy']) for n in LENGTHS)
        learning[key]['run_id'] = r[10]['run_id']
        learning[key]['majority_source'] = r[10]['majority_source']
    for r in rows(SOURCE / 'source_learning_summary.csv'):
        if r['condition'] == 'trained' and r['action'] == 'identity':
            rec = learning[r['architecture'], r['encoding'], r['task'], r['split']]
            for field in ('answer_accuracy', 'answer_ce', 'length_majority_accuracy',
                          'accuracy_minus_length_mode', 'minimum_length_accuracy'):
                a, b = rec[field], num(r[field])
                require((a is None) == (b is None) and (a is None or abs(a-b) <= TOL),
                        'Independent source learning summary differs')

    def attach_sources(record):
        rec = dict(record); relation = relations[rec['relation']]
        for side in ('left', 'right'):
            task = relation[side]; rec[side + '_task'] = task
            for split in ('validation', 'test'):
                source = learning[rec['architecture'], rec['encoding'], task, split]
                for field, value in source.items(): rec[f'{side}_{split}_{field}'] = value
        rec['learning_scope'] = 'identity; full188-vocabulary accuracy/CE; train-defined length mode'
        return rec

    primary = [attach_sources(r) for r in primary]
    negative = [r for r in primary if r['Bt'] is not None and r['Bt'] < 0]
    negative_lengths = []
    for rec in negative:
        key = tuple(rec[k] for k in KEYS)
        for n in LENGTHS:
            row = {k:rec[k] for k in KEYS} | {k:v for k,v in groups[key][n].items() if k != 'nan_reason'}
            for side in ('left', 'right'):
                task = relations[rec['relation']][side]; row[side + '_task'] = task
                for split in ('validation', 'test'):
                    source = learning_groups[rec['architecture'], rec['encoding'], task, split][n]
                    for field in ('answer_accuracy', 'answer_ce', 'length_majority_accuracy',
                                  'accuracy_minus_length_mode', 'training_majority_label'):
                        row[f'{side}_{split}_{field}'] = source[field]
            negative_lengths.append(row)

    direction_rows = []; disagreement = []; counts = {}
    for control in ('answer_strata', 'raw_within_length'):
        for field in ('Bt', 'Delta'):
            agreed = 0
            for arch in old['architectures']:
                for enc in old['encodings']:
                    for relation in relations:
                        rec = dict(architecture=arch, encoding=enc, relation=relation, control=control, component=field)
                        for metric in METRICS:
                            value = calculated[arch,enc,relation,control,'hidden','final_norm',metric][field]
                            rec[metric] = value; rec[metric + '_sign'] = sign(value)
                        signs = [rec[m + '_sign'] for m in METRICS]
                        rec['all_three_defined'] = '?' not in signs
                        rec['all_three_direction_agreement'] = rec['all_three_defined'] and len(set(signs)) == 1
                        direction_rows.append(rec)
                        if rec['all_three_direction_agreement']: agreed += 1
                        else: disagreement.append(rec)
            counts[control + '_' + field] = dict(agree=agreed, total=32)

    controls = []; layers = []
    for rec in primary:
        arch, enc, relation = rec['architecture'], rec['encoding'], rec['relation']
        raw = calculated[arch,enc,relation,'raw_within_length','hidden','final_norm','linear_cka']
        controls.append({k:rec[k] for k in KEYS[:3]} | {f'raw_{f}':raw[f] for f in ('B0','Bt','dc','dw','Delta')} |
                        {f'answer_{f}':rec[f] for f in ('B0','Bt','dc','dw','Delta')} |
                        {f'{f}_direction_flip':sign(raw[f]) != sign(rec[f]) for f in ('Bt','Delta')})
        fourth = 'block4' if arch == 'transformer' else 'layer4'
        reference = calculated[arch,enc,relation,'answer_strata','hidden',fourth,'linear_cka']
        for layer in (['block1','block2','block3','block4'] if arch == 'transformer' else
                      ['layer1','layer2','layer3','layer4']) + ['final_norm', 'embedding']:
            low = calculated[arch,enc,relation,'answer_strata','hidden',layer,'linear_cka']
            layers.append({k:rec[k] for k in KEYS[:3]} | dict(layer=layer, reference_layer=fourth,
                B0=low['B0'], Bt=low['Bt'], Delta=low['Delta'], reference_Bt=reference['Bt'],
                Bt_defined=low['Bt'] is not None,
                Bt_direction_flip_vs_fourth=None if low['Bt'] is None else sign(low['Bt']) != sign(reference['Bt'])))

    case = [r for r in primary if r['architecture'] == 'gru' and r['relation'] == 'exceedance_inverse']
    require(len(case) == 2, 'Two-encoding GRU exceedance case incomplete')
    c = {r['encoding']:r for r in case}; gamma = {field: c['canonical_cycle'][field] - c['one_line'][field]
        for field in ('B0','Bt','dc','dw','Delta')}
    gamma_names = dict(zip(('B0','Bt','dc','dw','Delta'),('Gamma0','Gammat','Gammac','Gammaw','GammaDelta')))
    for r in case: r.update({gamma_names[k]:v for k,v in gamma.items()})

    endpoint_path = bind(SOURCE / 'completed_checkpoint_manifest.json')
    graph_path = bind(SOURCE / 'final_review/graph_completed_checkpoint_manifest.json')
    final_path = bind(SOURCE / 'final_review/final_audit.json')
    manifest = read(endpoint_path); graph_manifest = read(graph_path); final = read(final_path)
    require(manifest['status'] == graph_manifest['status'] == final['status'] == 'passed', 'Actual cost manifests unverified')
    require(len(manifest['records']) == 64 and len(graph_manifest['records']) == 4, 'Actual68 receipts required')
    costs = []; lookup = {}
    for r in manifest['records'] + graph_manifest['records']:
        p = bind(r['completed_path'], r['completed_sha256']); receipt = read(p)
        require(receipt['step'] == 20000, 'Receipt lacks fixed20k')
        value = float(receipt['elapsed_seconds'])
        require(math.isfinite(value) and value >= 0 and abs(value-r['elapsed_process_wall_seconds']) <= TOL,
                'Receipt cost metadata differs')
        cost = dict(domain=r['domain'], run_id=r['run_id'], architecture=r['architecture'], encoding=r['encoding'],
            task=r['task'], reused_existing=r['reused_existing'], fixed_updates=20000,
            recorded_process_wall_seconds=value, hours=value/3600, completed_path=str(p),
            checkpoint_path=r['checkpoint_path'],
            checkpoint_apparent_bytes=Path(r['checkpoint_path']).stat().st_size,
            initial_checkpoint_path=r['initial_checkpoint_path'],
            initial_checkpoint_apparent_bytes=Path(r['initial_checkpoint_path']).stat().st_size,
            completed_sha256=r['completed_sha256'], scope='monotonic process wall including validation/IO; not GPU-active')
        costs.append(cost)
        if r['domain'] == 'permworld':lookup[r['architecture'],r['encoding'],r['task']] = cost
    cost_groups = []
    for domain, reused, arch in sorted({(r['domain'],r['reused_existing'],r['architecture']) for r in costs}):
        values = [r for r in costs if (r['domain'],r['reused_existing'],r['architecture']) == (domain,reused,arch)]
        cost_groups.append(dict(domain=domain,reused_existing=reused,architecture=arch,models=len(values),
            recorded_process_wall_seconds=math.fsum(r['recorded_process_wall_seconds'] for r in values),
            mean_seconds=math.fsum(r['recorded_process_wall_seconds'] for r in values)/len(values),
            minimum_seconds=min(r['recorded_process_wall_seconds'] for r in values),
            maximum_seconds=max(r['recorded_process_wall_seconds'] for r in values),
            scope='historical prior16' if reused else 'actual new52; process-wall, not end-to-end/GPU-active'))
    proposals = [
        ('P1','primary','gru','exceedance_inverse','检验近掌握取逆任务的编码Γ0/Γt/ΓΔ；不把Δ翻转称学习偏好翻转'),
        ('P2','primary','gru','record_complement','检验另一变换类型的编码分解，并保留轮换编码学习不足的限制'),
        ('P3','primary','transformer','fixed_complement','主动纳入轮换负Bt、正Δ和两源掌握不对称；近零负值不称显著反偏好')]
    proposal_rows = []; text = ['独立复现候选：仅提案，不执行', '',
        '这是看过既有结果后的探索性提案。三项主要比较P1/P2/P3全部预先纳入，含一个轮换负Bt条件；',
        '不能因新结果不理想转移主比较，也不能增加成功任务/长度。尚未确定确认单元数，不虚构方差、功效或精度。',
        '每个独立单元须同时更换数据世界和初始化；两个架构各保存真实step0，同架构内两编码克隆，',
        '共用底层数据/轨道、batch及20k预算。两个架构不声称同一初始化，现有容量仍未匹配。',
        '每比较为两任务×两编码=4模型。单元内任务/编码/关系/长度读数耦合，不视为4个独立重复。',
        '保存真实step0和RNG；事先固定源任务学习标准、全部21长度及三个指标。学习不足仍完整保留，不延长训练求正值。', '']
    for ident, role, arch, relation, reason in proposals:
        task = relations[relation]
        selected = [lookup[arch,enc,task[side]] for enc in ('one_line','canonical_cycle') for side in ('left','right')]
        seconds = math.fsum(r['recorded_process_wall_seconds'] for r in selected)
        checkpoint_paths = {r['checkpoint_path']:r['checkpoint_apparent_bytes'] for r in selected}
        proposal_rows.append(dict(candidate=ident,role=role,architecture=arch,relation=relation,
            left_task=task['left'],right_task=task['right'],models_per_unit=4,updates_per_unit=80000,
            observed_matching_models_process_wall_seconds=seconds,observed_hours=seconds/3600,
            observed_matching_checkpoint_unique_paths=len(checkpoint_paths),
            observed_matching_checkpoint_apparent_bytes=sum(checkpoint_paths.values()),
            source_run_ids=json.dumps([r['run_id'] for r in selected]),purpose=reason,
            scope='empirical matching20k receipt sum; excludes preparation/engineering/extraction/audit; not a precision estimate'))
        text += [f'{ident} ({role}) {arch} {relation}: {reason}',
                 f'每独立单元4模型/80000更新；既有同4配置的进程墙钟和 {seconds:.3f}秒 ({seconds/3600:.3f}小时)。',
                 '上述是相同配置历史进程时长参考；硬件争用、学习配置或世界改变会改变实际耗时。', '']
    selected_union = {r['run_id']:r for ident,role,arch,relation,reason in proposals
                      for enc in ('one_line','canonical_cycle') for side in ('left','right')
                      for r in [lookup[arch,enc,relations[relation][side]]]}
    unique_cp = {r['checkpoint_path']:r['checkpoint_apparent_bytes'] for r in selected_union.values()}
    unique_init = {r['initial_checkpoint_path']:r['initial_checkpoint_apparent_bytes'] for r in selected_union.values()}
    unit_seconds = math.fsum(r['recorded_process_wall_seconds'] for r in selected_union.values())
    scenarios = [dict(independent_units=units,unique_models_per_unit=len(selected_union),models=units*len(selected_union),
                     formal_updates=units*len(selected_union)*20000,
                     empirical_source_process_wall_seconds=units*unit_seconds,
                     empirical_source_process_wall_hours=units*unit_seconds/3600,
                     checkpoint_apparent_bytes_reference=units*sum(unique_cp.values()),
                     actual_initial_apparent_bytes_reference=units*sum(unique_init.values()),
                     cache_bytes_reference=None,accuracy_or_power_guarantee=False,
                     scope='Budget scenario only; measured apparent checkpoint bytes exclude new world/cache/engineering/logs')
                 for units in (1,2)]
    text += [f'三主比较合并后实际unique模型数={len(selected_union)}/独立单元，而非关系/编码格数；模型集合无重叠。',
             f'1单元/2单元仅为预算场景：{len(selected_union)}或{2*len(selected_union)}个模型；匹配历史源进程墙钟参考',
             f'{unit_seconds/3600:.3f}或{2*unit_seconds/3600:.3f}小时，不是GPU活跃时长或功效承诺。',
             f'现有匹配终点实际checkpoint表观字节和={sum(unique_cp.values())}，两个架构真实initial参考字节和={sum(unique_init.values())}。',
             '它们不包含新世界、工程模型、日志和重新提取cache；不是磁盘allocated blocks，也不是未来容量保证。',
             '未来cache可按冻结层/分支与批量存储方案控制，当前不猜测完整重训cache字节，不扩大为全768网格。',
             '旧16历史源累计成本与本轮48+4新增源成本分列；不能把receipt和queue/phase/driver墙钟相加。',
             'GPU-active/kernel时长未知；现有UTC/monotonic时钟偏移不转换成人为暂停或GPU空闲。',
             '本脚本不启动任何实验。用户若以后授权训练，需另外先冻结完整协议、范围、预算与停止边界。']

    outputs = [('recomputed_main32.csv',primary),('main32_by_length.csv',main_lengths),
        ('six_negative_bt.csv',negative),('six_negative_bt_by_length.csv',negative_lengths),
        ('three_metric_directions.csv',direction_rows),('metric_disagreements.csv',disagreement),
        ('gru_exceedance_encoding_case.csv',case),('control_sensitivity.csv',controls),
        ('layer_sensitivity.csv',layers),('undefined_component_boundaries.csv',boundaries),
        ('historical_receipt_costs.csv',costs),('historical_cost_groups.csv',cost_groups),
        ('replication_candidate_costs.csv',proposal_rows),('replication_budget_scenarios.csv',scenarios)]
    for name, values in outputs: write_csv(name, values)
    (OUT / 'replication_candidates.txt').write_text('\n'.join(text)+'\n')
    for path, expected in bindings.items():require(sha(path)==expected,'Read-only source changed during review')
    result = dict(status='passed',reviewed_utc=datetime.now(timezone.utc).isoformat(),
        source_path=str(Path(__file__).resolve()),source_sha256=sha(__file__),input_sha256=bindings,
        main_conditions=32,lengths=list(LENGTHS),weight='1/21 fixed; no undefined reweighting',
        primary_negative_Bt=len(negative),primary_Bt_signs=dict(Counter(sign(r['Bt']) for r in primary)),
        primary_Delta_signs=dict(Counter(sign(r['Delta']) for r in primary)),
        saved_length_arithmetic_checks=arithmetic_checks,saved_summary_arithmetic_checks=summary_checks,
        maximum_saved_value_difference=max_difference,absolute_tolerance=TOL,
        metric_direction_agreement=counts,metric_disagreement_records=len(disagreement),
        gru_exceedance_gamma={gamma_names[k]:v for k,v in gamma.items()},
        undefined_component_records=len(boundaries),independent_worlds=1,
        historical_receipts=68,historical_perm_receipts=64,new_formal_receipts=52,
        existing_final_audit_cost_reference=final['timing']['recorded_training_process_wall_seconds'],
        limitations=['Independent arithmetic is not an independent experiment or new similarity metric.',
            '32cells share8relations/2transform types/1world and saved initialization within architecture.',
            'Length mode is copied from existing train-defined source table, not newly estimated using test answers.',
            'High accuracy or answer-stratum centering does not establish a mathematical encoding mechanism.',
            'Main negativeBt is a point estimate; nearzero values are not significant negative effects.',
            'Control/metric/layer sensitivities and every undefined length remain visible.',
            'No variance/power/sample size/causal architecture or universal-domain claim.',
            'Receipt process wall is not active GPU time, source sums are not queue/phase end-to-end time.'],
        scientific_metrics_recomputed=0,arithmetic_of_saved_tables_recomputed=True,
        Torch_imported=False,models_constructed=0,optimizers_constructed=0,forward=0,training_updates=0,
        old_artifacts_modified=False,new_training_proposed_only=True,stopped=True,
        elapsed_CPU_review_wall_seconds=time.monotonic()-begun,
        output_sha256={n:sha(OUT/n) for n,_ in outputs}|{'replication_candidates.txt':sha(OUT/'replication_candidates.txt')})
    write_json('existing_results_review.json',result)
    print(json.dumps({k:result[k] for k in ('status','main_conditions','primary_negative_Bt','metric_direction_agreement','maximum_saved_value_difference','stopped')},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true',required=True)
    run()
