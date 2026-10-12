"""Read-only independent CSV/reader audit; never recalculate geometry metrics.

Only complete frozen analysis tables are accepted. Arithmetic uses a separate
Decimal equal-weight path, and original values are checked as binary64 values.
The sole writes are this directory's final_reader_audit.json/.txt.
"""
from __future__ import annotations

import csv
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct


HERE = Path(__file__).resolve().parent
DIAG = HERE.parent
WORKSPACE = DIAG.parents[1]
KEYS = ('architecture', 'encoding', 'relation', 'control', 'branch', 'layer', 'metric')
FIELDS = ('initial_correct', 'initial_wrong', 'initial_identity',
          'trained_correct', 'trained_wrong', 'trained_identity', 'B0', 'Bt', 'dc', 'dw', 'di', 'Delta')
TOL = 1e-10


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def js(path):
    return json.loads(Path(path).read_text())


def num(value):
    return float(value) if value not in ('', None, 'None') else float('nan')


def boolean(value):
    require(value in ('True', 'False', True, False), 'Unexpected boolean value')
    return value in ('True', True)


def same(observed, expected, context, exact=False):
    a, b = num(observed), num(expected)
    if not math.isfinite(a) or not math.isfinite(b):
        require(math.isnan(a) and math.isnan(b), 'Undefined mismatch: '+context)
    elif exact:
        require(struct.pack('!d', a) == struct.pack('!d', b), 'Original binary64 changed: '+context)
    else:
        require(abs(a-b) <= TOL, 'Arithmetic mismatch: '+context+f' observed={a!r} expected={b!r}')


def equal21(values):
    require(len(values) == 21, 'Cannot shorten a 21-length group')
    parsed = [num(x) for x in values]
    if not all(math.isfinite(x) for x in parsed):
        return float('nan')
    with localcontext() as context:
        context.prec = 60
        return float(sum(Decimal(str(x)) for x in parsed)/Decimal(21))


def key(row):
    return tuple(row[field] for field in KEYS)


def audit():
    required = ('protocol.json', 'freeze.json', 'per_length_results.csv', 'cross_arch_same_task.csv',
                'contrast_summary.csv', 'cross_arch_same_task_summary.csv', 'gold_baseline_length.csv')
    require(all((DIAG/name).is_file() for name in required), 'Wait for complete metric and contrast tables')
    protocol, frozen = js(DIAG/'protocol.json'), js(DIAG/'freeze.json')
    require(sha(DIAG/'protocol.json') == frozen['protocol_sha256'], 'Protocol freeze mismatch')
    for path, expected in frozen['source_sha256'].items():
        require(sha(path) == expected, 'Frozen executable source differs: '+path)
    analysis, actual = js(HERE/'analysis_audit.json'), js(HERE/'independent_metric_actual.json')
    require(analysis['status'] == actual['status'] == 'passed', 'Actual analysis/independent metric check incomplete')
    require(actual['checks_count'] == len(actual['records']) == 48, 'Actual48-check coverage changed')
    combinations = {(c, phase, role, m) for c in protocol['controls']
        for phase in ('initial', 'trained') for role in ('correct', 'wrong', 'identity')
        for m in ('linear_cka', 'debiased_cka', 'knn_overlap', 'procrustes')}
    require({(r['control'], r['condition'], r['role'], r['metric']) for r in actual['records']} == combinations,
            'Actual metric fixture omits a phase/control/pairing/metric')
    require(all(r['passed'] and r['tolerance'] == 1e-8 and r['fit_rows'] == r['eval_rows'] == 64
                and (r['absolute_error'] is None or r['absolute_error'] <= r['tolerance']) for r in actual['records']),
            'Actual numerical check violates frozen tolerance/rows')
    data = rows(DIAG/'per_length_results.csv')
    require(len(data) == 6048, 'Full per-length table must have6048 rows')
    expected = set()
    for a in protocol['architectures']:
        for e in protocol['encodings']:
            for relation in protocol['relations']:
                for c in protocol['controls']:
                    for n in range(10, 31):
                        for layer in protocol['layer_rules'][a]:
                            metrics = ('linear_cka', 'debiased_cka', 'knn_overlap', 'procrustes') if layer == 'final_norm' else ('linear_cka', 'debiased_cka')
                            expected.update((a, e, relation['pair_id'], c, 'hidden', layer, m, n) for m in metrics)
                        expected.update((a, e, relation['pair_id'], c, 'output_prob', 'output', m, n)
                                        for m in ('linear_cka', 'debiased_cka', 'knn_overlap', 'procrustes'))
    found = [key(row)+(int(row['length']),) for row in data]
    require(len(set(found)) == 6048 and set(found) == expected, 'Full relation/config/layer/control/metric/length coverage differs')
    groups = {}
    original_lookup = {tuple(row[k] for k in ('architecture', 'encoding', 'relation', 'control', 'branch', 'metric'))+(int(row['length']),): row
                       for row in rows(Path(protocol['source_root'])/'geometry_length.csv')}
    inherited_count = exact_values = identities = embeddings = decompositions = 0
    for row in data:
        groups.setdefault(key(row), []).append(row)
        for target, left, right in (('B0','initial_correct','initial_wrong'), ('Bt','trained_correct','trained_wrong'),
                                   ('dc','trained_correct','initial_correct'), ('dw','trained_wrong','initial_wrong'),
                                   ('di','trained_identity','initial_identity'), ('Delta','Bt','B0'), ('Delta','dc','dw')):
            same(row[target], num(row[left])-num(row[right]), 'per-row '+target)
            decompositions += 1
        reason = json.loads(row['nan_reason'])
        if row['layer'] == 'embedding':
            embeddings += 1
            require(all(math.isnan(num(row[field])) for field in FIELDS), 'Constant embedding got numeric geometry')
            require(len(reason) == 6 and all('structurally_constant' in value for value in reason.values()),
                    'Constant embedding undefined explanation missing')
        else:
            for phase in ('initial', 'trained'):
                for role in ('correct', 'wrong', 'identity'):
                    field = phase+'_'+role
                    if math.isnan(num(row[field])):
                        require(field in reason and bool(reason[field]), 'Undefined raw similarity lacks reason')
            if row['branch'] == 'hidden':
                identities += 1
                target = 0.0 if row['metric'] == 'procrustes' else 1.0
                same(row['initial_identity'], target, 'Actual same-initial hidden identity')
        if boolean(row['inherited_original_values']):
            inherited_count += 1
            oldbranch = 'prefix_final' if row['branch'] == 'hidden' else 'output_prob'
            old = original_lookup[tuple(row[k] for k in ('architecture', 'encoding', 'relation', 'control'))+
                                  (oldbranch, row['metric'], int(row['length']))]
            for field in FIELDS:
                oldfield = {'dc':'change_correct','dw':'change_wrong','di':'change_identity'}.get(field,field)
                same(row[field], old[oldfield], 'Inherited '+field, exact=True)
                exact_values += 1
            for phase in ('initial','trained'):
                same(row['error_'+phase+'_correct'], old[phase+'_correct_error'], 'Inherited correct raw error', exact=True)
                exact_values += 1
        for phase in ('initial','trained'):
            for role in ('correct','wrong','identity'):
                expected_error = -num(row[phase+'_'+role]) if row['metric']=='procrustes' else float('nan')
                same(row['error_'+phase+'_'+role], expected_error, 'Metric-specific rawerror')
    require(inherited_count==2688 and identities==4032 and embeddings==672, 'Source/identity/constant count differs')
    require(len(groups)==288, 'Condition summary group count differs')
    summaries, undefined = {}, 0
    for group_key, values in groups.items():
        require(sorted(int(row['length']) for row in values)==list(range(10,31)), 'Group lacks21 lengths')
        sums = {field:equal21([row[field] for row in values]) for field in FIELDS}
        sums['undefined_lengths']=sum(math.isnan(num(row['Delta'])) for row in values)
        undefined += sum(math.isnan(value) for field,value in sums.items() if field != 'undefined_lengths')
        summaries[group_key] = sums
        if group_key[-1] in ('linear_cka','debiased_cka','knn_overlap','procrustes'):
            variant_key=group_key[:-1]
            hashes=[row['raw_feature_array_hashes'] for row in sorted(values,key=lambda row:int(row['length']))]
            for alternative in ('linear_cka','debiased_cka','knn_overlap','procrustes'):
                other=groups.get(variant_key+(alternative,))
                if other is not None:
                    require(hashes==[row['raw_feature_array_hashes'] for row in sorted(other,key=lambda row:int(row['length']))],
                            'Different metric variants changed raw activations/row IDs')
    contrasts=rows(DIAG/'contrast_summary.csv')
    condition=[row for row in contrasts if row['contrast_type']=='condition_summary']
    require(len(condition)==288 and {key(row) for row in condition}==set(summaries),'Condition summary coverage differs')
    for row in condition:
        target=summaries[key(row)]
        for field in FIELDS:
            same(row[field],target[field],'Equal1/21 summary '+field)
        require(int(row['undefined_lengths'])==target['undefined_lengths'],'Undefined-length count differs')
        require(int(row['length_count'])==21 and '1/21' in row['weights'],'Missing fixed21 weighting metadata')
    contrast_counts={name:sum(row['contrast_type']==name for row in contrasts) for name in
                     ('condition_summary','encoding_difference','configuration_difference','layer_difference','interaction','metric_direction')}
    require(contrast_counts==dict(condition_summary=288,encoding_difference=144,configuration_difference=80,
                                layer_difference=160,interaction=40,metric_direction=16),'Contrast type counts differ')
    contrast_numeric=0
    for row in contrasts:
        if row['contrast_type'] in ('encoding_difference','configuration_difference','layer_difference'):
            left,right=json.loads(row['left_condition']),json.loads(row['right_condition'])
            leftkey,rightkey=key(left),key(right)
            for field in FIELDS:
                same(row[field],summaries[rightkey][field]-summaries[leftkey][field],'Controlled contrast '+field)
                contrast_numeric+=1
            if row['contrast_type']=='encoding_difference':
                for field,alias in (('B0','Gamma0'),('Bt','Gammat'),('dc','Gammac'),('dw','Gammaw'),('Delta','GammaDelta')):
                    same(row[alias],row[field],'Gamma alias '+alias)
        elif row['contrast_type']=='interaction':
            left,right=json.loads(row['left_condition']),json.loads(row['right_condition'])
            for field in FIELDS:
                def gamma(meta):
                    line=dict(meta,encoding='one_line');cycle=dict(meta,encoding='canonical_cycle')
                    return summaries[key(cycle)][field]-summaries[key(line)][field]
                same(row[field],gamma(right)-gamma(left),'Configuration×encoding interaction '+field)
                contrast_numeric+=1
            for field,alias in (('B0','I0'),('Bt','It'),('Delta','I')):
                same(row[alias],row[field],'Interaction alias '+alias)
        elif row['contrast_type']=='metric_direction':
            for field in ('B0','Bt','dc','dw','Delta'):
                directions=json.loads(row[field+'_directions'])
                expected_dirs={}
                for metric in ('linear_cka','knn_overlap','procrustes'):
                    meta=dict(row,metric=metric);value=summaries[key(meta)][field]
                    expected_dirs[metric]='undefined' if not math.isfinite(value) else '+' if value>0 else '-' if value<0 else '0'
                require(directions==expected_dirs,'Metric direction table differs')
                require(boolean(row[field+'_direction_consistent'])==(len(set(directions.values()))==1 and 'undefined' not in directions.values()),
                        'Cross-metric direction consistency differs')
    cross=rows(DIAG/'cross_arch_same_task.csv')
    expected_cross={(t,e,c,m,n) for t in protocol['tasks'] for e in protocol['encodings']
                    for c in protocol['controls'] for m in ('linear_cka','debiased_cka') for n in range(10,31)}
    require(len(cross)==672 and {(row['task'],row['encoding'],row['control'],row['metric'],int(row['length'])) for row in cross}==expected_cross,
            'Isolated same-task cross-architecture coverage differs')
    crossgroups={}
    for row in cross:
        same(row['similarity_change'],num(row['trained_similarity'])-num(row['initial_similarity']),'Cross-architecture change')
        require(row['pairing']=='identity--identity' and 'differs from relation-task' in row['centering_key'],
                'Cross-architecture grouping/pairing interpretation missing')
        crossgroups.setdefault(tuple(row[k] for k in ('task','encoding','control','metric')),[]).append(row)
    crosssummary=rows(DIAG/'cross_arch_same_task_summary.csv')
    require(len(crosssummary)==len(crossgroups)==32,'Cross-architecture summary count differs')
    for row in crosssummary:
        group=crossgroups[tuple(row[k] for k in ('task','encoding','control','metric'))]
        for dest,field in (('initial','initial_similarity'),('trained','trained_similarity'),('change','similarity_change')):
            same(row[dest],equal21([r[field] for r in group]),'Cross-architecture fixed21 mean')
    source_gold=Path(protocol['source_root'])/'gold_baseline_length.csv'
    require(sha(source_gold)==sha(DIAG/'gold_baseline_length.csv'),'Static gold baseline not byte-identical copy')
    main=[dict(zip(KEYS,group_key),**sums) for group_key,sums in summaries.items()
          if group_key[3]=='answer_strata' and group_key[4]=='hidden' and group_key[5]=='final_norm' and group_key[6]=='linear_cka']
    return dict(status='passed',auditor_sha256=sha(__file__),metric_values_recalculated=False,
        rows=6048,all_condition_groups_complete21=288,original_final_rows_reused=2688,
        original_numeric_binary64_comparisons=exact_values,initial_hidden_identity_checks=identities,
        constant_embedding_undefined_rows=embeddings,per_row_decomposition_comparisons=decompositions,
        contrast_counts=contrast_counts,contrast_numeric_comparisons=contrast_numeric,
        cross_architecture_rows=672,cross_architecture_fixed21_summaries=32,
        undefined_full21_component_aggregates_preserved=undefined,no_NaN_reweighting=True,
        actual_independent_metric_checks=48,actual_independent_metric_max_error=actual['maximum_absolute_error'],
        raw_activation_and_row_id_hashes_equal_across_metric_variants=True,gold_baseline_byte_copy=True,
        protocol_sha256=sha(DIAG/'protocol.json'),input_sha256={name:sha(DIAG/name) for name in required},
        main_final_CKA_answer_control_decomposition=main,one_world=True,no_confirmation_pvalues=True,
        trained=False,new_test_loaded=False,old_results_modified=False,F17_modified=False)


def main():
    try:
        result=audit()
    except Exception as error:
        result=dict(status='failed',error_type=type(error).__name__,error=str(error),trained=False,
                    metric_values_recalculated=False,new_test_loaded=False,old_results_modified=False,F17_modified=False)
        (HERE/'final_reader_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        (HERE/'final_reader_audit.txt').write_text('独立结果审查失败；原产物保留，不能把本次审查视为通过。\n'+str(error)+'\n')
        raise
    (HERE/'final_reader_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    text=['首轮20k16模型受控比较：独立只读结果审查通过。',
        '6048逐长度行完整覆盖两关系、四配置、两控制与全部预定层/指标；288组每组21长度，没有NaN重加权。',
        '2688原final/output行逐值binary64复用；静态答案基线原文件按字节复用。',
        '固定真实64行数值fixture48项通过；新中间层指标未重复计算，本审查只核对表格和冻结定义。',
        '初始化隐藏恒等配对4032项符合共享真实初始权重预期；672常量embedding行全保留未定义。',
        '所有B0/Bt/dc/dw/di/Delta、1/21汇总、编码/配置/层差与交互逐项核对通过。',
        'same-task跨架构672行/32汇总独立展示；它使用自己的共同答案key，不与关系任务对选择性混为一项效应。',
        '一个数据世界、每架构一个初始化；21长度和16模型不是独立复制。学习质量与容量差异仍限制机制解释。',
        '未训练、未读取新的test、未改变旧报告/F17；完成本审查后停止。',
        '',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)]
    (HERE/'final_reader_audit.txt').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:result[k] for k in ('status','rows','original_final_rows_reused','initial_hidden_identity_checks',
                                          'actual_independent_metric_max_error','cross_architecture_rows')}))


if __name__=='__main__':
    main()
