"""Completion overview from finished, exported PermWorld evidence only.

No torch, training, checkpoints, adaptive condition selection, or frozen-code
imports. --fixture builds synthetic tables in a temporary directory. The real
report refuses partial development outputs and checks their recorded hashes.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import statistics
import tempfile

ROOT = Path(__file__).resolve().parent
ARCHS = ('transformer', 'gru')
ENCODINGS = ('one_line', 'canonical_cycle')
TASKS = ('descents', 'recoils', 'peaks', 'valleys')
RELATIONS = ('descent_inverse', 'peak_complement')
CONTROLS = ('raw_within_length', 'answer_strata')
BRANCHES = ('prefix_final', 'output_prob')
METRICS = ('linear_cka', 'knn_overlap', 'procrustes', 'debiased_cka')
LENGTHS = tuple(range(10, 31))
COMPONENTS = ('B0', 'Bt', 'change_correct', 'change_wrong', 'Delta')
GAMMA_MAP = {'B0':'Gamma0', 'Bt':'Gammat', 'change_correct':'Gamma_correct',
             'change_wrong':'Gamma_wrong', 'Delta':'GammaDelta'}
GK = ('architecture', 'encoding', 'relation', 'control', 'branch', 'metric')
IK = ('architecture', 'relation', 'control', 'branch', 'metric')
PARAMS = {'transformer':3240448, 'gru':1660416}
OLD_NAMES = ('protocol.json', 'summary_audit.json', 'existing_final_audit.json', 'final_audit.json',
    'statistical_review.json', 'statistical_review.txt', 'relation_metric_summary.csv',
    'source_metric_summary.csv', 'length_metric_summary.csv', 'layer_metric_results.csv',
    'leave_one_relation_family.csv', 'report.html', 'reader_report.html')
DEV_NAMES = ('export_audit.json', 'report.html', 'model_minus_length_baseline_summary_supplemental.csv',
    'model_minus_length_baseline_supplemental.csv', 'length_baseline_protocol_supplemental.json')
ARTIFACT_NAMES = ('protocol.json', 'freeze_manifest.json', 'engineering_complete.json',
    'training_complete.json', 'evaluation_complete.json', 'independent_evaluation.json',
    'cpu_check.json', 'overfit_summary.json', 'resume_summary.json', 'data_ready.json',
    'source_learning.csv', 'geometry_summary.csv', 'geometry_length.csv',
    'encoding_interactions.csv', 'task_accuracy_length.csv', 'gold_baseline_length.csv',
    'initializations/execution_binding_v4.json', 'initializations/transformer/step0.json',
    'initializations/gru/step0.json', 'explicit_dropout_repair_log.json',
    'development_exclusion_final_audit.json')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def load_json(path): return json.loads(Path(path).read_text())


def load_csv(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def num(v):
    if v is None or v == '': return None
    z=float(v)
    if not math.isfinite(z): raise ValueError('Nonfinite metric is not an explicit undefined value')
    return z


def complete(values):
    return statistics.fmean(values) if values and all(v is not None for v in values) else None


def same(actual, expected, name, tolerance=1e-8):
    a=num(actual)
    if (a is None) != (expected is None) or (a is not None and abs(a-expected)>tolerance):
        raise ValueError(f'Inconsistent {name}: {actual!r} vs {expected!r}')


def key(row, fields): return tuple(row[k] for k in fields)


def require_keys(rows, fields, expected, name):
    actual={key(r,fields) for r in rows}
    if len(rows)!=len(expected) or actual!=expected:
        raise ValueError(f'Incomplete/duplicate {name}: {len(rows)} rows; expected {len(expected)}')


def missing_files(root, development):
    return [str(p) for p in ([root/n for n in OLD_NAMES]+[development/n for n in DEV_NAMES]
            +[development/'artifacts'/n for n in ARTIFACT_NAMES]) if not p.is_file()]


def validate(root, development, fixture=False):
    missing=missing_files(root,development)
    if missing: raise RuntimeError('Completion report withheld; missing completed artifacts: '+', '.join(missing))
    # No partial result table is loaded before all fixed required files exist.
    export=load_json(development/'export_audit.json')
    if export['status']!='passed' or bool(export['fixture_only'])!=bool(fixture):
        raise ValueError('Export is unfinished, invalid, or synthetic/real scope mismatched')
    if export['source_models']!=16 or export['fixed_final_updates']!=20000 or not export['all_four_conditions_two_relations_controls_metrics_present']:
        raise ValueError('Export does not describe the fixed complete development manifest')
    if not export['no_confirmatory_inference'] or not export['original_undefined_values_retained']:
        raise ValueError('Unexpected inference or missingness protocol')
    for record in export['source_small_artifacts']:
        path=development/record['export_relative']
        if not path.is_file() or sha(path)!=record['sha256']:
            raise ValueError('Export artifact/hash mismatch: '+str(path))
    verified_paths={r['export_relative'] for r in export['source_small_artifacts']}
    if not {'artifacts/'+n for n in ARTIFACT_NAMES} <= verified_paths:
        raise ValueError('A required development artifact has no export hash record')
    art=development/'artifacts'
    protocol=load_json(art/'protocol.json'); engineering=load_json(art/'engineering_complete.json')
    training=load_json(art/'training_complete.json'); evaluation=load_json(art/'evaluation_complete.json')
    if tuple(protocol['tasks'])!=TASKS or tuple(protocol['architectures'])!=ARCHS or tuple(protocol['lengths'])!=LENGTHS:
        raise ValueError('Changed task/architecture/length scope')
    if tuple(r['pair_id'] for r in protocol['relations'])!=RELATIONS or len(protocol['units'])!=1:
        raise ValueError('The development scope must contain one inverse, one complement, one world')
    jobs=protocol['training_manifest']
    require_keys(jobs, ('architecture','encoding','task'), {(a,e,t) for a in ARCHS for e in ENCODINGS for t in TASKS}, '16 training jobs')
    if protocol['training']['updates']!=20000 or training['models']!=16 or training['updates_per_model']!=20000:
        raise ValueError('Development final budget mismatch')
    if any(r.get('formal_training_started') is not False for r in (engineering,training,evaluation)):
        raise ValueError('Formal training was started; stop-boundary claim would be false')
    if engineering['models']!=16 or engineering['development_only'] is not True or engineering['status'] not in ('passed','failed'):
        raise ValueError('Missing valid completed engineering status')
    if engineering['stop_after_report'] is not True or engineering['no_confirmation_pvalues'] is not True:
        raise ValueError('Engineering completion does not preserve the authorized stop/inference scope')
    if training['protocol_sha256']!=sha(art/'protocol.json') or evaluation['protocol_sha256']!=sha(art/'protocol.json'):
        raise ValueError('Development protocol provenance mismatch')
    initial={a:load_json(art/'initializations'/a/'step0.json') for a in ARCHS}
    binding=load_json(art/'initializations/execution_binding_v4.json')
    repair=load_json(art/'explicit_dropout_repair_log.json')
    if not binding['gru']['original_parameter_values_bitwise_preserved'] or not repair['all_corresponding_tensors_bitwise_equal']:
        raise ValueError('Actual GRU step-0 migration was not value preserving')
    if not binding['transformer']['container_unchanged'] or not repair['head_embedding_tied']:
        raise ValueError('Initialization/tied-head proof missing')
    data_hashes=set(); versions=set()
    for job in jobs:
        path=art/'models'/job['run_id']/'completed.json'
        if not path.is_file(): raise RuntimeError('Missing fixed final model metadata: '+str(path))
        if 'artifacts/models/'+job['run_id']+'/completed.json' not in verified_paths:
            raise ValueError('Unhashed final-model metadata')
        done=load_json(path); init=initial[job['architecture']]
        if done['step']!=20000 or done['curve'][-1]['step']!=20000 or not done['actual_initial_identity_verified'] or not init['actual_initial_identity_verified']:
            raise ValueError('Final model lacks complete updates/actual step0 identity')
        if done['initial_parameter_sha256']!=init['parameter_sha256'] or done['parameters']!=PARAMS[job['architecture']]:
            raise ValueError('Model does not clone the actual architecture-specific initializer')
        if done['protocol_sha256']!=sha(art/'protocol.json'):
            raise ValueError('Completed model protocol mismatch')
        if any(done[k]!=job[k] for k in ('architecture','encoding','task','run_id')):
            raise ValueError('Final model manifest mismatch')
        data_hashes.add(done['data_sha256']); versions.add(done['execution_source_version_sha256'])
    if len(data_hashes)!=1 or versions!={binding['source_version_sha256']}:
        raise ValueError('The fixed models do not share a data/code version')
    exclusion=load_json(art/'development_exclusion_final_audit.json')
    if exclusion['status']!='passed' or exclusion['prior_new_orbit_intersection']!=0:
        raise ValueError('Development worlds overlap')
    summary=load_csv(art/'geometry_summary.csv'); interactions=load_csv(art/'encoding_interactions.csv')
    lengths=load_csv(art/'geometry_length.csv'); learning=load_csv(art/'source_learning.csv')
    accuracy=load_csv(art/'task_accuracy_length.csv')
    expected={(a,e,r,c,b,m) for a in ARCHS for e in ENCODINGS for r in RELATIONS for c in CONTROLS for b in BRANCHES for m in METRICS}
    require_keys(summary,GK,expected,'geometry summaries')
    require_keys(interactions,IK,{(a,r,c,b,m) for a in ARCHS for r in RELATIONS for c in CONTROLS for b in BRANCHES for m in METRICS},'encoding interactions')
    require_keys(lengths,GK+('length',),{k+(str(n),) for k in expected for n in LENGTHS},'geometry lengths')
    require_keys(learning,('architecture','encoding','task'),{(a,e,t) for a in ARCHS for e in ENCODINGS for t in TASKS},'source learning')
    require_keys(accuracy,('architecture','encoding','task','condition','length','action'),
        {(a,e,t,s,str(n),g) for a in ARCHS for e in ENCODINGS for t in TASKS for s in ('initial','trained') for n in LENGTHS for g in ('identity','complement','inverse')},'source accuracy lengths')
    idx={key(r,GK):r for r in summary}; groups=defaultdict(list)
    for row in lengths: groups[key(row,GK)].append(row)
    for k,items in groups.items():
        for field in COMPONENTS:
            same(idx[k][field],complete([num(r[field]) for r in items]),'1/21 '+field)
    for r in summary:
        values={f:num(r[f]) for f in COMPONENTS}
        same(r['Delta'],values['Bt']-values['B0'] if values['Bt'] is not None and values['B0'] is not None else None,'Delta=Bt-B0')
        same(r['Delta'],values['change_correct']-values['change_wrong'] if values['change_correct'] is not None and values['change_wrong'] is not None else None,'Delta=dc-dw')
    for r in interactions:
        a,rel,c,b,m=key(r,IK); line=idx[a,'one_line',rel,c,b,m]; cycle=idx[a,'canonical_cycle',rel,c,b,m]
        for f,gamma in GAMMA_MAP.items():
            x,y=num(cycle[f]),num(line[f]); same(r[gamma],x-y if x is not None and y is not None else None,gamma+' cycle-line')
    baseline=load_csv(development/'model_minus_length_baseline_summary_supplemental.csv')
    require_keys(baseline,('architecture','encoding','task','condition','action'),{(a,e,t,s,g) for a in ARCHS for e in ENCODINGS for t in TASKS for s in ('initial','trained') for g in ('identity','complement','inverse')},'baseline model summaries')
    baseline_note=load_json(development/'length_baseline_protocol_supplemental.json')
    if not baseline_note['post_protocol_descriptive'] or baseline_note['main_metric_or_gate_or_budget_changed']:
        raise ValueError('Unexpected length-baseline protocol change')
    old_audit=load_json(root/'summary_audit.json'); independent=load_json(root/'statistical_review.json')
    final=load_json(root/'final_audit.json'); old_protocol=load_json(root/'protocol.json')
    if old_audit['initialization_grade']!='C' or old_audit['original_initial_identity_verified'] is not False or old_protocol['new_source_training']!=0:
        raise ValueError('Historical initialization/source scope mismatch')
    if independent['status']!='PASS' or final['status']!='passed': raise ValueError('Final audits did not pass')
    if final['development']['models']!=16 or final['development']['formal_training_started'] is not False or final['development']['engineering_status']!=engineering['status']:
        raise ValueError('Combined final audit is incomplete or does not match the export')
    for name,digest in independent['sha256'].items():
        if name in ('relation_metric_summary.csv','source_metric_summary.csv','length_metric_summary.csv','layer_metric_results.csv','leave_one_relation_family.csv','protocol.json') and sha(root/name)!=digest:
            raise ValueError('Historical independent-review hash mismatch: '+name)
    old=load_csv(root/'relation_metric_summary.csv')
    oldidx={(r['cohort'],r['relation'],r['branch'],r['control'],r['metric']):r for r in old}
    for rel in ('descent_inverse','exceedance_inverse'):
        for branch in ('block_01_end','block_02_end','block_03_end','block_04_end','final_norm_end','final_norm_value_mean'):
            for metric in METRICS:
                if ('specialist16',rel,branch,'answer_strata',metric) not in oldidx:
                    raise ValueError('Missing fixed historical inverse layer/metric')
    return dict(export=export,protocol=protocol,engineering=engineering,training=training,evaluation=evaluation,
        summary=summary,interactions=interactions,learning=learning,accuracy=accuracy,baseline=baseline,
        baseline_note=baseline_note,old_audit=old_audit,independent=independent,final=final,old=old,oldidx=oldidx,
        binding=binding,initial=initial,data_sha256=next(iter(data_hashes)),version=next(iter(versions)))


def fmt(v):
    if v in (None,''): return '未定义'
    if isinstance(v,(bool,int)): return str(v)
    if isinstance(v,str) and v.lstrip('+-').isdigit(): return v
    try: return f'{float(v):+.5f}'
    except (ValueError,TypeError): return str(v)


def table(rows, columns):
    head=''.join('<th>'+html.escape(k)+'</th>' for k in columns)
    body=''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k)))+'</td>' for k in columns)+'</tr>' for r in rows)
    return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'


def plain_table(rows, columns):
    return '\n'.join(['\t'.join(columns)]+['\t'.join(fmt(r.get(k)) for k in columns) for r in rows])


def render(bundle,root,development,fixture=False):
    paragraphs=[]; sections=[]
    def prose(text): paragraphs.append(text); sections.append('<p>'+html.escape(text)+'</p>')
    def tab(rows, cols): paragraphs.append(plain_table(rows,cols)); sections.append(table(rows,cols))
    def title(text): paragraphs.append('\n'+text); sections.append('<h2>'+html.escape(text)+'</h2>')
    prose('用户授权边界为“多层分析＋16模型开发预跑，完成后停止”。本轮已完成已有 PermWorld 检查点的多层探索诊断，以及固定16模型的编码×架构工程开发实验。正式训练清单未启动，GroupWorld 未启动；F17论文、实验结果与发布产物保持原样。开发报告后停止，不自动追加训练。')
    if fixture: prose('合成 CPU fixture：本页仅验证报告生成流程，全部分数是人工数据，不是研究结果。')
    title('已有检查点：正增量与正确对齐增长须区分')
    prose('旧多层诊断没有新增源模型训练。历史初始化参照全部为 C 级：按原程序重建，但原真实 step-0 权重身份无法验证。结论只对应这些重建参照；不能将相同种子解释为精确匹配初始化。')
    oi=bundle['oldidx']; compact=[]
    for rel in ('descent_inverse','exceedance_inverse'):
        r=oi['specialist16',rel,'final_norm_end','answer_strata','linear_cka']
        compact.append(dict(relation=rel,sources=r['planned_sources'],B0=r['B0_mean'],Bt=r['Bt_mean'],dc=r['delta_correct_mean'],dw=r['delta_wrong_mean'],Delta=r['delta_mean']))
    tab(compact,('relation','sources','B0','Bt','dc','dw','Delta'))
    prose('两条取逆关系在旧主位置有稳定的正 Δ 候选，但 final end 普通 CKA 的大增量主要来自错误配对下降，而非正确配对大幅趋同。全部关系、源、长度、层和固定指标均保留；多层新增结果仍是探索性诊断。')
    profile=[]
    for rel in ('descent_inverse','exceedance_inverse'):
        for branch in ('block_01_end','block_02_end','block_03_end','block_04_end','final_norm_end','final_norm_value_mean'):
            r=oi['specialist16',rel,branch,'answer_strata','linear_cka']
            profile.append(dict(relation=rel,branch=branch,B0=r['B0_mean'],Bt=r['Bt_mean'],dc=r['delta_correct_mean'],dw=r['delta_wrong_mean'],Delta=r['delta_mean']))
    tab(profile,('relation','branch','B0','Bt','dc','dw','Delta'))
    a=oi['specialist16','descent_inverse','block_01_end','answer_strata','linear_cka']; b=oi['specialist16','descent_inverse','block_02_end','answer_strata','linear_cka']
    prose('下降—逆下降从 block1 到 block2：Δ 的层际变化为 '+fmt(num(b['delta_mean'])-num(a['delta_mean']))+'，Bt 的变化为 '+fmt(num(b['Bt_mean'])-num(a['Bt_mean']))+'，B0 的变化为 '+fmt(num(b['B0_mean'])-num(a['B0_mean']))+'。这次 Δ 跃升主要对应初始化偏好的层际变化，不能写成第二层训练后偏好更强。')
    prose('超越—亏缺在中层以及数值 token mean pooling 中，正确 CKA 的增长明显大于 final end。下降关系的正确变化 dc 也有指标方向差异。因此 final end 的拆解不能代替所有层或所有指标的结论。三主指标为 CKA、邻域重叠、留出 Procrustes；debiased CKA 为固定敏感性，不按结果挑层、长度或指标。')
    prose('均按10–30逐长度计算、固定1/21汇总；缺定义长度不重新加权。embedding结束位置及无序数值均值是同长度结构常量，指标未定义。源间离散程度可包含初始化、dropout、数据顺序和优化随机性，不能精确拆分，也不形成独立数据世界置信区间。共享任务/轨道的关系保留依赖结构，留一关系和留一家族是敏感性分析。')
    title('新开发：一条取逆＋一条补排列，四条件共16模型')
    prose('开发任务为下降数、逆下降数、峰、谷；关系仅 descent_inverse 与 peak_complement。四任务×一行/规范轮换两编码×Transformer/GRU两架构=16模型，新数据世界、每架构一个真实初始化单元。它不是原来两条取逆关系的新确认实验。')
    prose('每个架构内的两编码和四任务复制相同真实保存 step-0，跨架构不声称权重相同。GRU的原始真实初始化经过逐tensor保持数值的键迁移，使用四个单层GRU与显式层间Dropout；迁移和恢复检查的完整证据保留。两编码共底层排列、轨道划分、批次及20,000更新，保存固定最终检查点，不选择最佳阶段。')
    prose('Transformer 256维、4层、8头，共3,240,448参数；GRU 256维、4层，共1,660,416参数。容量和计算量未匹配。源任务学习程度不同、单世界与单初始化共同限制架构解释；不作确认性p值或等价性判断。')
    prose('工程完成状态：'+bundle['engineering']['status']+'。失败的门槛和弱任务同样完整保留，报告生成不把失败改为通过，也不延长固定预算。')
    primary=[r for r in bundle['interactions'] if r['branch']=='prefix_final' and r['control']=='answer_strata' and r['metric']=='linear_cka']
    primary.sort(key=lambda r:(ARCHS.index(r['architecture']),RELATIONS.index(r['relation'])))
    tab(primary,('architecture','relation','Gamma0','Gammat','Gamma_correct','Gamma_wrong','GammaDelta'))
    prose('上表只是事先固定的主读数：任务前最终层、答案分层、普通CKA。编码交互为规范轮换减一行：ΓΔ=Γt−Γ0，同时等于Γcorrect−Γwrong。必须区分初始化偏好改变、训练后偏好改变、正确增长和错误下降；Δ随编码变号不等于训练后关系偏好翻转。')
    title('全部源模型精度与长度参照')
    tab(bundle['learning'],('run_id','architecture','encoding','task','parameters','initial_validation_accuracy','final_validation_accuracy','gain','learning_gate_passed'))
    base=[r for r in bundle['baseline'] if r['condition']=='trained' and r['action']=='identity']
    base.sort(key=lambda r:(ARCHS.index(r['architecture']),ENCODINGS.index(r['encoding']),TASKS.index(r['task'])))
    tab(base,('architecture','encoding','task','full_vocab_accuracy','numeric_accuracy','length_majority_accuracy','full_vocab_gain_over_length_majority_pp'))
    prose('验证精度是完整词表的答案位置准确率；numeric_accuracy限制于数字词表，单独报告。相对初始化提升10个百分点只是工程学习门槛。长度众数基线在冻结协议后作为描述性补充：仅由各任务/长度的训练标签定众数，并列取最小标签，再固定用于全部测试action。它不改变主指标、门槛或训练预算。超过该基线排除了这一个长度预测器，低于它不证明完全没有使用排列内容。')
    title('全部预定分解和编码交互')
    prose('下表保留两分支（任务前隐藏、输出概率）、两控制、三主指标和debiased敏感性；不减去输出Δ来估算额外机制。答案分层不能去除全部置信度或预测误差结构。Procrustes分数为负归一化预测误差，分数增加表示留出误差降低。')
    for branch in BRANCHES:
        for control in CONTROLS:
            for metric in METRICS:
                selected=[r for r in bundle['summary'] if (r['branch'],r['control'],r['metric'])==(branch,control,metric)]
                selected.sort(key=lambda r:(ARCHS.index(r['architecture']),ENCODINGS.index(r['encoding']),RELATIONS.index(r['relation'])))
                gamma=[r for r in bundle['interactions'] if (r['branch'],r['control'],r['metric'])==(branch,control,metric)]
                gamma.sort(key=lambda r:(ARCHS.index(r['architecture']),RELATIONS.index(r['relation'])))
                label=branch+' / '+control+' / '+metric
                paragraphs.extend(['\n'+label,plain_table(selected,('architecture','encoding','relation','B0','Bt','change_correct','change_wrong','Delta','undefined_lengths')),plain_table(gamma,('architecture','relation','Gamma0','Gammat','Gamma_correct','Gamma_wrong','GammaDelta'))])
                sections.append('<details><summary>'+html.escape(label)+'</summary>'+table(selected,('architecture','encoding','relation','B0','Bt','change_correct','change_wrong','Delta','undefined_lengths'))+table(gamma,('architecture','relation','Gamma0','Gammat','Gamma_correct','Gamma_wrong','GammaDelta'))+'</details>')
    paragraphs.append('\n全部模型、初始/训练条件、三action的长度基线对照\n'+plain_table(bundle['baseline'],('architecture','encoding','task','condition','action','full_vocab_accuracy','numeric_accuracy','length_majority_accuracy','full_vocab_gain_over_length_majority_pp','numeric_gain_over_length_majority_pp')))
    sections.append('<details><summary>全部96个模型条件/action的长度基线对照</summary>'+table(bundle['baseline'],('architecture','encoding','task','condition','action','full_vocab_accuracy','numeric_accuracy','length_majority_accuracy','full_vocab_gain_over_length_majority_pp','numeric_gain_over_length_majority_pp'))+'</details>')
    title('复核范围、完整产物与停止')
    prose('旧汇总通过独立完整长度算术复核，旧52,080个单元逐字段保持一致。已有审计记录保护的F17文件未变。新开发检查包括真实初始化身份、独立进程恢复、保存激活与独立CKA路径、旧/新开发轨道排除。CKA、邻域和Procrustes都不能单独证明数学机制或功能价值。')
    audits=[dict(name='旧统计复核',status=bundle['independent']['status']),dict(name='旧完整性/F17审计',status=bundle['final']['status']),dict(name='开发工程汇总',status=bundle['engineering']['status'])]
    for name,status in bundle['engineering'].get('gates',{}).items(): audits.append(dict(name=name,status=status))
    tab(audits,('name','status'))
    links=[('reader_report.html','已有检查点面向读者的解释'),('report.html','多层完整报告'),('statistical_review.txt','独立统计复核正文'),('statistical_review.json','独立完整算术审计'),('final_audit.json','全实验完成与F17保护审计'),('existing_final_audit.json','旧完整性与F17保护审计'),('layer_metric_results.csv','旧全部逐关系/源/长度/层单元'),('source_metric_summary.csv','旧source汇总'),('relation_metric_summary.csv','旧关系汇总'),('length_metric_summary.csv','旧逐长度汇总'),('leave_one_relation_family.csv','旧关系与家族敏感性'),('encoding_development/report.html','编码×架构完整报告'),('encoding_development/export_audit.json','编码开发导出审计'),('encoding_development/artifacts/geometry_summary.csv','开发全部几何分解'),('encoding_development/artifacts/geometry_length.csv','开发全部逐长度几何'),('encoding_development/artifacts/encoding_interactions.csv','开发全部编码交互'),('encoding_development/artifacts/gold_baseline_length.csv','开发逐长度真实答案基线'),('encoding_development/model_minus_length_baseline_supplemental.csv','开发全部逐长度长度众数对照'),('encoding_development/length_baseline_protocol_supplemental.json','长度基线补充协议'),('encoding_development/artifacts/initializations/execution_binding_v4.json','开发源码与真实初始化身份绑定'),('encoding_development/artifacts/engineering_complete.json','开发全部工程门槛')]
    # Paths remain correct when the caller uses a different completed export.
    sections.append('<ul>')
    for path,label in links:
        target=(root/path) if not path.startswith('encoding_development/') else development/path.removeprefix('encoding_development/')
        relative=os.path.relpath(target,root)
        sections.append('<li><a href="'+html.escape(relative,quote=True)+'">'+html.escape(label)+'</a></li>')
        paragraphs.append(label+'：'+relative)
    sections.append('</ul>')
    prose('本轮到此停止。正式训练清单和GroupWorld没有启动；F17论文、结果及发布产物未改。任何后续确认、更多种子、不同系统或扩容都不由本报告自动触发。')
    page='<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PermWorld 本轮完成报告</title><style>body{font:16px system-ui,sans-serif;line-height:1.6;color:#263341;max-width:1200px;margin:30px auto;padding:0 18px}h1,h2{line-height:1.3}.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:12px;white-space:nowrap;width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eef2f5}details{margin:16px 0}summary{cursor:pointer;font-weight:600}a{color:#345e95}</style><h1>PermWorld 本轮完成报告</h1>\n'+'\n'.join(sections)+'</html>\n'
    return page,'PermWorld 本轮完成报告\n\n'+'\n\n'.join(paragraphs)+'\n'


def generate(root,development,fixture=False):
    bundle=validate(root,development,fixture)
    page,text=render(bundle,root,development,fixture)
    (root/'completion_report.html').write_text(page)
    (root/'completion_report.txt').write_text(text)
    evidence=dict(status='passed',fixture_only=fixture,created_utc=datetime.now(timezone.utc).isoformat(),
        read_completed_export_only=True,source_models=16,development_relations=list(RELATIONS),
        original_second_inverse_confirmed=False,geometry_summary_rows=128,interaction_rows=64,
        geometry_length_rows=2688,source_learning_rows=16,source_accuracy_length_rows=2016,
        baseline_model_condition_action_rows=96,all_fixed_conditions_retained=True,
        missing_lengths_reweighted=False,actual_initialization_within_architecture_verified=True,
        same_initializer_across_architectures_claimed=False,formal_training_started=False,
        GroupWorld_started=False,new_training_started_by_this_script=False,torch_or_GPU_imported=False,
        engineering_status=bundle['engineering']['status'],confirmatory_pvalues_generated=False,
        data_sha256=bundle['data_sha256'],source_version_sha256=bundle['version'],script_sha256=sha(__file__),
        input_sha256={str(p.relative_to(root)):sha(p) for p in [root/'summary_audit.json',root/'final_audit.json',root/'existing_final_audit.json',root/'statistical_review.json',root/'relation_metric_summary.csv',development/'export_audit.json'] if p.is_relative_to(root)},
        output_sha256={n:sha(root/n) for n in ['completion_report.html','completion_report.txt']})
    (root/'completion_report_audit.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    return evidence


def fixture():
    """Only synthetic CPU data; no real development output is inspected."""
    with tempfile.TemporaryDirectory(prefix='permworld_overview_fixture_') as temp:
        root=Path(temp);dev=root/'encoding_development';art=dev/'artifacts';art.mkdir(parents=True)
        def js(path,data):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        for name in OLD_NAMES:
            path=root/name
            if path.suffix=='.json':js(path,{})
            elif path.suffix=='.csv':write_csv(path,[{'synthetic':'only'}])
            else:path.write_text('Synthetic fixture placeholder only\n')
        old=[]
        for relation in ('descent_inverse','exceedance_inverse'):
            for branch in ('block_01_end','block_02_end','block_03_end','block_04_end','final_norm_end','final_norm_value_mean'):
                for metric in METRICS:
                    B0=-.05 if branch=='block_01_end' else -.5
                    Bt=.04 if branch=='block_01_end' else .03 if branch=='block_02_end' else .09
                    dc=-.03;delta=Bt-B0;dw=dc-delta
                    old.append(dict(cohort='specialist16',relation=relation,branch=branch,control='answer_strata',metric=metric,
                        planned_sources=3,B0_mean=B0,Bt_mean=Bt,delta_correct_mean=dc,delta_wrong_mean=dw,delta_mean=delta))
        write_csv(root/'relation_metric_summary.csv',old)
        js(root/'protocol.json',{'new_source_training':0})
        js(root/'summary_audit.json',{'initialization_grade':'C','original_initial_identity_verified':False})
        js(root/'statistical_review.json',{'status':'PASS','sha256':{n:sha(root/n) for n in ['relation_metric_summary.csv','protocol.json']}})
        final={'status':'passed','protected_f17_files_unchanged':191,'development':{'models':16,'formal_training_started':False,'engineering_status':'passed'}}
        js(root/'existing_final_audit.json',{'status':'passed'});js(root/'final_audit.json',final)
        for name in ARTIFACT_NAMES:js(art/name,{}) if name.endswith('.json') else write_csv(art/name,[{'synthetic':'only'}])
        jobs=[dict(run_id=f'fixture_{a}_{e}_{t}',architecture=a,encoding=e,task=t) for a in ARCHS for e in ENCODINGS for t in TASKS]
        protocol={'tasks':list(TASKS),'architectures':list(ARCHS),'lengths':list(LENGTHS),
                  'relations':[{'pair_id':r} for r in RELATIONS],'units':[{'unit':'synthetic_world'}],
                  'training_manifest':jobs,'training':{'updates':20000}}
        js(art/'protocol.json',protocol);digest=sha(art/'protocol.json')
        engineering={'status':'passed','models':16,'development_only':True,'formal_training_started':False,
                     'stop_after_report':True,'no_confirmation_pvalues':True,'gates':{'synthetic_cpu':'passed'}}
        js(art/'engineering_complete.json',engineering)
        js(art/'training_complete.json',{'models':16,'updates_per_model':20000,'formal_training_started':False,'protocol_sha256':digest})
        js(art/'evaluation_complete.json',{'formal_training_started':False,'protocol_sha256':digest})
        binding={'source_version_sha256':'synthetic_source_version',
                 'transformer':{'container_unchanged':True},'gru':{'original_parameter_values_bitwise_preserved':True}}
        js(art/'initializations/execution_binding_v4.json',binding)
        js(art/'explicit_dropout_repair_log.json',{'all_corresponding_tensors_bitwise_equal':True,'head_embedding_tied':True})
        js(art/'development_exclusion_final_audit.json',{'status':'passed','prior_new_orbit_intersection':0})
        learning=[];accuracy=[];baseline=[]
        for arch in ARCHS:
            js(art/'initializations'/arch/'step0.json',{'actual_initial_identity_verified':True,'parameter_sha256':'synthetic_init_'+arch})
        for job in jobs:
            done=dict(job,step=20000,actual_initial_identity_verified=True,initial_parameter_sha256='synthetic_init_'+job['architecture'],
                parameters=PARAMS[job['architecture']],protocol_sha256=digest,data_sha256='synthetic_shared_data',
                execution_source_version_sha256='synthetic_source_version',curve=[{'step':20000}])
            js(art/'models'/job['run_id']/'completed.json',done)
            learning.append(dict(job,parameters=done['parameters'],initial_validation_accuracy=.1,final_validation_accuracy=.7,gain=.6,learning_gate_passed=True))
            for condition in ('initial','trained'):
                for action in ('identity','complement','inverse'):
                    val=.7 if condition=='trained' else .1
                    baseline.append(dict(architecture=job['architecture'],encoding=job['encoding'],task=job['task'],condition=condition,action=action,
                        full_vocab_accuracy=val,numeric_accuracy=val,length_majority_accuracy=.3,full_vocab_gain_over_length_majority_pp=100*(val-.3),numeric_gain_over_length_majority_pp=100*(val-.3)))
                    for n in LENGTHS:
                        accuracy.append(dict(architecture=job['architecture'],encoding=job['encoding'],task=job['task'],condition=condition,length=n,action=action,full_vocab_accuracy=val,numeric_accuracy=val))
        write_csv(art/'source_learning.csv',learning);write_csv(art/'task_accuracy_length.csv',accuracy)
        write_csv(dev/'model_minus_length_baseline_summary_supplemental.csv',baseline)
        write_csv(dev/'model_minus_length_baseline_supplemental.csv',[{'synthetic':'only'}])
        js(dev/'length_baseline_protocol_supplemental.json',{'post_protocol_descriptive':True,'main_metric_or_gate_or_budget_changed':False})
        (dev/'report.html').write_text('Synthetic fixture placeholder only\n')
        summary=[];lengths=[];gamma=[]
        for arch in ARCHS:
            for relation in RELATIONS:
                for control in CONTROLS:
                    for branch in BRANCHES:
                        for metric in METRICS:
                            undefined=(arch=='gru' and relation=='peak_complement' and control=='answer_strata' and branch=='output_prob' and metric=='procrustes')
                            for encoding,B0 in [('one_line',-.4),('canonical_cycle',.4)]:
                                Bt=.2;dc=.01;delta=Bt-B0;dw=dc-delta
                                row=dict(architecture=arch,encoding=encoding,relation=relation,control=control,branch=branch,metric=metric,
                                    B0=B0,Bt=Bt,change_correct=dc,change_wrong=dw,Delta=delta,undefined_lengths=21 if undefined else 0)
                                if undefined:
                                    for field in COMPONENTS:row[field]=None
                                summary.append(row)
                                lengths.extend(dict(row,length=n) for n in LENGTHS)
                            row=dict(architecture=arch,relation=relation,control=control,branch=branch,metric=metric,
                                     Gamma0=.8,Gammat=0.,Gamma_correct=0.,Gamma_wrong=.8,GammaDelta=-.8)
                            if undefined:
                                for field in GAMMA_MAP.values():row[field]=None
                            gamma.append(row)
        write_csv(art/'geometry_summary.csv',summary);write_csv(art/'geometry_length.csv',lengths)
        write_csv(art/'encoding_interactions.csv',gamma)
        def refresh_export():
            records=[dict(export_relative=str(p.relative_to(dev)),sha256=sha(p)) for p in sorted(art.rglob('*')) if p.is_file()]
            js(dev/'export_audit.json',{'status':'passed','fixture_only':True,'source_models':16,'fixed_final_updates':20000,
                'all_four_conditions_two_relations_controls_metrics_present':True,'no_confirmatory_inference':True,
                'original_undefined_values_retained':True,'source_small_artifacts':records})
        refresh_export()
        evidence=generate(root,dev,fixture=True)
        page=(root/'completion_report.html').read_text();text=(root/'completion_report.txt').read_text()
        assert '合成 CPU fixture' in page and '用户授权边界' in page and '未定义' in page
        assert 'Gamma_correct' in page and 'Gamma_wrong' in page and '+0.80000' in page and '-0.80000' in page
        assert text.count('prefix_final /')==8 and text.count('output_prob /')==8
        assert '它不是原来两条取逆关系的新确认实验' in text
        assert 'dropout、数据顺序和优化随机性' in text
        assert evidence['geometry_summary_rows']==128 and evidence['interaction_rows']==64
        # Failed source-learning/engineering gates stay visible, not upgraded.
        engineering['status']='failed';js(art/'engineering_complete.json',engineering)
        final['development']['engineering_status']='failed';js(root/'final_audit.json',final)
        learning[0]['learning_gate_passed']=False;write_csv(art/'source_learning.csv',learning);refresh_export()
        failed=generate(root,dev,fixture=True)
        assert failed['engineering_status']=='failed' and '工程完成状态：failed' in (root/'completion_report.txt').read_text()
        # Invalid or incomplete evidence must never silently become a report.
        rejected=[]
        def must_fail(label):
            try:validate(root,dev,fixture=True)
            except (RuntimeError,ValueError,KeyError):rejected.append(label)
            else:raise AssertionError(label+' was not rejected')
        (root/'final_audit.json').unlink();must_fail('missing_combined_final_audit');js(root/'final_audit.json',final)
        write_csv(art/'geometry_summary.csv',summary[:-1]);refresh_export();must_fail('missing_condition');write_csv(art/'geometry_summary.csv',summary)
        write_csv(art/'geometry_length.csv',lengths[:-1]);refresh_export();must_fail('missing_length');write_csv(art/'geometry_length.csv',lengths)
        gamma[0]['Gamma0']=.9;write_csv(art/'encoding_interactions.csv',gamma);refresh_export();must_fail('incorrect_Gamma0');gamma[0]['Gamma0']=.8;write_csv(art/'encoding_interactions.csv',gamma)
        refresh_export();(art/'source_learning.csv').write_text('tampered\n');must_fail('export_hash_mismatch')
        return dict(status='passed',synthetic_only=True,real_development_results_read=False,
            all_128_components_64_interactions_21_lengths_preserved=True,
            failed_engineering_gates_preserved=True,Delta_flip_with_equal_Bt_fixture=True,
            undefined_preserved=True,rejected_cases=rejected,training_or_GPU_started=False,
            outputs_written_only_in_temporary_directory=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--development',type=Path)
    parser.add_argument('--fixture',action='store_true')
    parser.add_argument('--check-ready',action='store_true',help='Check file presence only; never read result tables')
    args=parser.parse_args();root=args.root.resolve();development=(args.development or root/'encoding_development').resolve()
    if args.fixture: result=fixture()
    elif args.check_ready:
        missing=missing_files(root,development)
        result=dict(ready=not missing,missing_files=missing,result_tables_read=False,training_or_GPU_started=False)
    else:result=generate(root,development)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
