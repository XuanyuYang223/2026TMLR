"""CSV-only addendum: no model loading, inference, CKA fitting, or training."""
import csv
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parent.parent
SOURCE=ROOT.parent/'permworld_cka_20261009'
D=Decimal


def load(path):
    with path.open() as f:return list(csv.DictReader(f))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write_json(name,data):
    (ROOT/name).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_csv(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def mean(vs):return sum(vs,D(0))/D(len(vs)) if vs and all(v is not None for v in vs) else None


def median(vs):
    if not vs or any(v is None for v in vs):return None
    a=sorted(vs);m=len(a)//2
    return a[m] if len(a)%2 else (a[m-1]+a[m])/2


def component(r):
    result={key:D(r[key]) if r.get(key,'')!='' else None for key in [
        'init_correct','init_wrong','init_identity','trained_correct','trained_wrong','trained_identity']}
    def diff(a,b):return a-b if a is not None and b is not None else None
    result.update(initial_preference=diff(result['init_correct'],result['init_wrong']),
        trained_preference=diff(result['trained_correct'],result['trained_wrong']),
        delta_correct=diff(result['trained_correct'],result['init_correct']),
        delta_wrong=diff(result['trained_wrong'],result['init_wrong']),
        delta_identity=diff(result['trained_identity'],result['init_identity']))
    result['minus_delta_wrong']=-result['delta_wrong'] if result['delta_wrong'] is not None else None
    result['delta']=diff(result['trained_preference'],result['initial_preference'])
    return result


def execute():
    original_files={p.name:sha(p) for p in sorted(SOURCE.iterdir()) if p.is_file()}
    previous=json.loads((SOURCE/'deliverable_manifest.json').read_text())
    assert all(original_files[name]==h for name,h in previous['files'].items())
    source_rows=load(SOURCE/'relation_length_cka.csv');assert len(source_rows)==9072
    checks=0;maxerr=D(0)
    def compare(a,b,name):
        nonlocal checks,maxerr
        if a is None or b is None:assert a is None and b is None,name;return
        err=abs(a-b);assert err<D('1e-12'),(name,err)
        checks+=1;maxerr=max(maxerr,err)
    output=[];by_seed=defaultdict(list)
    for r in source_rows:
        comp=component(r)
        compare(comp['delta'],D(r['delta']),'cell Delta')
        compare(comp['delta'],comp['delta_correct']-comp['delta_wrong'],'component identity')
        compare(comp['initial_preference'],D(r['init_contrast']),'initial preference')
        compare(comp['trained_preference'],D(r['trained_contrast']),'trained preference')
        row={**{k:r[k] for k in ['relation','left_task','right_task','left_seed','right_seed','seed_pair_type','length','anchors','control','metric','initialization_status']},
            'encoding':'one_line',**comp}
        output.append(row);by_seed[r['relation'],r['left_seed'],r['right_seed'],r['control'],r['metric']].append(row)
    fields=list(component(source_rows[0]));seed_rows=[]
    seed_lookup={(r['relation'],r['left_seed'],r['right_seed'],r['control'],r['metric']):r for r in load(SOURCE/'relation_seed_summary.csv')}
    for key,rows in by_seed.items():
        assert sorted(int(r['length']) for r in rows)==list(range(10,31))
        base=rows[0]
        result={k:base[k] for k in ['relation','encoding','left_seed','right_seed','seed_pair_type','control','metric','initialization_status']}
        for field in fields:result[field]=mean([r[field] for r in rows])
        result['length_median_delta']=median([r['delta'] for r in rows]);result['length_weight']='1/21'
        compare(result['delta'],D(seed_lookup[key]['delta']),'equal length aggregation')
        seed_rows.append(result)
    by_relation=defaultdict(list)
    for r in seed_rows:by_relation[r['relation'],r['seed_pair_type'],r['control'],r['metric']].append(r)
    original_summary={(r['relation'],r['seed_pair_type'],r['control'],r['metric']):r for r in load(SOURCE/'relation_summary.csv')}
    summary=[]
    for key,rows in by_relation.items():
        rel,typ,control,metric=key
        result=dict(relation=rel,encoding='one_line',seed_pair_type=typ,control=control,metric=metric,pairs=len(rows))
        for field in fields:
            result[field+'_mean']=mean([r[field] for r in rows]);result[field+'_median']=median([r[field] for r in rows])
        result['delta_min']=min(r['delta'] for r in rows);result['delta_max']=max(r['delta'] for r in rows)
        compare(result['delta_mean'],D(original_summary[key]['delta_mean']),'seed pair aggregation')
        for field in ['trained_correct','trained_wrong','trained_identity','init_correct','init_wrong','init_identity']:
            compare(result[field+'_mean'],D(original_summary[key][field]),'raw summary identity')
        compare(result['delta_mean'],result['delta_correct_mean']-result['delta_wrong_mean'],'mean components')
        summary.append(result)
    write_csv('decomposition_length.csv',output);write_csv('decomposition_seed.csv',seed_rows);write_csv('decomposition_summary.csv',summary)
    baseline_rows=[];baseline_groups=defaultdict(list)
    for r in load(SOURCE/'answer_output_length_baselines.csv'):
        comp=component(r)
        row={k:r[k] for k in ['relation','length','control','branch','left_seed','right_seed','anchors']}
        row.update(encoding='one_line',baseline_type='trained_and_initial_output' if r['branch']=='output_prob' else 'fixed_truth_geometry',**comp)
        baseline_rows.append(row)
        typ=('same_seed' if r['left_seed']==r['right_seed'] else 'cross_seed') if r['left_seed'] else 'not_applicable'
        baseline_groups[r['relation'],r['control'],r['branch'],typ,r['left_seed'],r['right_seed']].append(row)
        if comp['delta'] is not None:compare(comp['delta'],D(r['delta']),'output baseline arithmetic')
    baseline_seed=defaultdict(list)
    for (rel,control,branch,typ,sa,sb),rows in baseline_groups.items():
        assert sorted(int(r['length']) for r in rows)==list(range(10,31))
        baseline_seed[rel,control,branch,typ].append({field:mean([r[field] for r in rows]) for field in fields})
    baseline_summary=[]
    for (rel,control,branch,typ),rows in baseline_seed.items():
        baseline_summary.append(dict(relation=rel,encoding='one_line',control=control,branch=branch,seed_pair_type=typ,
            **{field+'_mean':mean([r[field] for r in rows]) for field in fields},
            semantics='Truth has no initialization/training increment; constant-stratum CKA stays undefined.' if branch!='output_prob' else 'Output readout geometry, not an estimate of hidden mechanism.'))
    write_csv('baseline_decomposition_length.csv',baseline_rows);write_csv('baseline_decomposition_summary.csv',baseline_summary)
    report(summary,baseline_summary)
    # Preservation check: the prior delivered diagnosis and all originally protected study files remain identical.
    assert all(sha(SOURCE/name)==digest for name,digest in original_files.items())
    protected=json.loads((SOURCE/'protected_before.json').read_text())['files']
    assert all(sha(WORKSPACE/path)==digest for path,digest in protected.items())
    write_json('audit.json',dict(scope='Independent Decimal arithmetic and fixed-weight aggregation of supplied CSVs; no independent rerun of raw model features/CKA/SVD or previous 11,497 checks.',
        arithmetic_checks=checks,maximum_arithmetic_discrepancy=str(maxerr),
        original_diagnostic_files_unchanged=len(original_files),protected_files_unchanged=len(protected),
        sources=original_files,training_started=False,original_initial_identity_verified=False,
        new_encoding_results_available=False,completed_utc=datetime.now(timezone.utc).isoformat()))
    write_json('manifest.json',dict(files={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='manifest.json'},
        workspace_commit='1073dcb3be4db90ec23476bab8f503c9e00f364b',status='planning_addendum_complete_no_training'))
    print(dict(rows=len(output),seed_summaries=len(seed_rows),summary_rows=len(summary),arithmetic_checks=checks,
        max_error=str(maxerr),training_started=False),flush=True)


def report(summary,baselines):
    chosen=[r for r in summary if r['seed_pair_type']=='same_seed' and r['control']=='answer_strata' and r['metric']=='ordinary']
    def fmt(x):return '未定义' if x is None else f'{x:+.4f}'
    headers=['关系','初始化偏好 B₀','训练偏好 Bₜ','正确变化 δc','错误变化 δw','Δ=δc−δw']
    grid='<table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'
    for r in chosen:
        grid+='<tr><td>'+html.escape(r['relation'])+'</td>'+''.join('<td>'+fmt(r[f+'_mean'])+'</td>' for f in ['initial_preference','trained_preference','delta_correct','delta_wrong','delta'])+'</tr>'
    grid+='</table>'
    output_table='<table><tr><th>关系</th><th>输出初始化偏好</th><th>输出训练偏好</th><th>输出Δ</th></tr>'
    for r in baselines:
        if r['control']=='answer_strata' and r['branch']=='output_prob' and r['seed_pair_type']=='same_seed':
            output_table+='<tr><td>'+r['relation']+'</td>'+''.join('<td>'+fmt(r[f+'_mean'])+'</td>' for f in ['initial_preference','trained_preference','delta'])+'</tr>'
    output_table+='</table>'
    body='''<h1>PermWorld：配对选择性拆解与编码对照计划</h1>
<p>本文件是对已有探索结果的补充，不修改原报告，不构成新的确认实验。只读取现有CSV；未推理、未重新计算CKA、未训练新模型。</p>
<p><strong>现有结果直接支持：</strong>训练改变了变换配对的相对偏好，这种改变受初始化几何和关系类型影响。Δ不能直接代表正确表征大幅趋同。原始初始权重仍不可验证，本表使用原程序重建的参照。</p>
<p>下表为答案控制、三个同种子、21长度固定等权均值。δc=正确训练后−正确初始化；δw=错误训练后−错误初始化；Δ=δc−δw。分解项是算术关系，不是因果贡献比例。</p>'''+grid+'''
<p>两取逆的错误CKA下降约0.62，正确变化分别为−0.0339和+0.0205。六补排列的Δ都负，但它们训练后的正确−错误均值都正；不能据此断言没有学到东西。是否形成了代数机制仍不能由CKA认定。</p>
<p>输出概率的同种子、答案控制结果：</p>'''+output_table+'''
<p>真值层内减均值不移除概率、置信度和预测误差结构。保留真值/输出基线，禁止用隐藏Δ减输出Δ估算额外机制。完整真值基线及全部种子/长度见baseline_decomposition_length.csv；真值没有初始化状态，恒定真值CKA仍记未定义。</p>
<p>跨编码必须共同报告 Γ₀=B₀(E₁)−B₀(E₀)、Γₜ=Bₜ(E₁)−Bₜ(E₀)、Γc=δc(E₁)−δc(E₀)、Γw=δw(E₁)−δw(E₀)，以及 ΓΔ=Γₜ−Γ₀=Γc−Γw。若Δ翻转但训练偏好不翻转，不得写成学习后的关系偏好翻转；未显著也不得写成等价。</p>
<p>R0/k8的池间CKA合并长度0.803、逐长度0.214继续作为长度混合敏感性示例。它不能精确分解初始化/输入/学习各自的贡献。</p>
<p><strong>下一轮问题：</strong>输入编码如何影响训练对变换配对选择性的重塑？分析字段和解释规则已固定在protocol.json；具体编码、模型预算、数据世界与源数量尚未冻结。当前只有one_line数据，未生成任何跨编码交互数值。</p>
<p>正式确认前需冻结完整执行协议，保存真实step-0权重及哈希，使用新的确认输入/轨道。相同初始参数并不保证相同初始化几何，因为编码输入不同；每种编码都要单独计算初始化与训练后的正确/错误/恒等CKA。改变词表、序列长度或模型结构时必须明确额外差异。</p>
<p>交付：<a href="decomposition_summary.csv">全部均值/中位数拆解</a>；<a href="decomposition_seed.csv">逐种子拆解</a>；<a href="decomposition_length.csv">逐长度拆解</a>；<a href="protocol.json">下一轮分析协议</a>；<a href="audit.json">CSV算术审计与保护校验</a>。复现只需在工作区运行 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python diagnostics/permworld_encoding_plan_20261009/decompose.py。</p>
<p><strong>本轮补充已完成并停止。</strong>没有启动新训练或编码实验，F17和上轮诊断产物保持不变。</p>'''
    style='body{font-family:system-ui,sans-serif;max-width:1100px;margin:32px auto;line-height:1.7;padding:0 18px}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0}td,th{border:1px solid #cdd7e0;padding:9px;text-align:left}th{background:#eef4f8}a{color:#075ba1}'
    (ROOT/'addendum.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PermWorld selection decomposition</title><style>'+style+'</style><body>'+body+'</body></html>')


if __name__=='__main__':execute()
