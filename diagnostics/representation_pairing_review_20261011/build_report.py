"""Reader report from saved review tables. No new CKA, model or fitting."""
from __future__ import annotations
import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
import csv
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def rows(n):
    with (HERE/n).open() as f:return list(csv.DictReader(f))
def num(x):
    if x in (None,'','None','NaN','nan'):return None
    y=float(x);return y if math.isfinite(y) else None
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def stats(rr,field):
    x=[num(r[field]) for r in rr];z=[v for v in x if v is not None]
    return dict(total=len(x),finite=len(z),undefined=len(x)-len(z),positive=sum(v>0 for v in z),negative=sum(v<0 for v in z),zero=sum(v==0 for v in z))
def table(rr,fields):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'')))+'</td>' for k in fields)+'</tr>' for r in rr)+'</tbody></table></div>'

def run():
    old=read(HERE/'existing_results_review.json');new=read(HERE/'analysis_audit.json')
    protected=read(HERE/'protection_end.json');reader=read(HERE/'result_reader_audit.json')
    assert old['status']==new['status']==protected['status']==reader['status']=='passed'
    p=read(HERE/'protocol.json');pairs=rows('pair_summary.csv');comparisons=rows('designated_category_comparisons.csv')
    change=rows('common_support_training_comparisons.csv');negative=rows('six_negative_bt.csv');w=rows('W_actual_summary.csv')
    relevant=[r for r in comparisons if r['phase']=='trained' and r['comparison_category']=='not_designated']
    summary={}
    for domain in p['systems']:
        summary[domain]={}
        for control in p['systems'][domain]['controls']:
            rr=[r for r in relevant if r['domain']==domain and r['control']==control]
            summary[domain][control]=stats(rr,'designated_minus_category_mean_state')
    known=rows('classification_manifest.csv')
    assert Counter(r['category'] for r in known)=={'designated':12,'not_designated':120}
    missing=[r for r in pairs if any(num(r['fixed_transform_score_'+phase]) is None for phase in ('initial','trained'))]
    # Plot fixed scores on the full frozen candidate grid; missing and excluded
    # cells remain masked. No best-model/task/layer subset is selected.
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    relation_names={'descent_inverse':'DESC / RECOIL','peak_complement':'PEAK / VALLEY','fixed_complement':'FIXED / ANTI',
                    'exceedance_inverse':'EXCEED / DEF','subsequence_complement':'LIS / LDS','run_complement':'INC / DEC RUN',
                    'record_complement':'MAX / MIN RECORD','decomposition_complement':'GLOBAL DESC / COMP'}
    perm_protocol=read(p['systems']['permworld']['protocol_path']); tasks=perm_protocol['label_task_names']; relations=[r['pair_id'] for r in perm_protocol['relations']]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dedede')
    fig,axes=plt.subplots(4,2,figsize=(18,17),layout='constrained')
    for i,(a,e) in enumerate(( (a,e) for a in perm_protocol['architectures'] for e in perm_protocol['encodings'])):
        for j,control in enumerate(('raw_within_length','answer_strata')):
            ax=axes[i,j];m=np.full((8,16),np.nan);stars=[]
            for r in pairs:
                if (r['domain'],r['architecture'],r['encoding'],r['control'])!=('permworld',a,e,control):continue
                y,x=relations.index(r['relation']),tasks.index(r['right_task']);v=num(r['fixed_transform_score_trained'])
                if v is not None:m[y,x]=v
                if r['category']=='designated':stars.append((x,y))
            im=ax.imshow(m,cmap=cmap,vmin=0,vmax=1,aspect='auto');ax.set_title(f'{a} / {e} / {control}')
            ax.set_xticks(range(16),tasks,rotation=70,ha='right',fontsize=8);ax.set_yticks(range(8),[relation_names[x] for x in relations],fontsize=9)
            if stars:ax.scatter(*zip(*stars),marker='*',c='white',edgecolors='black',s=70,linewidths=.7)
    fig.colorbar(im,ax=axes,label='Fixed-transform trained ordinary CKA (each candidate uses all21 lengths)');fig.suptitle('Full PermWorld right-task control; white star = designated. Grey = excluded/undefined; see CSV for distinction.',fontsize=14)
    fig.savefig(HERE/'perm_all_candidate_scores.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,7),layout='constrained');configs=[(a,e) for a in perm_protocol['architectures'] for e in perm_protocol['encodings']]
    for j,control in enumerate(('raw_within_length','answer_strata')):
        m=np.full((8,4),np.nan)
        for r in relevant:
            if r['domain']=='permworld' and r['control']==control:m[relations.index(r['relation']),configs.index((r['architecture'],r['encoding']))]=num(r['designated_minus_category_mean_state'])
        im=axes[j].imshow(m,cmap='coolwarm',vmin=-1,vmax=1,aspect='auto');axes[j].set_title(control)
        axes[j].set_yticks(range(8),[relation_names[x] for x in relations]);axes[j].set_xticks(range(4),[f'{a}\n{e}' for a,e in configs],fontsize=8)
        for y,x in np.ndindex(m.shape):
            if np.isfinite(m[y,x]):axes[j].text(x,y,f'{m[y,x]:+.3f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=axes,label='Designated score - finite mean of not_designated candidates');fig.suptitle('Same input transform, different task specialist. Descriptive, candidate-specific strata.')
    fig.savefig(HERE/'perm_designated_contrasts.png',dpi=180);plt.close(fig)
    graph_relations=[r['relation'] for r in read(p['systems']['graphworld']['protocol_path'])['relations']]
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    graph_tasks=['max_out','min_out','max_in','min_in']
    for j,control in enumerate(p['systems']['graphworld']['controls']):
        m=np.full((4,4),np.nan);stars=[]
        for r in pairs:
            if r['domain']!='graphworld' or r['control']!=control:continue
            y,x=graph_relations.index(r['relation']),graph_tasks.index(r['right_task']);v=num(r['fixed_transform_score_trained'])
            if v is not None:m[y,x]=v
            if r['category']=='designated':stars.append((x,y))
        im=axes[j].imshow(m,cmap=cmap,vmin=0,vmax=1);axes[j].set_title(control,fontsize=10)
        axes[j].set_xticks(range(4),graph_tasks,rotation=35,ha='right');axes[j].set_yticks(range(4),graph_relations)
        if stars:axes[j].scatter(*zip(*stars),marker='*',c='white',edgecolors='black',s=100)
    fig.colorbar(im,ax=axes,label='Fixed-transform trained CKA, equal n6/7/8');fig.suptitle('GraphWorld12 fixed candidate pairs; density is sensitivity, not a replacement primary control')
    fig.savefig(HERE/'graph_all_candidate_scores.png',dpi=180);plt.close(fig)
    lines=['PermWorld / GraphWorld 当前结果审查补充（2026-10-11；已有缓存分析，零新增训练/前向）','',
        '本轮解决了什么：核实64+4真实20k终点、独立复算旧32主格和六负格、固定同一输入变换更换右任务的480+12配对控制，并在实际分析输入上核实原错误变换的答案公式区分能力。所有新增跨任务指标是看过旧结果后的探索性分析，冻结发生在新CKA计算前，不追溯称为确认性预注册。','',
        '旧结论：在答案分层、最终层及逐长度等权的主分析中，PermWorld的32个条件有26个表现出正的配对选择性。选择性随编码、层和控制改变；相对于匹配初始化的训练增量需另行解释。正确数学对应常伴随几何选择性，但当前证据不单独识别数学结构机制。',
        'B0=S_correct,initial−S_wrong,initial；Bt=S_correct,trained−S_wrong,trained；delta_correct与delta_wrong分别是两项训练变化；Delta=Bt−B0=delta_correct−delta_wrong。对给定训练结果，修改初始化参照不能改变Bt，只改变Delta及增量解释。本轮没有多个独立初始化识别初始化对Bt的因果作用。',
        '旧表复算主Bt26正/6负、Delta18正/14负，特指答案分层、prefix_final最终norm、普通CKA及n10–30逐长度1/21等权。六负格全部在轮换编码，详细双源准确率/训练定义长度众数/逐长度符号见six_negative_bt.csv。三指标主Bt32/32方向一致，Delta30/32；两例是GRU一行的峰/谷、连续段，kNN Delta负而CKA/Procrustes正。这是共用激活的几何读数一致，不是独立复现或功能证据。',
        '弱任务与容量差异不能忽略，也不能解释所有结果：GRU轮换全局下降/组件两源约99.414%/99.991%，仍有负Bt；GRU超越/亏缺两编码都接近满分，Bt均正而Delta翻转，Gamma0=1.09407、Gamma_t=0.06432、Gamma_Delta=−1.02975。超过众数不等于充分掌握。','',
        '新增控制：固定左任务f、原正确T、编码配置、输入与最终取点，更换右specialist g；与旧“固定任务对、改变输入正确/错误角色”是不同对比，不混合计数。字段是fixed_transform_score_initial/trained，不叫旧Bt或Delta。两类控制均逐长度原权重：Perm n10–30各1/21，Graph n6/7/8各1/3；任何长度未定义则该候选汇总未定义，不填零或重权。',
        '分类目录只主张已证明的指定公式；other_known_exact本目录为空，仍单列空类。not_designated是未列入本对应，不等于毫无数学关系；没有依CKA挑候选。答案分层仍用原共同答案key，包含右任务原输入角色的答案，但不计算新的错误角色CKA。完整raw anchors共用，各候选答案mask与分层不同，group count/residual df/IDs/key hashes见pair_masks.csv。指定−类均值差是描述性对照；任务身份、标签分布、掌握与分层支持仍改变，不是数学关系的纯因果效应。','',
        '固定变换时指定对应是否比未列入候选更相似（trained，指定score−该类有限完整长度score均值；正/负/未定义是条件计数，不是独立重复）：']
    for domain,cc in summary.items():
        for control,s in cc.items():lines.append(f'{domain} / {control}: {s["positive"]}正、{s["negative"]}负、{s["zero"]}零、{s["undefined"]}未定义，共{s["total"]}个耦合条件。')
    lines+=['答案控制改变了新观察：Perm指定对应超过未列入候选均值的条件从raw30/32变为答案分层29/32；Graph从raw4/4变为答案分层3/4，再加原精确边数控制为2/4。这里的“超过”是超过该类均值，不能写成在全部候选中排名第一。',
        'Perm答案控制下的三个负新对照为轮换编码的Transformer不动/反不动、GRU不动/反不动、Transformer全局下降/组件；它们不是旧六个负Bt的同一个计数。Graph reverse_min的新对照从raw +0.31530变为答案控制 −0.01319；complement_in从答案控制 +0.00901变为密度敏感性 −0.00845。接近零的符号不代表显著正负。',
        '新Gcat的正条件数另列：Perm raw30/32、答案控制27/32；Graph raw4/4、答案控制2/4、密度敏感性2/4。这些是对共同有限候选集合的相对初始化变化，不是原关系Delta，也不是新的独立证据数量。']
    lines+=['','每个条件全部候选的原值、各类有限均值/中位数/有限数量，以及initial与trained各自支持名单均保存。同配置真实step0参数共用，因此raw初始化的右任务分数相同；答案分层的初始化差异还可能来自候选特定的分组与支持，不是不同初始化权重。相对初始化新差值Gcat只用该类两阶段均有限候选的交集C：Gcat=(S_designated,t−mean_C S_g,t)−(S_designated,0−mean_C S_g,0)，并列C名单。它不是原Delta；候选自己每阶段也需完整21/3个长度。',
        f'全候选汇总{len(pairs)}行（包括控制版本），至少一阶段未定义的行{len(missing)}；480+12是候选配对数量，不是新增模型或独立样本数。','',
        '错误变换的实际区分能力：Graph四关系在原实际全部分析anchors上的等权偶然公式保持率依次为56.771%、54.818%、56.771%、59.701%；原答案控制约57.8%–61.6%，原密度控制约66.9%–76.5%。它们与旧random-math51.9%–65.57%是不同输入批次。原W不保证逐样本破坏公式，限制原对比区分能力，但不因此宣称实验无效。所有匹配行、W及原结果保留。',
        'reverse_max与complement_out全部1536 raw布尔事件完全相同（均是max_out(A)=max_out(W(A))），不能当独立关系；各自答案selector仍不同。Perm全局下降/组件的原inverse错配保持率raw等权约77.0%。W_actual_per_anchor.csv有92160关系—输入行，实际独特底层对象只有10752 Perm+1536 Graph，372原分层样本数核对通过；等权率与pooled分子/分母分别报告。','',
        '独立核验与保护：旧表算术256022逐长度/12120汇总比较最大差0；新指定32+4各phase/control/长度直接与旧correct数值核对，另有固定10pair首长度的40项Gram/协方差独立计算。它们都来自保存激活，没有重做前向。result_reader_audit另从保存分数核验预定少量聚合、类别与差值。未定义项及原因保留；校准若讨论只指零假设，不称功能性度量。',
        f'开始和结束保护范围为{protected["files"]}个明确文件，全部字节匹配；Git HEAD和tracked worktree/index快照相同。此为具体范围的开始/结束核验，不证明所有文件或全过程始终未变。F17、旧冻结报告/协议/缓存、当前入口和旧归档均保持原样。工程修复记录独立保存，冻结科学源码以执行时SHA为准。','',
        '仍未解决：当前都是每系统一个世界、每配置共享真实step0，配置容量未匹配；标签、置信度/预测误差与学习程度仍可解释一部分几何模式。新控制补齐任务对应的描述性比较，不识别独立数学结构机制，更不提供功能收益。','',
        '有限复现提案（不执行）：GRU超越/亏缺近掌握编码对、GRU极大/极小记录补排列对、Transformer不动/反不动含轮换负选择性对。每独立world+true-step0单元12个unique specialists，同单元两编码配对，固定原20k与主分析；1–2单元仅是成本场景。历史matching source进程参考约2.727/5.455小时，包含验证IO，不是GPU活跃时间、未来速度保证或功效保证；未来缓存成本未知另列。三领域不预设全正，弱任务照交付，不自动调到准确率门槛。Graph扩关系/GroupWorld/第三领域仅下一阶段候选，见next_phase_candidates.txt，无数据准备或执行。','',
        '结论与停止：本轮为旧模型、旧输入、保存激活的限定审查，零新增训练、优化器、模型、前向、probe、迁移、世界、层指标扩展、上传、删除或移动。完成报告和保护审计后停止。']
    zh='在答案分层、最终层及逐长度等权的主分析中，PermWorld的32个条件有26个表现出正的配对选择性。选择性随编码、层和控制改变；相对于匹配初始化的训练增量需另行解释。在新增的固定输入变换、改变右任务对照中，答案分层后指定对应超过未列入对应候选均值的条件为PermWorld29/32、GraphWorld3/4；这些条件共享模型和输入，不是独立重复。这一新对照不等同于原Bt/Delta，也不识别任务关系的纯因果效应。正确数学对应常伴随几何选择性，但当前证据不单独识别数学结构机制或功能收益。'
    en='In the answer-stratified, final-layer analysis with equal weighting across lengths, 26 of 32 PermWorld conditions exhibit positive pairing selectivity. Selectivity varies with encoding, layer and control, while training increments relative to matched saved initializations require separate interpretation. In the new fixed-transform, varying-right-task control, designated correspondences exceed the mean of candidates not designated for that correspondence in 29/32 PermWorld and 3/4 GraphWorld answer-stratified conditions. These conditions share models and inputs and are not independent replications. The new score contrasts are distinct from the original Bt and Delta and do not identify a pure causal effect of the mathematical relation. Exact mathematical correspondences often accompany geometric selectivity, but these exploratory observations alone do not identify a mathematical-structure mechanism or functional benefit.'
    (HERE/'results_paragraph_zh_en.txt').write_text('中文\n'+zh+'\n\nEnglish\n'+en+'\n')
    text='\n'.join(lines)+'\n';(HERE/'reader_report.txt').write_text(text)
    extra=['<h2>六个负Bt条件（原主分析）</h2>',table(negative,['architecture','encoding','relation','B0','Bt','Delta','dc','dw','left_test_answer_accuracy','left_test_length_majority_accuracy','right_test_answer_accuracy','right_test_length_majority_accuracy','Bt_length_signs']),
        '<h2>新对照：各条件指定对应−预定类别</h2>',table(comparisons,['domain','relation','architecture','encoding','control','phase','comparison_category','designated_score','finite_mean','finite_median','defined_count','total_count','designated_minus_category_mean_state','finite_support_right_tasks']),
        '<h2>共享有限支持下的新训练变化（不是原Delta）</h2>',table([r for r in change if r['category']!='designated'],['domain','relation','architecture','encoding','control','category','finite_common_pair_count','designated_minus_common_category_initial','designated_minus_common_category_trained','Gcat','common_support_right_tasks']),
        '<h2>所有右任务（原值）</h2>']
    for domain in p['systems']:
        for control in p['systems'][domain]['controls']:
            extra.append('<details><summary>'+html.escape(domain+' / '+control)+'</summary>'+table([r for r in pairs if r['domain']==domain and r['control']==control],['relation','architecture','encoding','right_task','category','fixed_transform_score_initial','fixed_transform_score_trained','minimum_selected_count','maximum_selected_count','initial_undefined_lengths','trained_undefined_lengths'])+'</details>')
    images=['perm_all_candidate_scores.png','perm_designated_contrasts.png','graph_all_candidate_scores.png']
    csvs=sorted(x.name for x in HERE.glob('*.csv'))
    document='<!doctype html><html lang="zh"><meta charset="utf-8"><title>PermWorld / GraphWorld 审查补充</title><style>body{font:16px/1.6 system-ui,sans-serif;margin:32px auto;max-width:1400px;padding:0 20px;color:#17252b}pre{white-space:pre-wrap;font:inherit}.table{overflow:auto;margin:16px 0}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border:1px solid #ddd;white-space:nowrap}th{background:#e9f0f3;position:sticky;top:0}img{max-width:100%}details{padding:12px;border:1px solid #ccc;margin:12px 0}</style><h1>PermWorld / GraphWorld 当前结果审查补充</h1><pre>'+html.escape(text)+'</pre>'+''.join('<figure><img src="'+n+'"><figcaption>'+n+'</figcaption></figure>' for n in images)+''.join(extra)+'<h2>完整CSV与复现入口</h2><ul>'+''.join('<li><a href="'+n+'">'+n+'</a></li>' for n in csvs)+'</ul><p><a href="REPRODUCE.txt">REPRODUCE.txt</a> · <a href="protocol.json">冻结协议</a> · <a href="results_paragraph_zh_en.txt">中英文表述</a> · <a href="replication_candidates.txt">有限复现候选</a></p></html>'
    (HERE/'reader_report.html').write_text(document)
    input_names=['existing_results_review.json','analysis_audit.json','protection_end.json','result_reader_audit.json','pair_summary.csv','designated_category_comparisons.csv','common_support_training_comparisons.csv','six_negative_bt.csv','W_actual_summary.csv','classification_manifest.csv','protocol.json','freeze.json']
    outputs=['reader_report.txt','reader_report.html','results_paragraph_zh_en.txt']+images
    audit=dict(status='completed_reader_report',writer_sha256=sha(__file__),summary=summary,
        candidate_summary_rows=len(pairs),candidate_phase_undefined_rows=len(missing),
        inputs_sha256={n:sha(HERE/n) for n in input_names},outputs_sha256={n:sha(HERE/n) for n in outputs},
        mathematical_metrics_recomputed=0,models=0,forward=0,training=0,old_writes=0,statistical_tests=0)
    (HERE/'report_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=audit['status'],summary=summary),ensure_ascii=False))

if __name__=='__main__':run()
