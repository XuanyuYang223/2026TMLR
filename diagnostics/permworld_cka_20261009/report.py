"""Readable bounded diagnostic report, figures, and reproduction manifest."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, WORKSPACE, load_csv, write_csv, write_json, sha

NAMES={'descent_inverse':'下降数—逆下降数（取逆）','peak_complement':'峰—谷（补排列）',
 'fixed_complement':'不动点—反不动点（补排列）','exceedance_inverse':'超越数—亏缺数（取逆）',
 'subsequence_complement':'LIS—LDS（补排列）','run_complement':'最长升段—最长降段（补排列）',
 'record_complement':'左至右最大纪录—最小纪录（补排列）','decomposition_complement':'全局下降—分解块数（补排列）'}


def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table>'


def fmt(x,digits=3):return f'{float(x):+.{digits}f}' if x not in ['',None] else '未定义'


def link(file,label=None):return f'<a href="{file}">{html.escape(label or file)}</a>'


def run():
    summary=load_csv(ROOT/'relation_summary.csv');src=load_csv(ROOT/'relation_seed_summary.csv')
    cells=load_csv(ROOT/'relation_length_cka.csv');lor=load_csv(ROOT/'leave_one_relation.csv')
    k=load_csv(ROOT/'k_summary.csv');kpool=load_csv(ROOT/'k_pooled_control.csv')
    audit=json.loads((ROOT/'initialization_audit.json').read_text());ver=json.loads((ROOT/'verification.json').read_text())
    inventory=json.loads((ROOT/'artifact_inventory.json').read_text())
    relations=list(NAMES)
    def sel(rel,typ='same_seed',control='answer_strata',metric='ordinary'):
        return next(r for r in summary if (r['relation'],r['seed_pair_type'],r['control'],r['metric'])==(rel,typ,control,metric))
    # Report finite-sample resampling shifts explicitly. These are distributions, not nominal population CIs.
    b=np.load(ROOT/'input_bootstrap_delta.npz');resampling=[]
    for rel in relations:
        for typ,ids in [('same_seed',[0,4,8]),('cross_seed',[1,2,3,5,6,7])]:
            ds=b[rel][ids].mean(0);obs=float(sel(rel,typ,'length')['delta_mean'])
            resampling.append(dict(family='specialist',relation_or_replicate=rel,k='',metric='delta',seed_pair_type=typ,
                observed=obs,resampling_mean=float(ds.mean()),resampling_shift=float(ds.mean()-obs),resampling_sd=float(ds.std(ddof=1)),
                resampling_quantile_025=float(np.quantile(ds,.025)),resampling_quantile_975=float(np.quantile(ds,.975)),
                interpretation='descriptive paired input-resampling distribution; finite-sample shift is not removed and percentiles are not a nominal population CI'))
    kb=np.load(ROOT/'k_input_bootstrap_draws.npz')
    for r in k:
        draws=kb[f'{r["replicate"]}_k{r["k"]}']
        for metric,ds in [('intermodel_cka',draws[0]),('mean_self_initial_cka',(draws[1]+draws[2])/2)]:
            obs=float(r[metric]);resampling.append(dict(family='k_series',relation_or_replicate=r['replicate'],k=r['k'],metric=metric,
                seed_pair_type='',observed=obs,resampling_mean=float(ds.mean()),resampling_shift=float(ds.mean()-obs),
                resampling_sd=float(ds.std(ddof=1)),resampling_quantile_025=float(np.quantile(ds,.025)),resampling_quantile_975=float(np.quantile(ds,.975)),
                interpretation='descriptive paired input-resampling distribution; no source-seed/world population inference'))
    write_csv(ROOT/'resampling_diagnostics.csv',resampling)
    families=[]
    for typ in ['same_seed','cross_seed']:
        for control in ['length','answer_strata']:
            for metric in ['ordinary','debiased','input_adjusted']:
                for family in ['inverse','complement']:
                    rs=[r for r in summary if r['seed_pair_type']==typ and r['control']==control and r['metric']==metric and r['relation'].endswith(family)]
                    families.append(dict(seed_pair_type=typ,control=control,metric=metric,transform_family=family,fixed_relations=len(rs),
                        delta_mean=float(np.mean([float(r['delta_mean']) for r in rs])),delta_median=float(np.median([float(r['delta_mean']) for r in rs]))))
    write_csv(ROOT/'transform_family_summary.csv',families)
    # The points are seed summaries, not confidence intervals or independent relation replicates.
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),sharey=True)
    for ax,control,title in zip(axs,['length','answer_strata'],['Within length, original inputs','Within length, common answer strata']):
        for typ,offset,color in [('same_seed',-.12,'#1565c0'),('cross_seed',.12,'#d55e00')]:
            rs=[sel(rel,typ,control) for rel in relations]
            mean=np.array([float(r['delta_mean']) for r in rs]);lo=np.array([float(r['seed_pair_min']) for r in rs]);hi=np.array([float(r['seed_pair_max']) for r in rs])
            ax.errorbar(np.arange(8)+offset,mean,yerr=[mean-lo,hi-mean],fmt='o',capsize=3,color=color,label=typ.replace('_',' '))
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(8),['descent I','peak C','fixed C','exceed I','LIS C','run C','record C','decomp C'],rotation=50,ha='right')
        ax.set_title(title);ax.grid(axis='y',alpha=.2)
    axs[0].set_ylabel('Delta of correct-minus-wrong linear CKA');axs[1].legend(fontsize=9)
    fig.suptitle('Mean over 21 lengths; bars show fixed seed-pair range, not confidence intervals',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'relation_delta.png',dpi=180);fig.savefig(ROOT/'relation_delta.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.7),sharey=True)
    for ax,control,title in zip(axs,['length','answer_strata'],['Original inputs','Common answer strata']):
        vals=np.array([[np.mean([float(r['delta']) for r in cells if r['relation']==rel and r['control']==control and r['metric']=='ordinary' and r['seed_pair_type']=='same_seed' and int(r['length'])==n]) for n in range(10,31)] for rel in relations])
        im=ax.imshow(vals,aspect='auto',cmap='RdBu_r',vmin=-1.6,vmax=1.6);ax.set_xticks(range(0,21,2),range(10,31,2));ax.set_xlabel('Permutation length');ax.set_title(title)
        ax.set_yticks(range(8),['descent I','peak C','fixed C','exceed I','LIS C','run C','record C','decomp C'])
    fig.colorbar(im,ax=axs.ravel().tolist(),label='Delta, fixed same-seed mean');fig.subplots_adjust(right=.85,wspace=.12)
    fig.savefig(ROOT/'relation_length_heatmap.png',dpi=180,bbox_inches='tight');fig.savefig(ROOT/'relation_length_heatmap.pdf',bbox_inches='tight');plt.close(fig)
    # Replot the requested k curves without confusing resampling percentiles with confidence bands.
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True)
    colors=plt.cm.tab10(np.arange(5))
    for i,rep in enumerate(['r0','r1','r2','r3','r4']):
        rs=sorted([r for r in k if r['replicate']==rep],key=lambda r:int(r['k']))
        label=f'{rep.upper()}, seed {rs[0]["seed"]}'+(' (subset)' if i>=3 else '')
        for ax,metric in zip(axs,['intermodel_cka','mean_self_initial_cka']):
            ax.plot([int(r['k']) for r in rs],[float(r[metric]) for r in rs],marker='o',linestyle='--' if i>=3 else '-',color=colors[i],label=label)
    axs[0].set_title('Between task pools A and B');axs[1].set_title('Retention relative to own rebuilt initialization')
    for ax in axs:ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],[1,2,4,8,16]);ax.set_ylim(0,1);ax.set_xlabel('Number of tasks k');ax.grid(alpha=.2)
    axs[0].set_ylabel('Mean of 29 within-length linear CKA values');axs[1].legend(fontsize=8)
    fig.suptitle('Original initialization identity unverified; R3/R4 are fixed-seed subset sensitivity',fontsize=10)
    fig.tight_layout();fig.savefig(ROOT/'k_drift.png',dpi=180);fig.savefig(ROOT/'k_drift.pdf');plt.close(fig)
    relationtable=[]
    for rel in relations:
        raw=sel(rel,control='length');ans=sel(rel);cross=sel(rel,'cross_seed')
        relationtable.append([html.escape(NAMES[rel]),fmt(raw['delta_mean']),fmt(ans['delta_mean']),fmt(ans['delta_median']),
            f"[{fmt(ans['seed_pair_min'])}, {fmt(ans['seed_pair_max'])}]",fmt(cross['delta_mean']),
            f"{ans['min_positive_lengths']}–{ans['max_positive_lengths']}/21"])
    seedtable=[]
    for rel in relations:
        vals=[next(r for r in src if r['relation']==rel and r['left_seed']==str(s) and r['right_seed']==str(s) and r['control']=='answer_strata' and r['metric']=='ordinary') for s in [17,42,101]]
        seedtable.append([html.escape(NAMES[rel])]+[fmt(r['delta']) for r in vals])
    sensitivity=[]
    for rel in relations:
        sensitivity.append([html.escape(NAMES[rel])]+[fmt(sel(rel,metric=m)['delta_mean']) for m in ['ordinary','debiased','input_adjusted']])
    rawtable=[]
    for rel in relations:
        r=sel(rel)
        rawtable.append([html.escape(NAMES[rel])]+[f'{float(r[f]):.3f}' for f in ['trained_correct','trained_wrong','trained_identity','init_correct','init_wrong','init_identity']])
    ktable=[]
    for rep in ['r0','r1','r2','r3','r4']:
        rs=sorted([r for r in k if r['replicate']==rep],key=lambda r:int(r['k']))
        ktable.append([rep.upper()+' / '+rs[0]['seed']]+[' / '.join(f'{float(r[f]):.3f}' for f in ['intermodel_cka','mean_self_initial_cka']) for r in rs])
    lo= [r for r in lor if r['seed_pair_type']=='same_seed' and r['metric']=='ordinary']
    lotable=[[NAMES[r['omitted_relation']],r['control'],fmt(r['delta_mean']),fmt(r['delta_median'])] for r in lo]
    loop=load_csv(ROOT/'leave_one_length.csv')
    inverse_ranges={rel:(min(float(r['delta']) for r in loop if r['relation']==rel and r['control']=='answer_strata'),max(float(r['delta']) for r in loop if r['relation']==rel and r['control']=='answer_strata')) for rel in ['descent_inverse','exceedance_inverse']}
    macro=load_csv(ROOT/'macro_summary.csv')
    paragraphs=[]
    paragraphs.append('<h1>PermWorld：已有检查点 CKA 诊断</h1><p>探索性诊断 · 2026-10-09 · 本轮已完成并停止；未启动训练。</p>')
    paragraphs.append('<p><strong>主要结论：</strong>相对于按原程序重建的初始化参照，两条取逆关系的“正确−错误”差值增量在全部可用种子配对、全部21个长度，以及三种固定指标下均为正。六条补排列关系没有同等稳定的答案控制后增量。这个差异是表征几何的证据，尚不能识别代数计算机制。</p>')
    paragraphs.append('<p><strong>初始化限制：</strong>98个逻辑模型对应94个独立检查点，全部属于“按原初始化程序重建，但无法直接验证原始权重”；0个拥有直接可验证的原始初始权重，0个因无法重建而被舍弃。没有找到这些模型的step-0权重或初始权重哈希。历史分析代码、模型结构、188-token词表、标准初始化与RNG调用顺序已核对；原训练commit及PyTorch版本未记录，不能把同种子或缓存复现写成初始权重身份已验证。</p>')
    paragraphs.append('<h2>盘点与复用范围</h2>')
    paragraphs.append(table(['已有产物','已完成内容','本轮处理'],[
      ['neurips 单性质模型','16任务×3种子，48个20,000-step检查点；原八关系CKA','复用权重、配置、检查点SHA与完成标记；不重新训练'],
      ['specialist_cka_controls','固定新输入；长度均值控制后合并的CKA、答案分层、重建初始化/独立随机参照、输出/答案几何、32次条件置换、置信度补充','复用全部数据和前缀缓存；合并长度统计不替代逐长度主分析'],
      ['specialist_input_adjustment','21长度分别计算的普通/输入调整CKA，3同种子，答案控制与初始化比较','直接复用同种子数值；最大复算差2.22×10⁻¹⁶'],
      ['Property32任务数系列','R0–R4、k=1/2/4/8/16、两池共50模型；4,096固定前缀；原合并长度A/B CKA；seed17/42随机激活','复用全部训练缓存；仅对缺失的现有seed101初始化参照做前向提取'],
      ['本轮补充','跨种子的对应参照、debiased与逐长度置换、共同输入重采样、自身初始化漂移','全部独立保存，F17与原CKA目录只读']]))
    paragraphs.append('<p>146个缓存抽样检查均通过任务token之前的原始前向：每个specialist条件取10/20/30长度，Property32取2/10/20/30长度；最大绝对误差8.89×10⁻⁶以内。这验证当前程序与缓存取点的一致性，不验证历史原始权重身份。'+link('initialization_audit.json','初始化与缺失产物审计')+'；'+link('initialization_history.json','历史源代码/RNG审计')+'；'+link('artifact_inventory.json','版本与数据来源')+'；'+link('model_inventory.csv','逐模型检查点哈希')+'。</p>')
    paragraphs.append('<h2>八种关系：固定长度权重后的训练增量</h2><p>状态为final_norm层的&lt;ONE_END&gt;，位于任务提示之前。同一批2,688个测试orbit anchors，长度10–30各128个，逐长度计算后固定取1/21；答案匹配使用原协议的共同标签分层并保留每层≥3的样本。三对同种子与六对有序跨种子分别汇总，每一单元减去对应初始化模型对的差值，未减统一常数。</p>')
    paragraphs.append('<p>Δ=(正确−错误)<sub>训练后</sub>−(正确−错误)<sub>对应初始化</sub>。Δ是两个CKA差值的差，可超过1；它不是准确率百分点，也不表示正确CKA本身增长相同数值。</p>')
    paragraphs.append(table(['关系','原输入Δ均值','答案控制Δ均值','答案控制种子中位数','同种子Δ范围','跨种子Δ均值','同种子正长度数'],relationtable))
    paragraphs.append('<p>两条取逆在9种配对×21长度的答案控制下均为正。下降—逆下降的同种子Δ约+0.589，超越—亏缺约+0.641。原输入中最长升/降段的+0.371在答案控制后变为−0.332；峰/谷、LIS/LDS的原输入结果也有种子不一致，不能将全部补排列关系概括为获得训练增量。</p>')
    paragraphs.append('<p>这里的“答案控制”是共同标签层内减均值，不是全部输出信息被移除，也不是同一标签的所有样本均不可区分。43–128个anchors/长度被保留；训练模型的概率、置信度、以及输入相关信息仍可能存在。</p>')
    paragraphs.append('<img src="relation_delta.png" alt="八关系训练增量及种子配对范围">')
    paragraphs.append('<details><summary>三个同种子结果（全部关系）</summary>'+table(['关系','seed17','seed42','seed101'],seedtable)+'</details>')
    paragraphs.append('<details><summary>保留原始CKA（答案控制、同种子均值）</summary>'+table(['关系','训练正确','训练错误','训练恒等','初始化正确','初始化错误','初始化恒等'],rawtable)+'</details>')
    paragraphs.append('<p><strong>增量的含义要谨慎：</strong>下降—逆下降的正确CKA为0.145，重建初始化为0.179；错误CKA则从0.685降为0.062。它的正Δ主要体现正确/错误相对关系的改变，不能表述成正确配对的绝对相似度全面提高。超越—亏缺的正确CKA从0.261到0.281，而错误从0.723到0.102。恒等配对也完整保留在表中。</p>')
    paragraphs.append('<h2>指标、长度、答案和输入控制</h2>')
    paragraphs.append(table(['关系','普通CKA Δ','debiased Δ','固定输入Gram调整 Δ'],sensitivity))
    paragraphs.append('<p>三种指标均保留两取逆/六补排列的均值方向；两取逆所有种子配对和每个长度仍为正。U-centered指标是预定的敏感性检查，答案层残差使样本相互依赖，因此不声称它是无偏条件总体估计。输入调整沿用单一位置排列矩阵核、Gram回归与PSD截断；只排除这个特定核捕捉的成分，不能排除全部输入混杂。</p>')
    paragraphs.append('<p>每个长度上，位置排列矩阵输入Gram对恒等、取逆、补排列的CKA都为1，故输入核的正确−错误差值为0；数值序列核的补排列CKA也为1（固定长度补值是正交符号翻转加常数），取逆不同。不同输入核含义已固定，未根据结果搜索核。'+link('input_length_baselines.csv','原始输入对照表')+'。</p>')
    paragraphs.append('<p>真实答案one-hot在所有关系的正确变换CKA均为1，错误约0.018–0.047，说明输出关联本身能产生很强的正确配对优势。答案共同分层后纯真值编码的方差为0，CKA记为未定义，未填0。输出概率仍可复现取逆的正Δ：同种子约+0.336和+0.331。隐藏CKA数值更高并不能单独证明有额外机制。'+link('answer_output_summary.csv','答案/输出基线')+'；'+link('answer_output_length_baselines.csv','逐长度基线')+'。</p>')
    paragraphs.append('<p>32次原协议的层内置换使用相同排列同步处理训练与初始化、正确与错误。长度内计算后固定加权，两个取逆的答案控制Δ均超过全部32次打乱结果（同种子打乱Δ最大约0.008和0.030）；描述性上尾秩为1/33。置换控制也会打断输入对应，不能据此证明数学机制；不作为新的确认检验。'+link('permutation_summary.csv','置换汇总')+'。</p>')
    paragraphs.append('<img src="relation_length_heatmap.png" alt="完整逐长度结果">')
    paragraphs.append('<p>移除任意一个长度后，两条取逆全部配对、三指标的答案控制均值仍为正：下降—逆下降范围'+f'[{inverse_ranges["descent_inverse"][0]:.3f}, {inverse_ranges["descent_inverse"][1]:.3f}]，超越—亏缺范围[{inverse_ranges["exceedance_inverse"][0]:.3f}, {inverse_ranges["exceedance_inverse"][1]:.3f}]。'+link('leave_one_length.csv','留一长度完整结果')+'。</p>')
    paragraphs.append('<h2>关系依赖与不确定性</h2><p>八种关系是固定研究对象，含两取逆和六补排列；它们共享输入、对称轨道和变换类型。三个源种子、六个交叉配对不增加独立关系数量。R0/R3/R4也共享seed17；这些不是五个独立数据世界。源/种子范围只描述这些源模型变化，留一关系只描述固定关系集合的敏感性。</p>')
    paragraphs.append('<p>普通CKA同种子宏平均Δ为+0.251，但关系中位数为−0.022；答案控制后宏均值−0.109、中位数−0.290。留掉任何单关系时，答案控制宏均值都为负（−0.217至−0.059）。因此不能把两个强取逆结果写成数学关系整体普遍提高对齐。'+link('transform_family_summary.csv','固定变换家族汇总')+'；'+link('macro_summary.csv','关系均值/中位数')+'。</p>')
    paragraphs.append('<details><summary>全部留一关系结果（普通CKA、同种子）</summary>'+table(['省略关系','控制','剩余均值','剩余中位数'],lotable)+'</details>')
    paragraphs.append('<p>输入抽样使用256次按长度分层的配对orbit-anchor重采样；同一索引用于所有关系/条件/种子，保留共享输入带来的依赖。输出保存重采样标准差、偏移和百分位。重复抽样会改变有限样本CKA的对角项权重，当前高维数据的重采样分布与原估计明显偏移；这些百分位明确作为描述性抽样敏感性，不能当作无偏总体置信区间，也没有据其显著性筛选关系。'+link('resampling_diagnostics.csv','输入重采样的标准差与偏移')+'。</p>')
    paragraphs.append('<h2>任务数系列：模型间相似度与初始化保留</h2><p>全部R0–R4、两池、五个k，共50个既有模型。固定原4,096个前缀，长度2–30分别计算，主分析固定1/29；原合并长度数值作为独立对照。A/B的重建初始参数在每组每个k都相同，原始训练起点是否比特相同仍无直接证据。'+link('k_shared_initialization_audit.csv','共享初始化审计')+'。</p>')
    paragraphs.append('<img src="k_drift.png" alt="任务数系列的两条并列曲线">')
    paragraphs.append(table(['划分 / seed','k1：池间 / 自初始化','k2','k4','k8','k16'],ktable))
    paragraphs.append('<p>从k1到k16，五组的平均自身初始化CKA都上升，R0/R1/R2分别为0.113→0.221、0.078→0.170、0.084→0.207。R0、R2、R4各相邻步骤都上升；R1从k8到k16下降，R3也有下降。因此支持“更多任务通常伴随更强的初始几何保留”的探索性线索，不支持普遍单调。两池相似度的轨迹也不完全单调，两类相似度不能精确分解学习与保留的贡献。</p>')
    paragraphs.append('<p>合并长度会改变结论的量级：R0的k8池间CKA原值0.803，逐长度主分析为0.214；其平均自身初始化CKA合并为0.740，逐长度为0.193。两项同时较高提供保留长度/初始几何的线索，但不能仅用这两项精确归因。最短长度有重复输入：n2只有2个唯一排列（143条记录），n3只有6个，n4只有24个；n2秩一几何可产生CKA天花板。全部长度按固定权重保留，未据结果剔除。'+link('k_pooled_control.csv','合并长度对照')+'；'+link('k_length_input_audit.csv','重复输入审计')+'。</p>')
    paragraphs.append('<h2>交付、验证与停止</h2><p>'+f'{ver["numeric_checks"]:,}项数值比较通过：feature-space CKA复核、独立无偏HSIC公式复核、显式重采样与加权Gram一致性、逐单元Δ/等长度权重、原k系列数值复现与初始化身份标签检查。最大误差{ver["maximum_absolute_discrepancy"]:.2e}。{ver["protected_files_checked"]:,}个受保护文件SHA保持不变，Git HEAD与已跟踪工作树保持不变。F17论文、结果、发布资产与原CKA产物均未改动。'+link('verification.json','验证记录')+'。</p>')
    paragraphs.append('<p>可复现代码及审计全部位于本目录。'+link('README.txt','运行步骤与产物索引')+'；'+link('protocol.json','固定分析口径')+'；'+link('relation_length_cka.csv','全部9,072行逐单元CKA/Δ')+'；'+link('k_model_self_initial_length.csv','50模型自身初始化逐长度表')+'；'+link('k_length_cka.csv','任务数系列逐长度表')+'。CSV提供完整原始数值，不以报告中的四舍五入数据替代。</p>')
    paragraphs.append('<h2>最后四个问题</h2>')
    paragraphs.append('<ol><li><strong>哪些关系稳定超过初始化参照？</strong>下降—逆下降、超越—亏缺两条取逆关系，在答案控制及输入核敏感性下稳定呈正Δ；只能称超过程序重建参照，原始初始化身份尚无法直接验证。</li><li><strong>是否依赖任务对、长度或指标？</strong>两条取逆自身不依赖某个长度/种子/三种指标之一；但总体结论强烈依赖关系类型和答案控制。六条补排列没有同等优势，宏平均也不能说明普遍效应。</li><li><strong>任务数上升是否伴随更强初始化保留？</strong>五个划分的k16均高于k1，三个完整轨迹单调；R1/R3有中途下降。原始共享初始权重未被直接验证，且输入/模型对之间不独立。</li><li><strong>下一步编码对照能解决哪个未决问题？</strong>可检验取逆差值增量是否依赖当前数值token、位置与one-line序列编码，还是在保持精确数学关系的另一种编码中仍存在。它还需各编码自身初始化和答案/输出参照；不能单靠CKA证明内部代数机制。本轮不开展该实验。</li></ol><p><strong>已按要求停止。</strong>没有新增模型训练、种子、任务、编码实验、零空间实验或有限域/GELU组合实验。</p>')
    style='body{font-family:system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#16212c;line-height:1.65}table{border-collapse:collapse;width:100%;font-size:13px;margin:16px 0}th,td{padding:9px;border:1px solid #d3dce5;text-align:left}th{background:#eef4f8}img{width:100%;margin:10px 0}details{border:1px solid #d3dce5;padding:12px;margin:14px 0}summary{cursor:pointer}a{color:#075aa1}h2{margin-top:32px}strong{font-weight:650}'
    (ROOT/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PermWorld CKA checkpoint diagnosis</title><style>'+style+'</style><body>'+''.join(paragraphs)+'</body></html>')
    (ROOT/'summary.txt').write_text('''PermWorld 已有检查点诊断：已完成并停止，未启动新训练。

范围：八关系、三个既有种子及其九种有序配对、21个固定长度；Property32五划分、两池、k=1/2/4/8/16，共50模型。

初始化身份：98个逻辑模型/94个唯一检查点均为“按原初始化程序重建，但无法直接验证原始权重”。缺失原始step-0权重/哈希及训练commit、框架版本。现有缓存与原程序的抽样前向一致；这不构成原初始权重身份验证。

1. 哪些关系有稳定训练增量？
下降数—逆下降数、超越数—亏缺数两条取逆关系，在答案控制下，同种子均值Δ分别为+0.589、+0.641；相对于对应的程序重建初始化参照，全部9种配对、21长度、3个固定指标均为正。六补排列关系没有同等稳定的答案控制后优势。

2. 依赖某对任务、长度或指标吗？
两取逆自身对留一长度与三指标检查稳健；总体主张依赖关系类型和答案控制。八关系的普通CKA宏均值Δ为+0.251、中位数−0.022；答案控制后均值−0.109、中位数−0.290，留掉任一关系后宏均值仍负。正Δ是正确/错误相对差值改变，不能读作正确CKA绝对上升，更不能从CKA认定编码或数学机制。

3. 更多任务是否伴随更强初始化保留？
五划分k16的平均自身初始化CKA均高于k1；R0/R2/R4单调，R1/R3有中途下降。池间CKA也不普遍单调。A/B的重建参数相同，原始共享初始权重未获直接验证；两项CKA不能精确分解相似性来源。

4. 编码对照能解决什么？
检验取逆差值增量是否依赖当前数值token、位置及one-line序列编码，还是在另一种保留数学关系的编码下仍出现。软输出与一般输入效应尚未完全排除。本轮没有开展编码实验。

输入重采样存在有限样本偏移，百分位仅作描述性敏感性；关系、种子与共享输入的不确定性分别报告，未增加独立关系数。完整结果、留一关系表、k曲线与可复现脚本见report.html和README.txt。

11,497项数值比较通过，1,214个受保护文件SHA未变。F17论文、结果、发布产物及原CKA目录保持不变。
''')
    write_json(ROOT/'completion.json',dict(status='complete_stopped',completed_utc=datetime.now(timezone.utc).isoformat(),training_started=False,
        stable_positive_relations=['descent_inverse','exceedance_inverse'],original_initial_identity_verified=False,
        protected_files_unchanged=ver['protected_files_checked'],scope='existing PermWorld checkpoints only',
        unanswered=['Original initialization bitwise identity','All input-encoding/soft-output effects','Internal algebraic mechanism','New data-world replication']))
    # All executable inputs are immutable upstream references with hashes or local scripts.
    files={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='deliverable_manifest.json'}
    write_json(ROOT/'deliverable_manifest.json',dict(files=files,workspace_commit=inventory['workspace_commit'],upstream_commit=inventory['upstream_commit']))
    print({'report':str(ROOT/'report.html'),'status':'complete_stopped'},flush=True)


if __name__=='__main__':run()
