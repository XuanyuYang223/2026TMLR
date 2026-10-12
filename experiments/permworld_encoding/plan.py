"""Freeze a complete paired-encoding protocol using timing, not neural outcomes."""
import argparse
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np
import torch
from scipy.stats import t as student_t
from .core import *
from .evaluate import write_csv


def halfwidth(n,sigma):return float(student_t.ppf(.9875,n-1)*sigma/math.sqrt(n))


def exclusions():
    parent=Path('/home/yangx/neurips/data/permutation-properties-16m-v1/manifest.json')
    names={'data.npz','dataset.npz','probe_dataset.npz','source_data.npz','training_orbit_audit.npz'}
    candidates={p.resolve() for p in (WORKSPACE/'results').rglob('*.npz') if p.name in names}
    candidates.add(WORKSPACE/'diagnostics/permworld_cka_20261009/k_probe_inputs.npz')
    selected=[]
    for p in sorted(candidates):
        with np.load(p,allow_pickle=False) as a:
            if ('permutations' in a.files and 'lengths' in a.files) or any(k=='input' or k.endswith('_input') for k in a.files):
                selected.append(dict(path=str(p),sha256=sha(p)))
    return dict(parent_manifest=str(parent),parent_manifest_sha256=sha(parent),archives=selected,
        scope='All original 16m parent inputs plus all currently identified local input-bearing archives; all eight members of every new selected orbit must avoid the fixed history index.')


def create(root,target=.10,sigma=.20):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True);configpath=root/'protocol.json'
    if configpath.exists():
        verify_frozen(configpath);print('Protocol already frozen. Make an explicit new revision instead of overwriting.');return
    benchmark=json.loads((root/'benchmark.json').read_text());profile=json.loads((root/'analysis_profile.json').read_text())
    validation=json.loads((root/'encoding_validation.json').read_text());checks=json.loads((root/'integration_checks.json').read_text())
    assert validation['status']=='passed' and checks['status']=='passed' and benchmark['actual_parameter_updates']==0
    N=next(n for n in range(8,513) if halfwidth(n,sigma)<=target)
    units=[dict(unit=f'u{i:03d}',model_seed=73100000+i,data_seed=81200000+i,batch_seed=82300000+i,dropout_seed_base=83100000+1000*i) for i in range(1,N+1)]
    jobs=[dict(run_id=f'{u["unit"]}_{e}_{task}',unit=u['unit'],model_seed=u['model_seed'],data_seed=u['data_seed'],batch_seed=u['batch_seed'],
        encoding=e,task=task,updates=20000,effective_batch=64,shared_step0=f'{u["unit"]}/step0.pt') for u in units for e in ENCODINGS for task in TASKS]
    config=dict(protocol_version='permworld-encoding-selection/v1',status='frozen_before_formal_data_or_training',
        registered_utc=datetime.now(timezone.utc).isoformat(),research_question='输入编码如何影响训练对变换配对选择性的重塑？',
        phase_authorized_now='implementation, verification, protocol and cost planning only; formal training not launched',
        output='results/permworld_encoding_confirmation_v1',freeze_manifest=str((root/'freeze_manifest.json').relative_to(WORKSPACE)),
        encodings=list(ENCODINGS),tasks=list(TASKS),relations=RELATIONS,actions=list(ACTIONS),lengths=list(range(10,31)),
        model=dict(d_model=256,layers=4,heads=8,mlp_ratio=4,dropout=.1,tie_embeddings=True,max_seq_len=128,vocab_size=len(VOCABULARY)),
        canonical_cycle_grammar=dict(rule='Follow pi, rotate each cycle to start at its minimum, sort cycles by their minima, retain all singleton cycles.',
            prefix='<BOS> <SIZE> n <CYCLE_START> cycles <CYCLE_END>',within_cycle_separator=',',between_cycle_separator=';',
            commas_per_input='n minus number_of_cycles',semicolons_per_input='number_of_cycles minus one',
            prefix_tokens='2n+4',supervised_tokens_in_sequence='2n+8',one_line_same_token_length=True),
        data=dict(examples_per_length={'train':4096,'validation':256,'test':512},maximum_attempts_per_world=2000000,
            split_rule='Sequential unbiased acceptance of seeded random permutations, reject non-eight-element orbits/history/within-world overlap; no answer-based balancing or selection.',
            independence='Independent data/model/batch/dropout seeds per unit, conditional on the fixed mathematical universe and historical exclusions. Do not force global cross-world disjointness; accidental overlaps do not enter cross-world model comparisons.',
            training_actions=['identity'],validation_and_test_actions=list(ACTIONS),labels_source='Frozen upstream exact PROPERTY_FUNCTIONS, 16 properties'),
        training=dict(updates=20000,micro_batch=16,accumulation=4,effective_batch=64,learning_rate=.0003,weight_decay=.01,
            betas=[.9,.999],epsilon=1e-8,warmup_updates=1000,minimum_lr_ratio=.1,gradient_clip=1.,
            precision='FP32 parameters, CUDA BF16 autocast, TF32 disabled; FP32 CPU fallback',attention='validated equivalent SDPA',
            loss='Answer token plus EOS, shifted causal CE, other targets ignored; same two supervised tokens/example in both encodings.',
            length_schedule='Uniform random permutations of all 21 lengths, repeat then truncate to20k; final length counts differ by at most1.',
            batch_schedule='64 distinct anchors sampled per update from that length; saved batch_indices.npy shared by all32 models of a unit.',
            validation_every=1000,checkpoint_every=1000,endpoint='Fixed final20k update checkpoint; no best-checkpoint or early-stopping selection.',
            execution='Single GPU, serial jobs; no concurrent step0 creation. Resume only when protocol/data/start hashes agree.'),
        initialization=dict(original_step0='One real post-constructor-and-weight-tying CPU snapshot/unit, before optimizer or training forwards; all32 models load its exact parameters.',
            task_start_rng='One real task-specific pre-forward RNG snapshot shared by its paired encodings; parameters trace to unit step0; independent task-dropout seeds fixed above.',
            evidence=['actual weights','canonical parameter/buffer hash','file SHA256','Python/NumPy/CPU/CUDA RNG states','full188-token mapping','constructor','frozen code hashes','environment','checkpoint links'],
            limitation='Same exact weights do not imply equal initialization geometry across encodings; do not promise bitwise trained trajectories across CUDA/software versions.'),
        units=units,training_manifest=jobs,historical_exclusions=exclusions(),
        evaluation=dict(landmark='final_norm at actual encoded-input end token, strictly before task; task query readout used only for output/accuracy controls.',
            feature_dtype='FP32 evaluation, no autocast, TF32 disabled; float64 CKA kernels.',length_weights='Fixed1/21, complete all21 or mark unit endpoint missing; never renormalize by dropping undefined lengths.',
            primary_relations=['descent_inverse','exceedance_inverse'],primary_metric='ordinary linear CKA',primary_control='common gold answer strata',
            strata='(n,left answer on identity,right answers on identity/complement/inverse), retain count>=3; subtract within-stratum mean separately; identical anchors across encodings/conditions.',
            primary_contrast='Gamma_Delta = Delta(canonical_cycle) - Delta(one_line), paired within unit; two-sided mean test.',
            decomposition=['initial/trained correct, wrong, identity CKA','B0=initial_correct-initial_wrong','BT=trained_correct-trained_wrong',
                'delta_correct','delta_wrong','Delta=BT-B0=delta_correct-delta_wrong',
                'Gamma_initial','Gamma_trained','Gamma_correct','Gamma_wrong','Gamma_Delta=Gamma_trained-Gamma_initial=Gamma_correct-Gamma_wrong'],
            required_controls=['All8 relations and raw within-length ordinary CKA','true-answer scalar/one-hot geometry, zero variance remains undefined','model numeric probability geometry and its full paired decomposition',
                'full-vocabulary task accuracy and numeric-conditional accuracy per length/action','fixed token-position categorical input kernel',
                'underlying permutation-matrix object input kernel','U-centered metric sensitivity','original single-object-Gram regression/PSD clipping sensitivity'],
            no_extra_analysis='No kernel/length/metric/task search, no output-minus-hidden mechanism calculation, no new geometry loss or combination/LIS intervention; no task-count/larger-model extension.',
            source_failures='Report weak task learning rather than drop it. Missing checkpoints: do not issue a complete primary result. Undefined unit endpoints: report missing and no full planned-sample test; do not add units or replace data.',
            interpretability='Delta sign does not diagnose learning; Gamma_Delta sign or significance does not imply BT flips. B0 and BT are co-reported; components are algebraic not causal percentages.'),
        statistics=dict(independent_unit='One jointly resampled source initialization/data-world/batch stream; fixed task domain and relations are not independent universes.',
            primary_family=2,alpha_family=.05,test='Paired unit effects, two-sided Student t; Bonferroni alpha=.025 each.',
            primary_interval='Simultaneous95% across the two primary Gamma_Delta relations, t quantile .9875.',
            auxiliary_intervals='Pointwise95% descriptive for Gamma_initial/Gamma_trained/correct/wrong components. No joint10-contrast guarantee or equivalence claim.',
            source_count=N,precision_target_halfwidth=target,planning_unit_effect_sd=sigma,planning_halfwidth=halfwidth(N,sigma),
            variance_status='Prospective planning scenario, not an empirically established upper bound; not estimated from an unobserved encoding contrast or borrowed F17 variance.',
            design_policy='FixedN selected before any new neural outcome; no adaptive source expansion or stopping based on signs/significance. Actual precision may miss target.',
            assumption='Unit-effect t inference requires adequate distributional behavior; report estimates/distribution, not a distribution-free guarantee.'),
        completion_rule='For this requested stage stop after code, tests, frozen protocol, manifests and cost report. Do not invoke prepare/train/evaluate on formal units.')
    json_write(configpath,config);json_write(root/'vocabulary.json',dict(TOKEN_TO_ID));write_csv(root/'training_manifest.csv',jobs)
    precision=[]
    for assumption in [.05,.10,.15,.20,.30]:
        needed=next(n for n in range(8,1025) if halfwidth(n,assumption)<=target)
        precision.append(dict(planning_sd=assumption,target_halfwidth=target,units_needed=needed,models_needed=needed*32,
            halfwidth_at_8=halfwidth(8,assumption),halfwidth_at_final_N=halfwidth(N,assumption)))
    write_csv(root/'precision_sensitivity.csv',precision)
    timing=benchmark['uniform_length_mean_update_seconds'];core_per_unit=16*20000*sum(timing.values())/3600
    inference_per_unit=32*config['data']['examples_per_length']['test']*21*3*sum(profile['gpu_forward_seconds_per_sequence'].values())/3600
    psd_per_unit=8*21*2*2*2*4*profile['cpu_psd_adjust_seconds_mean']/3600
    parameters=benchmark['parameters'];Ntest=21*512*3
    disk_per_unit=(32*parameters*4*3+64*Ntest*(256*4+31*4+8)+parameters*4)/2**30
    costs=[]
    for count,label in [(8,'requested_eight_unit_cost_baseline'),(N,'precision_based_fixed_final_plan')]:
        core=count*core_per_unit;inf=count*inference_per_unit;psd=count*psd_per_unit
        costs.append(dict(scenario=label,units=count,models=count*32,optimizer_updates=count*32*20000,core_gpu_hours=core,
            estimated_gpu_inference_hours=inf,estimated_cpu_psd_hours=psd,
            expected_single_gpu_hours_with25pct_training_overhead=core*1.25+inf,
            cautious_single_gpu_hours_with50pct_training_overhead=core*1.5+inf,
            uncompressed_checkpoint_and_feature_GiB=count*disk_per_unit,
            unmeasured_historical_index_data_preparation_and_other_analysis_reserve_hours='1..8 CPU hours, provisional allowance; not directly measured'))
    write_csv(root/'cost_estimate.csv',costs)
    codefiles=list((WORKSPACE/'experiments/permworld_encoding').glob('*.py'))+[
        WORKSPACE/'experiments/longrun_attention.py',WORKSPACE/'experiments/permutation_audit.py',WORKSPACE/'experiments/specialist_input_adjustment.py',
        UPSTREAM/'neurips_permutations/models.py',UPSTREAM/'neurips_permutations/passage.py',UPSTREAM/'neurips_permutations/math_ops.py',
        WORKSPACE/'external/neurips/configs/property_task_geometry.toml']
    # Capture transitive local imports as well as directly used functions.
    # Merely imported F17 helpers remain read-only dependencies.
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if not filename:continue
        path=Path(filename)
        if path.suffix=='.py' and (path.is_relative_to(WORKSPACE/'experiments') or path.is_relative_to(UPSTREAM)):
            codefiles.append(path)
    codefiles=set(codefiles)
    hashes={str(p.relative_to(WORKSPACE)):sha(p) for p in sorted(codefiles)}
    json_write(root/'freeze_manifest.json',dict(frozen_utc=config['registered_utc'],config_sha256=sha(configpath),code_sha256=hashes,
        vocabulary_sha256=sha(root/'vocabulary.json'),training_manifest_sha256=sha(root/'training_manifest.csv'),
        validation_sha256=sha(root/'encoding_validation.json'),integration_checks_sha256=sha(root/'integration_checks.json'),
        timing_sha256=sha(root/'benchmark.json'),analysis_profile_sha256=sha(root/'analysis_profile.json'),
        workspace_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKSPACE,text=True).strip(),
        original_upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKSPACE/'external/neurips',text=True).strip(),
        source_capture='New uncommitted code is frozen by exact file hashes; no historical training commit is inferred.',
        environment=dict(torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),cuda=torch.version.cuda),
        formal_training_started=False))
    report(root,config,costs,precision)
    print({'protocol':'frozen','units':N,'models':len(jobs),'precision_assumed_sd':sigma,'planned_halfwidth':halfwidth(N,sigma),'costs':costs},flush=True)


def report(root,config,costs,precision):
    N=len(config['units']);base,final=costs
    text=f'''PermWorld 一行记号 vs 规范化轮换记号：第一、二步交付

状态：实现、验证、成本估算与正式协议冻结已完成；尚未生成正式数据或启动正式训练。

两种编码：一行记号表示 pi(1)..pi(n)；规范轮换沿 pi 的映射方向，每轮换从最小值开始、按轮换最小值排序、保留所有单元素轮换。轮换内逗号、轮换间分号，使用仓库已有CYCLE边界token。
共用188-token词表、256维4层8头模型、MLP倍率4、dropout0.1、绑定embedding、max_seq_len128。两种编码都含n个数字和n−1个分隔符：前缀2n+4、含答案/EOS监督序列2n+8，n10..30长度完全相同。

验证：S1..S7穷举加n10..30每长度256随机排列，共11,289个排列；90,312次关系公式检查。两种编码往返、全部16性质标签、正确/原错误变换一致；非法序列拒绝、端点因果性、SDPA等价、参数复制/RNG回放、轨道分割和批次一致性检查通过。

固定方案：每单元一个新模型种子和一个独立新数据采样世界，两编码/16性质共32单任务模型，共享实际step0参数和每任务的起始随机状态。每长度训练4,096、验证256、测试512个anchor，21个长度固定均匀；单元内分割八元轨道严格互斥，全部成员排除固定历史输入索引。不同单元独立采样，不强制互斥；不做跨单元模型对比较。
每模型20,000更新、micro16×累积4=有效batch64；共同保存批次表、同样1,280,000样本曝光/2,560,000答案+EOS监督token。AdamW lr3e-4、wd0.01、warmup1000、cosine到0.1×lr、clip1、BF16训练/FP32评估、TF32禁用。固定最终20k检查点，不按验证CKA/准确率选择或延长。

主要检验：下降—逆下降、超越—亏缺两条关系的Gamma_Delta=Delta(轮换)−Delta(一行)，答案共同分层后的普通CKA；每长度分别计算后等权1/21。两关系Bonferroni，单位配对t检验，同时95%区间。共同报告Gamma_initial、Gamma_trained、正确/错误变化、所有八关系、源准确率、真实答案和输出概率基线；三指标与两个固定输入核只作预定敏感性。
实际step0权重、参数哈希、文件SHA、构造器、完整词表和Python/NumPy/CPU/CUDA随机状态会在正式训练前保存并链接每个检查点。初始化参数相同不意味着编码后的初始化几何相同。

数量选择：八单元/256模型先作成本基线。精度目标为两主要Gamma_Delta同时95%区间计划半宽±{config['statistics']['precision_target_halfwidth']:.2f}；尚无跨编码方差数据，预先采用标准差{config['statistics']['planning_unit_effect_sd']:.2f}的规划情景，固定{N}单元/{N*32}模型，计划半宽{config['statistics']['planning_halfwidth']:.4f}。0.20不是已验证的方差上界，本方案不是精度保证，不借用旧单编码/F17方差，也不设结果驱动扩容。若真实标准差更大，就如实报告较宽区间。

本机RTX5070微基准（仅零学习率前后向和AdamW核计时，模型参数哈希未变）：
  八单元：核心训练{base['core_gpu_hours']:.1f} GPU小时；按25%/50%训练开销预留及实测推理外推，共约{base['expected_single_gpu_hours_with25pct_training_overhead']:.1f}–{base['cautious_single_gpu_hours_with50pct_training_overhead']:.1f} GPU小时。
  正式{N}单元：核心训练{final['core_gpu_hours']:.1f} GPU小时；含上述预留及推理外推约{final['expected_single_gpu_hours_with25pct_training_overhead']:.1f}–{final['cautious_single_gpu_hours_with50pct_training_overhead']:.1f} GPU小时，另预留未实测历史索引/数据准备/其余分析1–8 CPU小时。
  当前计划检查点+特征未压缩估计约{final['uncompressed_checkpoint_and_feature_GiB']:.1f} GiB；建议预留120GiB。计时不是长时热稳态或共享GPU负载保证，GPU/CPU时间不能全部简单并行抵消。

完整精度和成本敏感性：precision_sensitivity.csv、cost_estimate.csv。训练清单：training_manifest.csv。完整协议：protocol.json；冻结哈希：freeze_manifest.json。
代码：experiments/permworld_encoding/ 下的core/data/train/evaluate模块；README.txt提供复现和未来显式执行命令。本轮停止在准备阶段，不自动调用正式prepare/train/evaluate。
'''
    (root/'report.txt').write_text(text)
    (root/'README.txt').write_text('''从 /home/yangx/ICML 执行；PYTHONDONTWRITEBYTECODE=1 防止修改旧模块的缓存。

已完成的准备验证（不启动正式训练）：
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.validate --output results/permworld_encoding_plan_20261009/encoding_validation.json
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.checks --config results/permworld_encoding_plan_20261009/planning_candidate.json --output results/permworld_encoding_plan_20261009/integration_checks.json

未来正式执行（本轮没有执行以下命令；不能把微基准当作正式训练）：
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.data --config results/permworld_encoding_plan_20261009/protocol.json
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.train --config results/permworld_encoding_plan_20261009/protocol.json --run-id u001_one_line_descents
  按training_manifest.csv中的run_id串行执行全部模型；实际字母顺序见清单，不按成绩重排/删减。
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.evaluate --config results/permworld_encoding_plan_20261009/protocol.json --unit u001
  所有单元预定模型完成后，每单元显式评估一次，然后：
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.evaluate --config results/permworld_encoding_plan_20261009/protocol.json --summarize

正式数据/step0/训练检查点仅写results/permworld_encoding_confirmation_v1，不写F17或旧诊断目录。模型若中断，用完全相同命令续跑；协议/数据/代码哈希不匹配会拒绝续跑。正式协议不可覆盖；修改须建立显式新版本并重新冻结。
主要结果：unit_components.csv、encoding_interactions.csv、primary_statistics.csv；附带baseline_unit_components.csv、baseline_encoding_interactions.csv、task_accuracy_summary.csv，以及逐长度各表。
缺失/未定义单元不悄悄剔除，不自动补模型或增加样本。表征指标分量不是因果贡献比例；不以隐藏Delta减概率Delta定义机制。
''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/permworld_encoding_plan_20261009');p.add_argument('--precision',type=float,default=.10);p.add_argument('--planning-sd',type=float,default=.20)
    a=p.parse_args();create(a.output,a.precision,a.planning_sd)
