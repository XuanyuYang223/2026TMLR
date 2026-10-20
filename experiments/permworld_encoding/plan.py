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
        registered_utc=datetime.now(timezone.utc).isoformat(),research_question='How does input encoding affect training-induced reshaping of transformation-pairing selectivity?',
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
    text=f'''PermWorld One-line notation vs Canonical cycle notation: first- and second-stage deliverables

Status: implementation, checks, cost estimate, and formal protocol freeze are complete; formal data have not been generated and formal training has not started.

Two encodings: one-line notation represents  pi(1)..pi(n); canonical cycles follow  pi  mapping direction, start each cycle at its minimum, order cycles by their minima, and retain singletons. Use commas within cycles and semicolons between cycles, with the repository's existing CYCLE boundaries token.
Shared 188-token vocabulary, 256-dimensional 4 layers, 8 heads, MLP ratio 4, dropout0.1, tied embedding, max_seq_len128. Both encodings contain n numbers and n−1 delimiters: prefix 2n+4, answer/EOS supervision sequence 2n+8,n10..30 have identical lengths.

Validation: S1..S7 exhaustive enumeration plus n10..30, per length, 256 random permutations, totaling 11,289 permutations; 90,312 relationship-identity checks. Round trips, all 16 property labels, and correct/original wrong transformations agree between encodings; malformed-sequence rejection, endpoint causality, SDPA equivalence, parameter-copy/RNG replay, orbit-split, and batch-consistency checks passed.

Fixed design: a new model seed and independent new data world per unit; two encodings x /16 properties give 32 specialists sharing actual step0 parameters and per-task starting random states. Per-length training 4,096, validation 256, test 512 anchor,21; uniform fixed length sampling. Eight-element orbits are disjoint across splits within units and exclude all historical-index inputs. Units are independently sampled, not forced disjoint; no cross-unit model-pair comparisons.
Per model: 20,000 updates, micro16 x accumulation 4= effective batch64; common saved batch schedules, identical 1,280,000 example exposures/2,560,000 Answer+EOS supervised token.AdamW lr3e-4, wd0.01, warmup1000, cosine to 0.1×lr, clip1, BF16 training/FP32 evaluation, TF32 disabled. Fixed final 20k checkpoint, not selected/extended by validation CKA/ accuracy.

Primary tests: descents--inverse descents and exceedances--deficiencies Gamma_Delta=Delta(cycle)−Delta(one-line), ordinary CKA after joint-answer stratification; separately computed lengths with equal weight 1/21. Two relationships Bonferroni, unit-paired t tests, simultaneous 95% intervals. Jointly report Gamma_initial, Gamma_trained, correct/wrong changes, all eight relationships, source accuracies, gold-answer/output-probability references; three metrics and two fixed input kernels are prescribed sensitivities only.
Actual step0 weights, parameter hashes, file SHA, constructor, full vocabulary, and Python/NumPy/CPU/CUDA random states will be saved before formal training and linked to every checkpoint. Identical initial parameters do not imply identical encoded-input initialization geometry.

Sample-count choice: eight units/256 models first serve as a cost baseline. The precision target for the two primary Gamma_Delta simultaneous 95% intervals is a planned half-width ±{config['statistics']['precision_target_halfwidth']:.2f}; cross-encoding variance is unavailable, so planning assumes SD {config['statistics']['planning_unit_effect_sd']:.2f} and fixes {N} units//{N*32} models, planned half-width {config['statistics']['planning_halfwidth']:.4f}.0.20 is not a verified variance bound. This is not a precision guarantee, does not borrow old single-encoding /F17 variance, and has no result-driven expansion. Larger actual SDs will be reported as wider intervals.

Local RTX5070 microbenchmark (zero-learning-rate forward/backward and AdamW kernel timing only; model-parameter hashes unchanged):
  Eight units: core training {base['core_gpu_hours']:.1f} GPU hours; allowing 25%/50% training overhead plus extrapolated measured inference, approximately {base['expected_single_gpu_hours_with25pct_training_overhead']:.1f}–{base['cautious_single_gpu_hours_with50pct_training_overhead']:.1f} GPU hours.
  Formal {N} units: core training {final['core_gpu_hours']:.1f} GPU hours; with the same overhead/inference estimate, approximately {final['expected_single_gpu_hours_with25pct_training_overhead']:.1f}–{final['cautious_single_gpu_hours_with50pct_training_overhead']:.1f} GPU hours, plus unmeasured historical-index//data-preparation//other-analysis reserve 1–8 CPU hours.
  Current planned checkpoint/+feature uncompressed estimate: approximately {final['uncompressed_checkpoint_and_feature_GiB']:.1f} GiB; reserve 120GiB. Timings do not guarantee long-run thermal steady state or shared GPU load, and GPU/CPU time cannot all be canceled through simple parallelism.

Full precision/cost sensitivities: precision_sensitivity.csv, cost_estimate.csv. Training manifest: training_manifest.csv. Full protocol: protocol.json; freeze hash: freeze_manifest.json.
Code: experiments/permworld_encoding/  modules under core/data/train/evaluate; README.txt supplies reproduction and future explicit execution commands. This round stops at preparation and does not automatically invoke formal prepare/train/evaluate.
'''
    (root/'report.txt').write_text(text)
    (root/'README.txt').write_text(''' from  /home/yangx/ICML execution; PYTHONDONTWRITEBYTECODE=1 prevents modification of legacy-module caches.

Completed preparation checks (no formal training):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.validate --output results/permworld_encoding_plan_20261009/encoding_validation.json
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.checks --config results/permworld_encoding_plan_20261009/planning_candidate.json --output results/permworld_encoding_plan_20261009/integration_checks.json

Future formal execution (commands below were not executed in this round; microbenchmarks are not formal training):
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.data --config results/permworld_encoding_plan_20261009/protocol.json
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.train --config results/permworld_encoding_plan_20261009/protocol.json --run-id u001_one_line_descents
  Following training_manifest.csv in run_id, execute every model serially in recorded alphabetical order, without performance-based reordering/deletion.
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.evaluate --config results/permworld_encoding_plan_20261009/protocol.json --unit u001
  After all prespecified models complete, explicitly evaluate each unit once, then:
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.permworld_encoding.evaluate --config results/permworld_encoding_plan_20261009/protocol.json --summarize

Formal data/step0/training checkpoints are written only to results/permworld_encoding_confirmation_v1, not F17 or old diagnostic directories. Resume interruptions using the identical command; mismatched protocol/data/code hashes reject resumption. Never overwrite the formal protocol; changes require an explicit new version and refreeze.
Primary results: unit_components.csv, encoding_interactions.csv, primary_statistics.csv; accompanied by baseline_unit_components.csv, baseline_encoding_interactions.csv, task_accuracy_summary.csv and all per-length tables.
missing/undefined units are not silently dropped or automatically replaced/expanded. Representation-metric components are not causal contribution percentages; hidden Delta minus probability Delta does not define a mechanism.
''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/permworld_encoding_plan_20261009');p.add_argument('--precision',type=float,default=.10);p.add_argument('--planning-sd',type=float,default=.20)
    a=p.parse_args();create(a.output,a.precision,a.planning_sd)
