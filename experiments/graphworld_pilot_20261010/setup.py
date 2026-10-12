"""CPU-only setup, audit and protocol freeze before any new-model result."""
import copy
from datetime import datetime,timezone
import shutil
import time
from .common import *
from .data import mathematics,prepare

def setup():
    if CONFIG.exists():return verify()
    DIAG.mkdir(parents=True,exist_ok=True);ROOT.mkdir(parents=True,exist_ok=True)
    original=read(SOURCE_CONFIG);old_freeze=read(SOURCE_ROOT/'freeze_manifest.json')
    assert sha(SOURCE_CONFIG)==old_freeze['protocol_sha256']
    dependencies={str((WORKSPACE/p).resolve()):h for p,h in old_freeze['source_sha256'].items()}
    for p,h in dependencies.items():assert sha(p)==h,('Original source changed',p)
    src_init=SOURCE_ROOT/'initializations/transformer/step0.pt';src_rng=SOURCE_ROOT/'initializations/transformer/start_rng_descents.pt'
    init_meta=read(src_init.with_suffix('.json'));saved=torch.load(src_init,map_location='cpu',weights_only=False)
    rs=torch.load(src_rng,map_location='cpu',weights_only=False)
    assert sha(src_init)==init_meta['checkpoint_sha256'] and init_meta['actual_initial_identity_verified']
    assert saved['model_config']==original['model'] and saved['vocabulary']==TOKEN_TO_ID
    assert rs['parameter_sha256']==init_meta['parameter_sha256'] and rng_hash(rs['rng'])==rs['rng_hash']
    init_dir=ROOT/'initializations';init_dir.mkdir(exist_ok=True)
    shutil.copyfile(src_init,init_dir/'step0.pt');shutil.copyfile(src_rng,init_dir/'start_rng.pt')
    assert sha(src_init)==sha(init_dir/'step0.pt') and sha(src_rng)==sha(init_dir/'start_rng.pt')
    math_audit=mathematics()
    fixed_artifacts=[SOURCE_CONFIG,SOURCE_ROOT/'freeze_manifest.json',src_init,src_init.with_suffix('.json'),src_rng]
    tf_records=[]
    for p in sorted((SOURCE_ROOT/'models').glob('archdev001_transformer_*/completed.json')):
        r=read(p);assert r['step']==20000 and r['execution_source_version_sha256']=='d8fc9e153e45c15f2ad82997fdf34bf4988c92f0fa6a78afdc70aa454d8bdc3a'
        tf_records.append(dict(run_id=r['run_id'],elapsed_seconds=r['elapsed_seconds'],updates=r['step']))
        fixed_artifacts.append(p)
    assert len(tf_records)==8
    cfg=dict(protocol_version='graphworld-fixed-four-task-pilot/v1',registered_utc=datetime.now(timezone.utc).isoformat(),
        exploratory=True,worlds=1,reference_architecture='transformer',encodings=['row_major_offdiagonal_binary'],
        lengths=[6,7,8],length_weights={str(n):1/3 for n in (6,7,8)},tasks=list(TASKS),actions=list(ACTIONS),relations=list(RELATIONS),
        source_root=str(SOURCE_ROOT),source_protocol_sha256=sha(SOURCE_CONFIG),
        original_execution_source_sha256='d8fc9e153e45c15f2ad82997fdf34bf4988c92f0fa6a78afdc70aa454d8bdc3a',
        source_artifacts_sha256={str(p):sha(p) for p in fixed_artifacts},model=copy.deepcopy(original['model']),
        model_seed=original['architecture_model_seeds']['transformer'],training=copy.deepcopy(original['training']),
        initial_checkpoint_path=str(init_dir/'step0.pt'),initial_checkpoint_sha256=sha(init_dir/'step0.pt'),
        initial_parameter_sha256=init_meta['parameter_sha256'],actual_initial_identity_verified=True,
        starting_rng_path=str(init_dir/'start_rng.pt'),starting_rng_sha256=sha(init_dir/'start_rng.pt'),starting_rng_hash=rs['rng_hash'],
        starting_rng_origin='Actual archived TF descents starting Python/NumPy/CPU/CUDA states, copied once and shared across all4 new graph tasks; no same-seed identity assumption.',
        data_seed=2026101001,batch_seed=2026101002,data=dict(per_n={'train':4096,'validation':256,'analysis':512},p=[.25,.5,.75],
             sampling='Uniform p-category draw, independent off-diagonal Bernoulli edges, global full<R,C,W> orbit rejection in train/validation/analysis order; all accidental wrong-label matches retained.'),
        wrong_tables=math_audit['records'],mathematics_audit_sha256=sha(DIAG/'mathematics_audit.json'),
        serialization=dict(prefix='BOS SIZE n ARG_START; row-major non-diagonal00/01, existing| between rows; ARG_END',
             vocabulary=dict(TOKEN_TO_ID),task_tokens=TASK_TOKENS,task_ids={k:TOKEN_TO_ID[v] for k,v in TASK_TOKENS.items()},
             end_position='n*n+3 before task',answer_query_position='n*n+5 at equals',
             prefix_lengths={str(n):n*n+4 for n in (6,7,8)},supervised_lengths={str(n):n*n+8 for n in (6,7,8)},
             reused_token_semantics='Graph-specific serializer and task token meanings; original permutation parser untouched. Existing random token IDs are reinterpreted, no new vocabulary.'),
        evaluation=dict(fit_split='validation',test_split='analysis',batch=128,primary_state='final_norm/input-end before task',
             model_mode='eval',autocast=False,dtype='float32',TF32=False,
             controls=['raw_within_n','answer_strata','answer_density_strata'],minimum_stratum_count=3,
             strata='Joint[n,left identity,mapped right identity/correct/wrong], independently>=3 in fit/test; density adds exact anchor edge count m. Masks fixed across all models/branches.',
             metrics=['linear_cka','knn_overlap','procrustes'],sensitivity=['debiased_cka'],knn_k=10,procrustes_rank_cap=8,
             pooling='input-end only',input_kernel='Position-wise binary two-class onehot, exact common Gram under fixed slot/category isometries; static input reference, no fabricated initialization/training Delta.',
             procrustes_similarity='negative held-out RMS normalized error; larger is more similar',
             baselines=['numeric0..30 conditional output probabilities','gold answer onehot31','gold scalar','fixed positional binary onehot input'],
             fields=['initial_correct','initial_wrong','initial_identity','trained_correct','trained_wrong','trained_identity','B0','Bt','dc','dw','di','Delta'],
             undefined_policy='NaN/None retained; fixed1/3 aggregate undefined if any n missing, never reweighted.'),
        engineering=dict(real_learning_updates=32,resume_from=16,resume_until=32,parameter_tolerance=1e-5,optimizer_tolerance=1e-5,
             activation_tolerance=2e-5,logit_tolerance=1e-4,task_learning_not_a_gate=True,geometry_not_a_gate=True,
             samples=48,supervision='Independent mask/shifted CE, finite nonzero gradient, changed parameter, task/EOS scores retained; no accuracy threshold.'),
        limits=dict(source_models=4,source_endpoint=20000,no_tuning=True,no_added_seeds=True,no_added_tasks=True,
             no_GroupWorld=True,no_F17_changes=True,no_automatic_training=True,stop_after_report=True),
        environment=dict(torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,networkx=math_audit['networkx_version']),
        limitations=['Four relationships share four tasks and two transformations; not4 independent mathematical replications.',
             'One graph world and one shared initialization; configuration-specific exploratory transfer across systems.',
             'Vertex relabeling/isomorphism classes are not excluded, only the specified<R,C,W> orbits.',
             'Different size/encoding/task/distribution from PermWorld; no domain-only causal contrast.'],
        training_manifest=[dict(run_id='graphpilot001_'+t,task=t,architecture='transformer',encoding='row_major_offdiagonal_binary',updates=20000) for t in TASKS])
    cfg['training']['length_schedule']='Uniform permutations of n6/7/8, repeated/truncated to20k; final counts differ by<=1.'
    cfg['training']['batch_schedule']='Fixed64 distinct graph anchors at the scheduled n; same saved table and start RNG for all4 models.'
    json_write(CONFIG,cfg);json_write(ROOT/'protocol.json',cfg)
    paths={**dependencies,**{str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py')}}
    paths[str(WORKSPACE/'diagnostics/permworld_layers_20261009/input_geometry.py')]=sha(WORKSPACE/'diagnostics/permworld_layers_20261009/input_geometry.py')
    json_write(DIAG/'freeze_manifest.json',dict(protocol_sha256=sha(CONFIG),source_sha256=paths,new_source_training_started=False))
    protection=read(WORKSPACE/'diagnostics/permworld_controlled_comparison_20261010/protected_before.json')
    json_write(DIAG/'protected_files.json',dict(files=protection,upstream_manifest_sha256=sha(WORKSPACE/'diagnostics/permworld_controlled_comparison_20261010/protected_before.json'),
             policy='Original F17, PermWorld and dependent old files retained unchanged; never delete or rewrite.'))
    csv_write(DIAG/'training_manifest.csv',cfg['training_manifest'])
    med=float(np.median([r['elapsed_seconds'] for r in tf_records]));old_lengths=np.arange(10,31)*2+8;new_lengths=np.array([6,7,8])**2+8
    ratios={'linear_sequence_ratio':float(new_lengths.mean()/old_lengths.mean()),'quadratic_attention_ratio':float(np.square(new_lengths).mean()/np.square(old_lengths).mean())}
    cost=dict(original_tf_20k_records=tf_records,median_recorded_process_seconds=med,sequence_ratios=ratios,
        nominal4_source_seconds=4*med,range_scaled4_source_seconds=[4*med*min(ratios.values()),4*med*max(ratios.values())],
        scope='Rough extrapolation of original20k TF process wall including validation; not short-length8k timing, GPU contention and graph serialization may change actual cost.',
        pending_graph_gpu_benchmark=True,source_parameter_count=init_meta['parameters'],checkpoint_budget_bytes=4*init_meta['parameters']*4*3,
        nominal_feature_cache_bytes=4*2*(256+512)*3*5*(256+31)*4,temporary_model_max=1,new_source_models=4)
    json_write(DIAG/'cost_estimate.json',cost)
    (DIAG/'formulas_and_controls.txt').write_text('''Exact graph formulas and negative control\n\nA_ij=1 means i->j; diagonal0. Outdegree d_out(i)=sum_j A_ij; indegree d_in(i)=sum_j A_ji. R(A)=A^T, so d_out^A(i)=d_in^R(A)(i). Maxima and minima over the same vertex set give both R relations exactly. C(A)_ij=1-A_ij for i!=j and0 otherwise, so d_out^C(A)(i)=n-1-d_out^A(i), and similarly for indegrees. Taking minima yields max_out(A)=n-1-min_out(C(A)) and max_in(A)=n-1-min_in(C(A)).\n\nW is a fixed permutation of unordered vertex pairs, built by Random(20261010+n) shuffle and adjacent swaps; odd last pair stays fixed. Both oriented slots of a pair move together, preserving the low->high slot orientation. Thus W^2=Id; R^2=C^2=Id; W commutes with R because both orientations move together, W commutes with C because C complements every non-diagonal slot, and R commutes with C. They generate at most2^3=8 transformed arrays. W preserves edge count. WR/WC therefore match their respective R/C controls in n and edge count.\n\nAny vertex-induced permutation of unordered pairs preserves pair intersections. mathematics_audit.json supplies explicit original intersecting pairs whose W images are disjoint for n6/7/8, proving this W is not a vertex relabeling. It also supplies concrete adjacency counterexamples for every wrong relation at each n, and the unselected random samples' accidental-match fractions. Accidental matches are retained in training/evaluation; no W seed is selected by outcome.\n\nThe fixed positional binary two-class-onehot kernel keeps vertex-pair positions. R and W permute slots; C exchanges00/01 coordinates in every slot, hence all act as orthogonal coordinate permutations. Centering globally or within the identical fixed strata commutes with that fixed coordinate map; fit/test centered self and cross Grams are unchanged. The input reference cannot distinguish correct versus wrong here. Constant size/boundary positions vanish under within-n centering.\n\nNetworkX local3.6.1 independently checked A_ij direction, reverse and loopless complement. Official reference pages inspected without changing environment:\nhttps://networkx.org/documentation/stable/reference/classes/generated/networkx.DiGraph.reverse.html\nhttps://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.unary.complement.html\nhttps://networkx.org/documentation/stable/reference/generated/networkx.linalg.graphmatrix.adjacency_matrix.html\n\nGraph-specific task labels reinterpret four fixed original token slots at random step0; no semantic transfer from a trained PermWorld model is claimed. Original actual TF constructor checkpoint and archived descents starting random states are copied byte-for-byte and shared across all four tasks.\n''')
    (DIAG/'REPRODUCE.txt').write_text('''Run from /home/yangx/ICML. Explicit stages only; preparation never cascades into training.\n\npython -m experiments.graphworld_pilot_20261010 --stage prepare\npython -m experiments.graphworld_pilot_20261010 --stage verify\n\nRoot schedules the fixed small engineering check; no formal graph fitting before PermWorld train+analysis finish:\npython -m experiments.graphworld_pilot_20261010 --stage engineering\n\nOnly after root starts the four-job fixed manifest:\npython -m experiments.graphworld_pilot_20261010 --stage train-all\npython -m experiments.graphworld_pilot_20261010 --stage evaluate\npython -m experiments.graphworld_pilot_20261010 --stage report\n\nEach final source is20k, no best-checkpoint choice or task learning/geometry gate. The engineering paths do not produce formal completed endpoints. Recovery may explicitly call train-one with run_id graphpilot001_<task> and fixed stop-step<=20000; --output is restricted to the new engineering subtree. All sources, data, true initial and start RNG are hash-bound. Freeze/protected originals are not rewritten. report writes completion=completed_and_stopped and starts no followup.\n''')
    return cfg

def prepare_all():
    c=setup();audit=prepare(c)
    from .checks import cpu_checks
    checks=cpu_checks(c)
    json_write(DIAG/'cpu_ready.json',dict(status='passed',protocol_sha256=sha(CONFIG),data_sha256=audit['data_sha256'],
              source_version_sha256=source_version(),cpu_checks=checks,new_source_models=0,gpu_training_started=False))
    return c
