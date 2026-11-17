"""Read-only frozen-source inventory and named endpoint protection snapshot.

The snapshot follows the authorized historical validation engine preflight,
and precedes this task's formal feature/readout fitting. It does not claim
protection began before the earliest read-only development forward.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PREVIOUS = ROOT/'diagnostics/permworld_lemma_composition_20261110/cache_audit.json'
REP = ROOT/'diagnostics/permworld_repetition_20261106/protocol.json'
MODEL_SHA = '6942c768e441235ac657949b43514c9ba5d0f2b5900c9bd89743fffce28b700b'
VOCAB_SHA = 'd200ffe6f5f6cff717c1b6ed6698a5a6af42bb6580fa4e8729a2fd84b79babd6'


def read(path): return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def state_sha(state):
    result = hashlib.sha256()
    for name,value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        for item in (name,str(array.dtype),str(array.shape)): result.update(item.encode())
        result.update(array.tobytes())
    return result.hexdigest()


def write(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(path)


def main():
    if (HERE/'protection_start.json').exists():
        # Preserve the first before-fitting snapshot on any later invocation.
        # A rerun is verification, never a newly dated replacement start.
        original = read(HERE/'protection_start.json')
        if sha(HERE/'source_audit.json') != original['source_audit_sha256']:
            raise SystemExit('The existing source audit differs from its original snapshot')
        changed = [path for path,digest in original['files'].items()
                   if not Path(path).is_file() or sha(path) != digest]
        if changed:
            raise SystemExit(f'Existing snapshot verification failed: {changed}')
        print(json.dumps(dict(status='passed_existing_snapshot_preserved',files=len(original['files']))),flush=True)
        return
    began = time.monotonic(); now = datetime.now(timezone.utc).isoformat()
    previous, repetition = read(PREVIOUS), read(REP)
    model = repetition['model']
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status = subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True)
    tracked = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    expected, roles, missing, failures, units = {}, {}, [], [], []
    count = 0

    def check(value,label,detail=None):
        nonlocal count
        count += 1
        if not bool(value): failures.append(dict(label=label,detail=detail))

    def add(path,digest=None,role='source'):
        path = str(Path(path))
        if not Path(path).is_file():
            missing.append(dict(path=path,role=role)); return False
        value = digest or sha(path)
        if path in expected: check(expected[path] == value,'consistent source SHA expectation',path)
        expected[path] = value; roles.setdefault(path,set()).add(role)
        return True

    check(model == dict(d_model=256,layers=4,heads=8,mlp_ratio=4,dropout=.1,
                       tie_embeddings=True,max_seq_len=128,vocab_size=188),'exact frozen old model configuration')
    for path,expected_sha,role in [
        (PREVIOUS,None,'prior_source_checkpoint_inventory'),(REP,None,'original_model_training_protocol'),
        (ROOT/'experiments/permworld_peaks_probe_20261109/vendor/archdev_common.py',MODEL_SHA,'exact_original_CausalGRU_definition'),
        (ROOT/'experiments/permworld_peaks_probe_20261109/vendor/passage.py',VOCAB_SHA,'exact_original_vocabulary_definition'),
        (ROOT/'experiments/permworld_peaks_probe_20261109/vendor/encoding_core.py',None,'old_token_batch_encoder'),
        (ROOT/'experiments/permworld_encoding/core.py',None,'original_encoding_registry'),
        (ROOT/'experiments/permworld_encoding/data.py',None,'original_world_generator'),
        (ROOT/'experiments/permworld_repetition_20261106/runner.py',None,'original_training_seed_and_sampling_program'),
        (ROOT/'experiments/permutation_audit.py',None,'original_transform_registry'),
        (ROOT/'external/neurips/src/neurips_permutations/math_ops.py',None,'original_property_formulas'),
        (ROOT/'external/neurips/configs/property_task_geometry.toml',None,'original_task_relation_registry'),
        (ROOT/'experiments/permworld_local_causal_20261110/recurrent.py',None,'accepted_new_recurrent_engine'),
        (ROOT/'tests/test_permworld_local_causal_recurrent.py',None,'accepted_engine_invariant_tests')]:
        add(path,expected_sha,role)

    for source_unit in previous['units']:
        name = source_unit['unit']; path = Path(source_unit['data_path'])
        if not add(path,source_unit['data_sha256'],'used_original_world_data'): continue
        check(sha(path) == source_unit['data_sha256'],'actual original data payload SHA',name)
        models = {}
        for task in ('peaks','valleys'):
            prior = next(r for r in source_unit['trained_checkpoints'] if r['task'] == task)
            cp_path,done_path = Path(prior['path']),Path(prior['metadata_path'])
            if not add(cp_path,prior['sha256'],'used_frozen20k_checkpoint') or not add(done_path,role='original_training_completion_receipt'):
                continue
            actual_cp_sha = sha(cp_path)
            check(actual_cp_sha == prior['sha256'],'actual checkpoint file SHA',[name,task])
            if actual_cp_sha != prior['sha256']: continue
            done = read(done_path)
            checkpoint = torch.load(cp_path,map_location='cpu',weights_only=False)
            check(checkpoint['architecture'] == 'gru','actual GRU checkpoint header',[name,task])
            check(checkpoint['step'] == checkpoint['finite_gradient_updates'] == done['step'] == done['finite_gradient_updates'] == 20000,'actual completed20k updates',[name,task])
            check(checkpoint['data_sha256'] == done['data_sha256'] == source_unit['data_sha256'],'checkpoint original-data binding',[name,task])
            check(checkpoint['run_id'].endswith(f'_gru_one_line_{task}'),'checkpoint specialist/encoding identity',[name,task])
            check(done['checkpoint_sha256'] == prior['sha256'],'completed receipt payload binding',[name,task])
            state = checkpoint['model']; parameter_sha = state_sha(state)
            for old_expected in (prior.get('parameter_sha256'),done.get('parameter_sha256')):
                if old_expected: check(parameter_sha == old_expected,'actual state_dict matches recorded parameter identity',[name,task])
            layer_numbers = sorted({int(k.split('.')[1]) for k in state if k.startswith('recurrent_layers.')})
            check(layer_numbers == [0,1,2,3],'exact four recurrent layers',[name,task])
            shapes = dict(token_embedding=(188,256),position_embedding=(128,256),lm_head=(188,256))
            for key,shape in shapes.items(): check(tuple(state[key+'.weight'].shape) == shape,'embedding/readout shape',[name,task,key])
            check(tuple(state['final_norm.weight'].shape) == (256,) and tuple(state['final_norm.bias'].shape) == (256,),'final_norm width',[name,task])
            for layer in range(4):
                for field in ('weight_ih_l0','weight_hh_l0'):
                    check(tuple(state[f'recurrent_layers.{layer}.{field}'].shape) == (768,256),'three-gate GRU state shape',[name,task,layer,field])
                for field in ('bias_ih_l0','bias_hh_l0'):
                    check(tuple(state[f'recurrent_layers.{layer}.{field}'].shape) == (768,),'GRU bias shape',[name,task,layer,field])
            check(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in state.values()),'finite FP32 frozen parameters',[name,task])
            check(torch.equal(state['lm_head.weight'],state['token_embedding.weight']),'tied readout/embedding values',[name,task])
            models[task] = dict(checkpoint_path=str(cp_path),checkpoint_sha256=prior['sha256'],
                parameter_sha256=parameter_sha,data_sha256=source_unit['data_sha256'],task=task,encoding='one_line',
                architecture='gru',phase='trained',step=20000,finite_gradient_updates=20000,
                initial_parameter_sha256=checkpoint['initial_parameter_sha256'],
                initial_checkpoint_sha256=checkpoint.get('initial_checkpoint_sha256'),
                source_run_id=checkpoint['run_id'],completion_path=str(done_path),
                prior_parameter_hash_missing=prior.get('parameter_sha256') is None,
                parameter_identity_evidence='Actual current hash-bound saved state_dict; no seed reconstruction or neural module construction')
            del state,checkpoint
        check(set(models) == {'peaks','valleys'},'both required specialists',[name])
        units.append(dict(unit=name,historical=source_unit['historical'],data_path=str(path),
                          data_sha256=source_unit['data_sha256'],models=models))

    # Explicitly enumerate old repository reports/tables/figures/protocol/code.
    # Large external token shards and unused activation archives are excluded.
    scientific_directories = [
        ROOT/'diagnostics/permworld_orthogonal_alignment_20261110',
        ROOT/'diagnostics/permworld_exceedance_alignment_20261110',
        ROOT/'diagnostics/permworld_lemma_composition_20261110',
        ROOT/'experiments/permworld_orthogonal_alignment_20261110',
        ROOT/'experiments/permworld_exceedance_alignment_20261110',
        ROOT/'experiments/permworld_lemma_composition_20261110',
    ]
    suffixes = {'.json','.csv','.md','.txt','.html','.py','.png','.svg','.pdf','.tex','.bib','.zip','.sh','.log'}
    for directory in scientific_directories:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in suffixes and '__pycache__' not in path.parts:
                add(path,role='named_prior_scientific_report_table_figure_protocol_or_code')
    for relative in tracked:
        if relative and (relative.startswith(('docs/','results/','manuscripts/','paper/')) or
                         'f17' in relative.lower() or 'reviewer_revision' in relative.lower()):
            path = ROOT/relative
            if path.is_file(): add(path,role='named_current_tracked_document_result_or_paper_path')
    for relative in ('README.md','CURRENT_RESEARCH.txt'):
        add(ROOT/relative,role='existing_research_entry_document')

    # These old F17 path catalogs are retained as provenance, not misrepresented
    # as proof that relocated/absent historical paper files exist in this checkout.
    optional_unavailable = []
    old_f17_catalog = ROOT/'manuscripts/representation_selectivity_20261012/provenance/old_paper_protection_start.json'
    if old_f17_catalog.is_file():
        for relative in read(old_f17_catalog)['file_SHA256']:
            path = ROOT/relative
            if path.is_file(): add(path,role='named_present_F17_paper_file_from_legacy_catalog')
            else: optional_unavailable.append(str(path))

    def hash_file(item):
        path,digest = item; file = Path(path); before = file.stat(); actual = sha(file); after = file.stat()
        return path,dict(sha256=actual,expected_sha256=digest,size_bytes=after.st_size,
            roles=sorted(roles[path]),passed=actual == digest and (before.st_size,before.st_mtime_ns) == (after.st_size,after.st_mtime_ns))
    with ThreadPoolExecutor(max_workers=2) as pool:
        files = dict(pool.map(hash_file,sorted(expected.items())))
    for path,record in files.items(): check(record['passed'],'named file endpoint hash/stability',path)
    check([u['unit'] for u in units] == ['archdev001','unit02','unit03','unit04','unit05'],'complete five fixed units')
    final_status = 'passed' if not failures and not missing else 'failed'
    audit = dict(status=final_status,captured_utc=now,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Actual file/state_dict and named old-file inventory before new formal feature fitting. Earlier authorized historical validation engine preflight was read-only and separately verified; this is not a before-earliest-forward snapshot.',
        git_head=head,git_status_at_snapshot=status,model=model,architecture='gru',encoding='one_line',tasks=['peaks','valleys'],
        units=units,unit_count=len(units),checkpoint_count=sum(len(u['models']) for u in units),checks=count,
        required_missing=missing,failures=failures,files_hashed=files,unique_files_hashed=len(files),
        bytes_hashed=sum(r['size_bytes'] for r in files.values()),
        model_definition=dict(path=str(ROOT/'experiments/permworld_peaks_probe_20261109/vendor/archdev_common.py'),sha256=MODEL_SHA,class_name='CausalGRU',
            vocabulary_path=str(ROOT/'experiments/permworld_peaks_probe_20261109/vendor/passage.py'),vocabulary_sha256=VOCAB_SHA),
        optional_legacy_F17_catalog_paths_not_present_in_current_checkout=optional_unavailable,
        limitations=['Hashes cover named paths at endpoints, not every filesystem object or continuous monitoring.',
            'Unused external token/activation payloads are not sourced or rehashed; prior reports and their provenance records are protected separately.',
            'Absent historical F17 paper paths from an old relative catalog are listed explicitly; this audit does not locate or change a relocated F17 project.',
            'The earlier development-only eight-anchor validation preflight preceded this snapshot; no formal test outputs were inspected here.'],
        no_neural_module_construction=True,no_neural_forward=True,no_probe_fitting=True,no_base_training=True,
        no_formal_test_output_access=True,no_old_file_writes=True,elapsed_seconds=time.monotonic()-began)
    write(HERE/'source_audit.json',audit)
    protection = dict(created_utc=now,git_head=head,git_status=status,
        stage='After read-only historical validation engine checks, before this task formal feature/readout fitting.',
        files={path:r['sha256'] for path,r in files.items()},
        scope='Explicit named current scientific documents/results/source files, actual used checkpoint/data files and current tracked document/paper paths. Endpoint comparison only; no global or continuous unchanged claim.',
        excluded='Unused external token-state/activation payloads; absent legacy-relative F17 catalog paths.',
        source_audit_sha256=sha(HERE/'source_audit.json'))
    if final_status == 'passed': write(HERE/'protection_start.json',protection)
    print(json.dumps({k:audit[k] for k in ['status','unit_count','checkpoint_count','checks','unique_files_hashed','bytes_hashed','elapsed_seconds']}),flush=True)
    if final_status != 'passed': raise SystemExit('Source audit failed; inspect gaps without training or repairing old sources.')


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
