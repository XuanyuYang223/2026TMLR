"""Bind existing stage-one timing inputs; no fitting or neural execution.

All writes go to this new diagnostic directory. Previously saved logits are
opaque hashed inputs: only their NPY headers are read, never their values.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.permworld_local_causal_20261110.shared import catalog,transformed

DIAG=Path(__file__).resolve().parent
OLD=ROOT/'diagnostics/permworld_local_causal_20261110'
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')
TASKS=('peaks','valleys')
LENGTHS=tuple(range(10,31))
SEEDS=tuple(range(1026,1034))
OLD_CONDITIONS=('unedited','local_projection','same_label_projection',
    *[f'random_direction_{s}' for s in SEEDS],'full_state_swap','actual_counterfactual')
FIXED_CELLS={(10,2),(20,6),(30,28)}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def array_sha(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


class Checks:
    def __init__(self):self.total=0;self.failures=[]
    def ok(self,value,label):
        self.total+=1
        if not bool(value):self.failures.append(label)


def npz_header(archive,key):
    with archive.open(key+'.npy') as stream:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
        else:shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
    return dict(shape=list(shape),dtype=str(dtype),fortran_order=bool(order))


def run():
    started=time.perf_counter();began=datetime.now(timezone.utc).isoformat()
    DIAG.mkdir(parents=True,exist_ok=True)
    checks=Checks();source=read(OLD/'source_audit.json');delivery=read(OLD/'delivery_manifest.json')
    old_protocol=read(OLD/'protocol.json');old_freeze=read(OLD/'freeze.json');old_protection=read(OLD/'protection_end.json')
    checks.ok(source['status']=='passed' and not source['failures'],'source audit passed')
    checks.ok(old_protection['status']=='passed' and not old_protection['failed_files'],'old protection endpoint passed')
    checks.ok(sha(OLD/'protocol.json')==old_freeze['protocol_sha256']==delivery['protocol_sha256'],'old frozen protocol binding')
    checks.ok(old_protocol['selected_layer']==0 and old_protocol['point']==1 and old_protocol['dose']==1,'old layer1 AFTER dose1')
    expected={};roles={};symlinks={}
    def add(path,digest,role):
        path=Path(path);absolute=str(path.absolute())
        if absolute in expected:checks.ok(expected[absolute]==digest,'Expected digest consistent: '+absolute)
        expected[absolute]=digest;roles.setdefault(absolute,[]).append(role)
        if path.is_symlink():symlinks[absolute]=dict(target=str(path.readlink()),physical_path=str(path.resolve()))
    for item in old_protection['files']:add(item['path'],item['actual_sha256'],'prior_named_protection_endpoint')
    for item in delivery['files'].values():
        add(item['path'],item['sha256'],'old_local_causal_delivery')
        checks.ok(str(Path(item['path']).resolve())==item['physical_path'],'Delivery physical path '+item['path'])
        checks.ok(Path(item['path']).stat().st_size==item['size_bytes'],'Delivery file size '+item['path'])
    for path in [OLD/'delivery_manifest.json',OLD/'completion.json',OLD/'protection_start.json',OLD/'protection_end.json']:
        add(path,sha(path),'prior_manifest_or_completion_endpoint')
    for rel,digest in old_freeze['sources'].items():add(ROOT/rel,digest,'old_frozen_source')
    for unit in source['units']:
        add(unit['data_path'],unit['data_sha256'],'used_original_dataset')
        for task,model in unit['models'].items():add(model['checkpoint_path'],model['checkpoint_sha256'],'used_fixed_checkpoint')
    state_references=[]
    for item in delivery['recurrent_states'].values():
        path=Path(item['path']);value=np.load(path,mmap_mode='r',allow_pickle=False)
        receipt=read(item['receipt'])
        used='/test/' in str(path)
        checks.ok(value.dtype==np.float32 and value.shape==(receipt['node_count'],2,4,256),'State header '+str(path))
        checks.ok(path.stat().st_size==item['size_bytes']==value.offset+value.nbytes,'State length '+str(path))
        checks.ok(receipt['files'][str(path)]==item['sha256'],'State creation digest '+str(path))
        reference=dict(item,used_by_timing=used,shape=list(value.shape),dtype=str(value.dtype),
                       hash_scope='Fresh whole-payload SHA at timing input audit' if used else 'Creation-time SHA plus current header/size only')
        state_references.append(reference)
        if used:add(path,item['sha256'],'used_test_state_fresh_initial_hash')
        del value
    # Hash each physical file once, including all3570 old cell NPZs and used
    # test-state caches. Unused40GB token caches are deliberately outside scope.
    physical={str(Path(path).resolve()) for path in expected}
    print(f'TIMING INPUT HASH {len(physical)} physical files',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fresh=dict(zip(sorted(physical),pool.map(sha,sorted(physical))))
    for path,digest in expected.items():checks.ok(fresh[str(Path(path).resolve())]==digest,'Fresh input SHA '+path)
    print('TIMING INPUT HASH COMPLETE',flush=True)
    units=[];cell_count=0;subset_count=0
    for old_unit in source['units']:
        unit=json.loads(json.dumps(old_unit));name=unit['unit']
        family_path=OUT/name/'selected_families.json.gz';family_receipt=OLD/'data'/name/'completion.json'
        receipt=read(family_receipt)
        checks.ok(fresh[str(family_path.resolve())]==receipt['output_sha256'][str(family_path)],name+' family binding')
        with gzip.open(family_path,'rt') as stream:selected=json.load(stream)
        family_by_pair={r['pair_id']:(f,r) for f in selected if f['split']=='test' for r in f['pairs'] if r['kind'] in ('simple','count_matched')}
        controls={r['control_for_pair_id']:r for f in selected if f['split']=='test' for r in f['pairs'] if r['kind']=='same_label'}
        unit.update(families=dict(path=str(family_path),sha256=expected[str(family_path)],receipt_path=str(family_receipt)),
                    probes={},test_states={},cells={})
        for task in TASKS:
            model=unit['models'][task]
            probe_path=OUT/name/'probes'/task/'layer1_point1.npz'
            with np.load(probe_path,allow_pickle=False) as z:probe={k:z[k].copy() for k in z.files}
            direction=probe['direction']
            checks.ok(direction.shape==(256,) and direction.dtype==np.float64,name+'/'+task+' probe direction dtype')
            checks.ok(abs(float(direction@direction)-1)<=2e-12,name+'/'+task+' unit direction')
            checks.ok(not set(probe['fit_a_node_ids'])&set(probe['fit_b_node_ids']),name+'/'+task+' original fit halves disjoint')
            unit['probes'][task]=dict(path=str(probe_path),sha256=expected[str(probe_path)],
                direction_sha256=array_sha(direction),editing_parameter_fields=[k for k in probe if k.startswith('edit_')],
                checker_parameter_fields=[k for k in probe if k.startswith('checker_')],
                count_parameter_fields=[k for k in probe if k.startswith('count_')],
                refit=False,layer=1,point=1)
            state_path=OUT/name/'states'/task/'test/states.npy'
            state_receipt_path=state_path.parent/'completion.json';sr=read(state_receipt_path)
            nodes,lookup=catalog(selected,task,{'test'})
            ids=[r['node_id'] for r in nodes];id_to_row={node_id:i for i,node_id in enumerate(ids)}
            checks.ok(ids==sr['node_ids'],name+'/'+task+' deterministic catalog order')
            checks.ok(sr['model_parameter_sha256']==model['parameter_sha256'],name+'/'+task+' checkpoint state identity')
            checks.ok(sr['actual_state']=='GRU h_n per layer, before/after right-value update',name+'/'+task+' actual recurrence cache')
            states=np.load(state_path,mmap_mode='r',allow_pickle=False)
            unit['test_states'][task]=dict(path=str(state_path),sha256=expected[str(state_path)],
                receipt_path=str(state_receipt_path),receipt_sha256=expected[str(state_receipt_path)],
                node_ids_sha256=array_sha(np.asarray(ids)),node_count=len(ids),shape=list(states.shape),dtype=str(states.dtype),
                state_axis=['node','point_before_after','layer','coordinate'],before_point=0,after_point=1,
                checkpoint_sha256=model['checkpoint_sha256'],parameter_sha256=model['parameter_sha256'],
                data_sha256=unit['data_sha256'],family_sha256=expected[str(family_path)],
                data_binding_evidence='Source audited data + selected-family SHA + deterministic catalog ID equality + all cell data/checkpoint provenance fields',
                hash_scope='Full payload freshly rehashed once at timing input audit')
            cells=[];reference_q=None
            for n in LENGTHS:
                for i in range(2,n-1):
                    path=OUT/name/'interventions'/task/f'n{n}_i{i}.npz'
                    with ZipFile(path) as archive:
                        headers={k:npz_header(archive,k) for k in ['logits','recipient_states','donor_states','same_label_states','random_directions','direction','coordinate']}
                    with np.load(path,allow_pickle=False) as z:
                        pair_ids=z['pair_ids'].tolist();recipient_ids=z['recipient_node_ids'].tolist();donor_ids=z['donor_node_ids'].tolist();cf_ids=z['actual_cf_node_ids'].tolist()
                        conditions=z['condition_names'].tolist();q=z['random_directions'];v=z['direction'];coordinate=z['coordinate']
                        checkpoint=str(z['source_checkpoint_sha256'].item());data_hash=str(z['data_sha256'].item());protocol_hash=str(z['protocol_sha256'].item())
                        checks.ok(conditions==list(OLD_CONDITIONS),str(path)+' old13 condition order')
                        checks.ok(checkpoint==model['checkpoint_sha256'] and data_hash==unit['data_sha256'],str(path)+' checkpoint/data')
                        checks.ok(protocol_hash==old_freeze['protocol_sha256'],str(path)+' old protocol')
                        checks.ok(np.array_equal(v,direction),str(path)+' exact saved direction')
                        checks.ok(q.shape==(8,256) and q.dtype==np.float64,str(path)+' actual random direction shape')
                        if reference_q is None:reference_q=q.copy()
                        checks.ok(np.array_equal(q,reference_q),str(path)+' exact common random directions')
                        checks.ok(headers['logits']['shape']==[13,len(pair_ids),188] and headers['logits']['dtype']=='float32',str(path)+' opaque logits header')
                        for member in ('recipient_states','donor_states','same_label_states'):
                            checks.ok(headers[member]['shape']==[len(pair_ids),4,256] and headers[member]['dtype']=='float32',str(path)+' state header '+member)
                        ri=[];di=[];ci=[];si=[];groups=[];signs=[];truth=[];goals=[];donor_counts=[]
                        for j,pair_id in enumerate(pair_ids):
                            f,r=family_by_pair[pair_id];same=controls[pair_id]
                            checks.ok(f['n']==n and f['center']==i,str(path)+' fixed pair cell')
                            def index(side):return lookup[n,i,transformed(r[side],task)]
                            rid,did,cid=index('recipient'),index('donor'),index('actual_local_counterfactual')
                            sid=lookup[n,i,transformed(same['donor'],task)]
                            checks.ok(recipient_ids[j]==nodes[rid]['node_id'] and donor_ids[j]==nodes[did]['node_id'] and cf_ids[j]==nodes[cid]['node_id'],str(path)+' pair/node ID binding')
                            ri.append(rid);di.append(did);ci.append(cid);si.append(sid)
                            groups.append('main' if r['kind']=='simple' else 'matched');signs.append(r['direction'])
                            truth.append(r['recipient_total_count']);goals.append(r['actual_local_counterfactual_total_count']);donor_counts.append(r['donor_total_count'])
                        record=dict(path=str(path),sha256=expected[str(path)],n=n,i=i,case_count=len(pair_ids),pair_ids=pair_ids,
                            recipient_state_rows=ri,donor_state_rows=di,same_label_state_rows=si,actual_cf_state_rows=ci,
                            groups=groups,signs=signs,recipient_true_counts=truth,actual_cf_true_counts=goals,donor_true_counts=donor_counts,
                            source_checkpoint_sha256=checkpoint,data_sha256=data_hash,protocol_sha256=protocol_hash,
                            direction_sha256=array_sha(v),random_directions_sha256=array_sha(q),coordinate_sha256=array_sha(coordinate),
                            condition_names=conditions,member_headers=headers,cut_token_index=4+2*i)
                        if (n,i) in FIXED_CELLS:
                            hr=z['recipient_states'];hd=z['donor_states'];hs=z['same_label_states']
                            checks.ok(np.array_equal(hr,states[ri,1]),str(path)+' fixed all recipient AFTER bytes')
                            checks.ok(np.array_equal(hd,states[di,1]),str(path)+' fixed all donor AFTER bytes')
                            checks.ok(np.array_equal(hs,states[si,1]),str(path)+' fixed all same-label AFTER bytes')
                            h1=(hr[:,0].astype(np.float64)+coordinate[:,None]*v).astype(np.float32)
                            record['fixed_subset_edited_h1_sha256']=array_sha(h1)
                            record['fixed_subset_before_state_sha256']=array_sha(states[ri,0])
                            record['fixed_subset_after_bytes_checked']=True
                            subset_count+=1
                    cells.append(record);cell_count+=1
            unit['cells'][task]=cells
            del states
        units.append(unit)
        print('TIMING INPUT UNIT',name,'passed' if not checks.failures else 'FAILURES',flush=True)
    status='passed' if not checks.failures else 'failed'
    input_audit=dict(status=status,captured_utc=began,completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Existing input/parameter provenance and metadata only. No recurrent model construction/forward, refit, data generation or new timing results. Old logits values are not read.',
        source_audit_path=str(OLD/'source_audit.json'),source_audit_sha256=sha(OLD/'source_audit.json'),
        old_protocol_path=str(OLD/'protocol.json'),old_protocol_sha256=old_freeze['protocol_sha256'],
        old_delivery_manifest_path=str(OLD/'delivery_manifest.json'),old_delivery_manifest_sha256=sha(OLD/'delivery_manifest.json'),
        model=source['model'],tasks=list(TASKS),units=units,unit_count=len(units),cell_count=cell_count,
        fixed_subset_cells=sorted([list(v) for v in FIXED_CELLS]),fixed_subset_cell_files=subset_count,
        state_references=state_references,used_test_state_payloads_freshly_hashed=10,
        unused_fit_dev_state_payloads_rehashed=0,unused_40GB_token_payloads_read=0,
        checks=checks.total,failures=checks.failures,
        immutable_reuse_contract=dict(layer=1,point=1,dose=1,probe_refit=False,data_regenerated=False,
            early_late_target='Build the FP32 edited current h1 ONCE from old hR/hD/hS, exact saved coordinate/direction/q; pass the identical bytes to early and late branches. Do not project the previous h1 separately.',
            previous_state='Old states.npy point0, keyed by audited recipient row; same checkpoint/data/catalog as point1.',
            late_replay='Rerun old13 late conditions without new deltas and compare logits at old absolute1e-4 tolerance.',
            early_semantics='Install identical edited current h1 while recomputing upper layers for the current value from their old BEFORE states; then continue common future suffix.',
            numeric_control='Original saved FP64 direction, q and coordinate bytes; original probe file; FP32 recurrent edited state; no refit or regenerated controls.'),
        interpretation=dict(statistical_units='Four new units, historical reference separately; examples/signs/tasks/positions not independent repeats.',
            local_goal='Recipient true full count plus intended sign, including matched pairs.',
            full_swap_goal='Actual donor prediction; matched donor count stays recipient count and differs from actual local-only counterfactual.',
            negative_stage1='The old AFTER null result does not establish earlier non-use because upper layers already held their current recipient updates.',
            timing_scope='Early versus late tests intervention timing with fixed layer1 target bytes. It does not automatically identify a unique abstract lemma, remove count/value proxies or authorize cross-model work.'),
        fresh_hash_physical_files=len(fresh),fresh_hash_bytes=sum(Path(p).stat().st_size for p in fresh),
        audit_source_sha256=sha(__file__),elapsed_seconds=time.perf_counter()-started,
        no_forward=True,no_fit=True,no_data_generation=True,no_new_timing_outcome_access=True,no_old_scientific_file_writes=True)
    write(DIAG/'input_audit.json',input_audit)
    # A fresh endpoint snapshot of the explicitly enumerated protection set.
    # Unused fit/dev payloads below have creation-time references, not new SHAs.
    protection=dict(created_utc=began,git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        files={path:fresh[str(Path(path).resolve())] for path in expected},
        file_records={path:dict(physical_path=str(Path(path).resolve()),size_bytes=Path(path).stat().st_size,roles=roles[path],hash_scope='Fresh current endpoint SHA') for path in expected},
        symlinks=symlinks,state_creation_references=state_references,
        scope='Explicit prior named protection endpoint + all prior local-causal delivery files + manifest/completion endpoints + all used checkpoint/data files + ten used test-state caches. Hashes are endpoint evidence, not continuous/global mutation monitoring.',
        excluded=['Unused fit/dev recurrent payloads retain creation-time hashes and current header/size checks only','Unused40GB token caches','Unenumerated files outside named scope'],
        input_audit_sha256=sha(DIAG/'input_audit.json'),status=status,failures=checks.failures,
        unique_physical_files_hashed=len(fresh),bytes_hashed=input_audit['fresh_hash_bytes'])
    write(DIAG/'protection_start.json',protection)
    print(dict(status=status,checks=checks.total,failures=checks.failures[:10],cells=cell_count,seconds=time.perf_counter()-started),flush=True)
    if checks.failures:raise SystemExit(1)


if __name__=='__main__':run()
