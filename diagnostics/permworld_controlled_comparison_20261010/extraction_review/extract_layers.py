"""Read-only extraction for the frozen, first 16-model, 20k PermWorld cohort.

No model fitting, optimizer, backwards, or initial-weight creation occurs here.
Final-norm arrays are audit outputs: the scientific primary endpoint remains the
unchanged, per-logical-job prefix_final arrays in the original cache.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
from experiments.permworld_encoding.core import TASKS, TOKEN_TO_ID, batch, parameter_hash
from experiments.permworld_encoding_archdev_20261009.common import model, device_setup

DIAG = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = DIAG.parent / 'protocol.json'
LAYERS = {
    'transformer': ['block_01', 'block_02', 'block_03', 'block_04', 'final_norm'],
    'gru': ['layer_01', 'layer_02', 'layer_03', 'layer_04', 'final_norm'],
}
TOLERANCE = 2e-5


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def verify(protocol_path):
    """Fail before CUDA/model construction unless a complete freeze exists."""
    protocol_path = Path(protocol_path).resolve()
    assert protocol_path.is_file() and protocol_path.with_name('freeze.json').is_file()
    protocol = read_json(protocol_path)
    frozen = read_json(protocol_path.with_name('freeze.json'))
    assert sha(protocol_path) == frozen['protocol_sha256']
    for relative, expected in frozen['source_sha256'].items():
        path = Path(relative)
        path = path if path.is_absolute() else WORKSPACE / path
        assert sha(path) == expected, ('Frozen dependency differs', path)
    source_root = Path(protocol['source_root']).resolve()
    source_path = Path(protocol['source_protocol_path']).resolve()
    assert source_path == source_root / 'protocol.json'
    assert sha(source_path) == protocol['source_protocol_sha256']
    source = read_json(source_path)
    assert source['training']['updates'] == 20000
    assert source['lengths'] == list(range(10, 31)) == protocol['lengths']
    assert source['tasks'] == ['descents', 'recoils', 'peaks', 'valleys']
    assert source['model']['d_model'] == 256 and source['model']['layers'] == 4
    source_freeze = read_json(source_root / 'freeze_manifest.json')
    version = hashlib.sha256(json.dumps(source_freeze['source_sha256'],
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert version == protocol['source_version_sha256']
    for relative, expected in source_freeze['source_sha256'].items():
        path = Path(relative)
        path = path if path.is_absolute() else WORKSPACE / path
        assert sha(path) == expected
    inventory = protocol['checkpoint_inventory']
    assert len(inventory) == len({j['run_id'] for j in inventory}) == 16
    expected = {j['run_id']: j for j in source['training_manifest']}
    assert set(expected) == {j['run_id'] for j in inventory}
    for job in inventory:
        for key, value in expected[job['run_id']].items():
            if key in job:
                assert job[key] == value, (job['run_id'], key)
        for key in ('architecture', 'encoding', 'task', 'model_seed'):
            assert job[key] == expected[job['run_id']][key]
        directory = source_root / 'models' / job['run_id']
        assert Path(job['checkpoint_path']).resolve() == directory / 'checkpoint.pt'
        assert sha(job['checkpoint_path']) == job['checkpoint_sha256']
        completed = read_json(directory / 'completed.json')
        assert completed['step'] == completed['finite_gradient_updates'] == 20000
        assert completed['protocol_sha256'] == protocol['source_protocol_sha256']
        assert completed['execution_source_version_sha256'] == version
        assert completed['checkpoint_sha256'] == job['checkpoint_sha256']
        initial = source_root / 'initializations' / job['architecture'] / 'step0.pt'
        assert initial.is_file() and initial.with_suffix('.json').is_file()
        assert Path(job['initial_checkpoint_path']).resolve() == initial
        assert sha(initial) == job['initial_checkpoint_sha256']
        meta = read_json(initial.with_suffix('.json'))
        assert meta['checkpoint_sha256'] == job['initial_checkpoint_sha256']
        assert meta['actual_initial_identity_verified']
        assert meta['protocol_sha256'] == protocol['source_protocol_sha256']
        assert meta['parameter_sha256'] == completed['initial_parameter_sha256']
    binding = read_json(source_root / 'initializations/execution_binding_v4.json')
    assert binding['protocol_sha256'] == protocol['source_protocol_sha256']
    assert binding['source_version_sha256'] == version
    cache_root = Path(protocol['cache_root']).resolve()
    assert cache_root.is_relative_to(Path('/mnt/yangx-projects-f/20261010'))
    assert not cache_root.is_relative_to(source_root)
    return protocol, source, source_root, cache_root, sha(protocol_path)


def load_data(source_root, source, protocol):
    world = source_root / 'worlds' / source['units'][0]['unit']
    audit = read_json(world / 'data_audit.json')
    assert sha(world / 'data.npz') == audit['data_sha256']
    if 'source_data_sha256' in protocol:
        assert audit['data_sha256'] == protocol['source_data_sha256']
    # Decode only the previously frozen fit/evaluation inputs and gold labels.
    with np.load(world / 'data.npz', allow_pickle=False) as archive:
        data = {f'{split}_{key}': archive[f'{split}_{key}'].copy()
            for split in ('validation', 'test')
            for key in ('lengths', 'labels', 'one_line_input', 'canonical_cycle_input')}
    for split, per_length in [('validation', 256), ('test', 512)]:
        ns = data[split + '_lengths']
        assert sorted(map(int, np.unique(ns))) == source['lengths']
        assert all(int((ns == n).sum()) == per_length for n in source['lengths'])
        assert data[split + '_labels'].shape[1] == 3
    return data, audit


def logical_key(job, condition, split):
    return '|'.join([job['run_id'], condition, split])


def old_cache(job, condition, split, source_root, protocol, data_sha):
    path = source_root / 'evaluation_cache' / (
        f'{job["architecture"]}_{job["encoding"]}_{job["task"]}_{condition}_{split}.npz')
    metadata = read_json(path.with_suffix('.json'))
    sig = metadata['signature']
    assert sig['protocol_sha256'] == protocol['source_protocol_sha256']
    assert sig['data_sha256'] == data_sha and sig['split'] == split
    expected = job['initial_checkpoint_sha256'] if condition == 'initial' else job['checkpoint_sha256']
    assert sig['checkpoint_sha256'] == expected
    assert sha(path) == metadata['sha256']
    return path, metadata['sha256']


def plan_sources(protocol, source_root, data_sha):
    """Share only hidden initial states proven exactly equal for all tasks."""
    old_map, equality = {}, []
    jobs = protocol['checkpoint_inventory']
    for job in jobs:
        for condition in ('initial', 'trained'):
            for split in ('validation', 'test'):
                path, digest = old_cache(job, condition, split, source_root, protocol, data_sha)
                old_map[logical_key(job, condition, split)] = dict(path=str(path), sha256=digest)
    for architecture in ('transformer', 'gru'):
        for encoding in ('one_line', 'canonical_cycle'):
            group = [j for j in jobs if j['architecture'] == architecture and j['encoding'] == encoding]
            representative = next(j for j in group if j['task'] == 'descents')
            for split in ('validation', 'test'):
                ref = old_map[logical_key(representative, 'initial', split)]
                with np.load(ref['path'], allow_pickle=False) as ar:
                    base = ar['prefix_final']
                for job in group:
                    rec = old_map[logical_key(job, 'initial', split)]
                    with np.load(rec['path'], allow_pickle=False) as ar:
                        other = ar['prefix_final']
                    equality.append(dict(run_id=job['run_id'], split=split,
                        representative_run_id=representative['run_id'],
                        bitwise_equal=bool(np.array_equal(base, other)),
                        max_abs=float(np.max(np.abs(base - other)))))
    share = all(row['bitwise_equal'] for row in equality)
    physical, mapping = {}, {}
    for job in jobs:
        for condition in ('initial', 'trained'):
            representative = job
            if condition == 'initial' and share:
                representative = next(j for j in jobs if j['architecture'] == job['architecture']
                    and j['encoding'] == job['encoding'] and j['task'] == 'descents')
            for split in ('validation', 'test'):
                key = logical_key(representative, condition, split)
                physical[key] = dict(job=representative, condition=condition, split=split)
                mapping[logical_key(job, condition, split)] = key
    assert len(mapping) == 64 and len(physical) == (40 if share else 64)
    return physical, mapping, old_map, equality, share


def load_model(job, condition, source, source_root, protocol, data_sha, device):
    path = Path(job['initial_checkpoint_path'] if condition == 'initial' else job['checkpoint_path'])
    expected = job['initial_checkpoint_sha256'] if condition == 'initial' else job['checkpoint_sha256']
    assert sha(path) == expected
    cp = torch.load(path, map_location='cpu', weights_only=False)
    if condition == 'trained':
        assert cp['run_id'] == job['run_id'] and cp['architecture'] == job['architecture']
        assert cp['step'] == cp['finite_gradient_updates'] == 20000
        assert cp['protocol_sha256'] == protocol['source_protocol_sha256']
        assert cp['execution_source_version_sha256'] == protocol['source_version_sha256']
        assert cp['data_sha256'] == data_sha
        assert cp['initial_checkpoint_sha256'] == job['initial_checkpoint_sha256']
    # No initial() helper: saved actual initializations are loaded directly.
    net = model(source, job['architecture'], job['model_seed'], device)
    net.load_state_dict(cp['model'], strict=True)
    net.eval()
    assert all(p.dtype == torch.float32 and p.grad is None for p in net.parameters())
    if condition == 'initial':
        meta = read_json(path.with_suffix('.json'))
        assert parameter_hash(net) == meta['parameter_sha256']
    del cp
    return net


class Capture:
    def __init__(self, net, architecture):
        self.rows = self.ends = None
        self.states = {}
        self.handles = []
        modules = net.blocks if architecture == 'transformer' else net.recurrent_layers
        for name, module in zip(LAYERS[architecture][:4], modules):
            self.handles.append(module.register_forward_hook(self.hook(name)))
        self.handles.append(net.final_norm.register_forward_hook(self.hook('final_norm')))

    def hook(self, name):
        def capture(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.states[name] = hidden[self.rows, self.ends].detach().cpu().numpy().copy()
        return capture

    def reset(self, rows, ends):
        self.rows, self.ends = rows, ends
        self.states.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def forward(net, capture, prefixes, ns, task, device, architecture):
    ids, mask, _, ends, equals = batch(prefixes, ns, task, device=device)
    rows = torch.arange(len(ns), device=device)
    assert torch.equal(ends, torch.as_tensor(2*np.asarray(ns)+3, device=device))
    assert torch.equal(equals, ends+2)
    capture.reset(rows, ends)
    assert not net.training and not torch.is_grad_enabled()
    with torch.autocast(device.type, enabled=False):
        logits = net(ids, mask)
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all()
    hidden = np.stack([capture.states[name] for name in LAYERS[architecture]], axis=1)
    assert hidden.dtype == np.float32 and np.isfinite(hidden).all()
    answer_logits = logits[rows, equals]
    return hidden, answer_logits, ids, mask, ends


@torch.inference_mode()
def manual(net, ids, mask, ends, architecture):
    positions = torch.arange(ids.shape[1], device=ids.device)
    rows = torch.arange(len(ids), device=ids.device)
    h = net.token_embedding(ids) + net.position_embedding(positions)[None]
    h = net.embedding_dropout(h) * mask[..., None]
    values = []
    if architecture == 'transformer':
        for block in net.blocks:
            h = block(h, mask)
            values.append(h[rows, ends].cpu().numpy())
    else:
        for i, recurrent in enumerate(net.recurrent_layers):
            h, _ = recurrent(h)
            values.append(h[rows, ends].cpu().numpy())
            if i < len(net.interlayer_dropout):
                h = net.interlayer_dropout[i](h)
    h = net.final_norm(h) * mask[..., None]
    values.append(h[rows, ends].cpu().numpy())
    return np.stack(values, axis=1)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark(protocol_path):
    protocol, source, source_root, cache_root, digest = verify(protocol_path)
    assert not (DIAG/'benchmark.json').exists(), 'Inspect existing benchmark instead of silently repeating it'
    data, audit = load_data(source_root, source, protocol)
    physical, mapping, old_map, equality, share = plan_sources(protocol, source_root, audit['data_sha256'])
    device = device_setup()
    # Test inputs and checks are precisely the existing frozen anchors, not new draws.
    nets = {key: value for key, value in physical.items() if value['split'] == 'test'}
    records = []
    total_started = time.monotonic()
    for key, spec in nets.items():
        job, condition = spec['job'], spec['condition']
        net = load_model(job, condition, source, source_root, protocol, audit['data_sha256'], device)
        before = parameter_hash(net)
        capture = Capture(net, job['architecture'])
        reference = old_map[key]
        with np.load(reference['path'], allow_pickle=False) as ar:
            old_final = ar['prefix_final']
        try:
            for n in (10, 20, 30):
                chosen = np.flatnonzero(data['test_lengths'] == n)[:128]
                assert len(chosen) == 128
                for action in range(3):
                    prefix = data[f'test_{job["encoding"]}_input'][chosen, action]
                    ns = data['test_lengths'][chosen]
                    sync(device); started = time.monotonic()
                    hidden, _, ids, mask, ends = forward(net, capture, prefix, ns,
                        job['task'], device, job['architecture'])
                    sync(device); seconds = time.monotonic()-started
                    # Compare extraction paths at the same batch/sequence shape.
                    # Prefix-only, smaller-batch raw residuals can have different
                    # floating-point kernels; retain them as a separate diagnostic.
                    count = 8
                    prefix_ids = ids[:count, :2*n+4]
                    prefix_mask = mask[:count, :2*n+4]
                    # Hooks are disabled while independently computing the manual states.
                    capture.close()
                    same_shape = manual(net, ids, mask, ends, job['architecture'])
                    independent = manual(net, prefix_ids, prefix_mask, ends[:count], job['architecture'])
                    capture = Capture(net, job['architecture'])
                    layer_error = float(np.max(np.abs(same_shape[:count]-hidden[:count])))
                    prefix_layer_error = float(np.max(np.abs(independent-hidden[:count])))
                    prefix_norm_error = float(np.max(np.abs(independent[:,-1]-hidden[:count,-1])))
                    legacy_error = float(np.max(np.abs(hidden[:count, -1]-old_final[chosen[:count], action])))
                    assert layer_error <= TOLERANCE and legacy_error <= TOLERANCE and prefix_norm_error <= TOLERANCE, (
                        key,n,action,layer_error,legacy_error,prefix_norm_error)
                    records.append(dict(source_key=key,architecture=job['architecture'],encoding=job['encoding'],
                        condition=condition,length=n,action=action,batch_size=128,forward_seconds=seconds,
                        independent_vectors=8,manual_same_shape_layers_max_abs=layer_error,
                        manual_same_shape_per_layer_max_abs=np.max(np.abs(same_shape[:count]-hidden[:count]),axis=(0,2)).tolist(),
                        manual_prefix_layers_max_abs=prefix_layer_error,
                        manual_prefix_per_layer_max_abs=np.max(np.abs(independent-hidden[:count]),axis=(0,2)).tolist(),
                        manual_prefix_final_norm_max_abs=prefix_norm_error,
                        prefix_only_raw_layers_are_shape_diagnostic_not_gate=True,
                        old_final_norm_max_abs=legacy_error))
            assert parameter_hash(net) == before and all(p.grad is None for p in net.parameters())
        finally:
            capture.close()
        del net, old_final
    # Per-config n10/20/30 medians estimate each frozen length by interpolation.
    estimate = 0.
    for key, spec in physical.items():
        template = key.rsplit('|',1)[0]+'|test'
        by_n = {n:float(np.median([r['forward_seconds'] for r in records
            if r['source_key']==template and r['length']==n])) for n in (10,20,30)}
        per_length = 256 if spec['split']=='validation' else 512
        estimate += 3 * (per_length//128) * float(np.interp(source['lengths'],[10,20,30],
            [by_n[n] for n in (10,20,30)]).sum())
    result = dict(status='passed',protocol_sha256=digest,device=str(device),records=records,
        tolerance=TOLERANCE,initial_shared_exact=share,initial_equality=equality,
        physical_split_count=len(physical),logical_split_count=len(mapping),
        estimated_full_extraction_forward_seconds=estimate,
        estimated_full_extraction_forward_gpu_hours=estimate/3600 if device.type=='cuda' else None,
        cost_scope='128-example forward plus5 endpoint transfers; excludes file hashing, CP loading and memmap writes; n11..29 interpolated, not an actual total.',
        activation_gate='Same128-example full-prompt explicit manual layers versus hook first8, prefix-only8 final_norm, and original cache final_norm; each tolerance2e-5. Smaller-batch prefix raw-layer discrepancies reported separately.',
        benchmark_wall_seconds=time.monotonic()-total_started,source_updates=0,
        old_cache_final_norm_retained_as_primary=True)
    write_json(DIAG/'benchmark.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','initial_equality')}),flush=True)


def extract(protocol_path):
    protocol, source, source_root, cache_root, digest = verify(protocol_path)
    gate = read_json(DIAG/'benchmark.json')
    assert gate['status']=='passed' and gate['protocol_sha256']==digest
    data, audit = load_data(source_root, source, protocol)
    physical, mapping, old_map, equality, share = plan_sources(protocol, source_root, audit['data_sha256'])
    assert share == gate['initial_shared_exact']
    root = cache_root/'layers'; root.mkdir(parents=True,exist_ok=True)
    device = device_setup()
    index_path = root/'index.json'
    index = dict(schema_version='controlled-layer-cache/v1',status='running',protocol_sha256=digest,
        layers=LAYERS,records=[],job_condition_split_map=mapping,logical_old_cache_map=old_map,
        initial_shared_exact=share,initial_equality=equality,source_data_sha256=audit['data_sha256'],
        source_updates=0,embedding_end='Structural constant within length; no forward storage; undefined metrics.',
        primary_final_norm_source='Original per-logical-job prefix_final cache unchanged',
        EOS='Not evaluated by this prompt-only extraction; no true answer appended.',
        initial_CE='Not newly evaluated; task-specific outputs must never be inherited from shared initial hidden.',
        new_CKA=False)
    started_all=time.monotonic()
    for key,spec in physical.items():
        job,condition,split=spec['job'],spec['condition'],spec['split']
        directory=root/key.replace('|','__');directory.mkdir(exist_ok=True)
        final_record=directory/'record.json'
        if final_record.exists():
            existing=read_json(final_record)
            assert existing['protocol_sha256']==digest and existing['source_key']==key
            for field in ('features','lengths','answer_ce','answer_argmax'):
                if field+'_path' in existing:
                    assert sha(existing[field+'_path'])==existing[field+'_sha256']
            index['records'].append(existing);write_json(index_path,index);continue
        net=load_model(job,condition,source,source_root,protocol,audit['data_sha256'],device)
        before=parameter_hash(net);capture=Capture(net,job['architecture'])
        ns=data[split+'_lengths'];N=len(ns)
        feature_tmp=directory/'features.partial.npy'
        features=np.lib.format.open_memmap(feature_tmp,mode='w+',dtype=np.float32,shape=(N,3,5,256))
        ce=pred=None
        if condition=='trained':
            ce=np.lib.format.open_memmap(directory/'answer_ce.partial.npy',mode='w+',dtype=np.float64,shape=(N,3))
            pred=np.lib.format.open_memmap(directory/'answer_argmax.partial.npy',mode='w+',dtype=np.int16,shape=(N,3))
        ref=old_map[key]
        with np.load(ref['path'],allow_pickle=False) as ar:
            old_final,old_prediction=ar['prefix_final'],ar['prediction']
        max_error=0.;prediction_disagreements=0;seen=np.zeros((N,3),np.bool_)
        started=time.monotonic()
        try:
            for start in range(0,N,128):
                use=np.arange(start,min(start+128,N));batch_ns=ns[use]
                for action in range(3):
                    prefix=data[f'{split}_{job["encoding"]}_input'][use,action]
                    hidden,logits,_,_,_=forward(net,capture,prefix,batch_ns,job['task'],device,job['architecture'])
                    features[use,action]=hidden
                    error=float(np.max(np.abs(hidden[:,-1]-old_final[use,action])))
                    max_error=max(max_error,error)
                    assert error<=TOLERANCE,(key,start,action,error)
                    predicted=logits.argmax(-1).cpu().numpy()
                    prediction_disagreements+=int((predicted!=old_prediction[use,action]).sum())
                    if condition=='trained':
                        labels=data[split+'_labels'][use,action,TASKS.index(job['task'])]
                        truth=torch.tensor([TOKEN_TO_ID[f'{int(v):02d}'] for v in labels],device=device)
                        ce[use,action]=F.cross_entropy(logits,truth,reduction='none').cpu().double().numpy()
                        pred[use,action]=predicted
                    assert not seen[use,action].any();seen[use,action]=True
            sync(device)
            assert seen.all() and parameter_hash(net)==before and all(p.grad is None for p in net.parameters())
            assert prediction_disagreements==0,(key,prediction_disagreements)
        finally:
            capture.close()
        features.flush();del features
        os.replace(feature_tmp,directory/'features.npy')
        np.save(directory/'lengths.npy',ns)
        if condition=='trained':
            ce.flush();pred.flush();del ce,pred
            os.replace(directory/'answer_ce.partial.npy',directory/'answer_ce.npy')
            os.replace(directory/'answer_argmax.partial.npy',directory/'answer_argmax.npy')
        cp_path=job['initial_checkpoint_path'] if condition=='initial' else job['checkpoint_path']
        cp_sha=job['initial_checkpoint_sha256'] if condition=='initial' else job['checkpoint_sha256']
        record=dict(source_key=key,run_id=job['run_id'],architecture=job['architecture'],encoding=job['encoding'],
            task=job['task'],condition=condition,split=split,protocol_sha256=digest,
            checkpoint_path=cp_path,checkpoint_sha256=cp_sha,layer_order=LAYERS[job['architecture']],
            features_shape=[N,3,5,256],features_dtype='float32',old_cache_path=ref['path'],old_cache_sha256=ref['sha256'],
            max_final_norm_difference=max_error,old_prediction_disagreements=prediction_disagreements,
            parameter_sha256_before=before,parameter_sha256_after=before,
            forward_and_store_seconds=time.monotonic()-started,source_updates=0,
            sample_order='Original source indices',actions=['identity','complement','inverse'],
            EOS_evaluated=False,initial_CE_evaluated=False)
        for field in ('features','lengths','answer_ce','answer_argmax'):
            path=directory/(field+'.npy')
            if path.is_file():
                record[field+'_path']=str(path.resolve());record[field+'_sha256']=sha(path)
        write_json(final_record,record);index['records'].append(record);write_json(index_path,index)
        print({'source_key':key,'max_final_norm_difference':max_error,'seconds':record['forward_and_store_seconds']},flush=True)
        del net,old_final,old_prediction
    assert len(index['records'])==len(physical)
    index['status']='completed';index['actual_extraction_wall_seconds']=time.monotonic()-started_all
    index['physical_split_count']=len(physical);index['logical_split_count']=len(mapping)
    write_json(index_path,index)
    write_json(DIAG/'extraction_complete.json',dict(status='completed',protocol_sha256=digest,
        index_path=str(index_path),index_sha256=sha(index_path),physical_split_count=len(physical),logical_split_count=len(mapping),
        actual_extraction_wall_seconds=index['actual_extraction_wall_seconds'],source_updates=0,
        new_training=False,old_cache_modified=False,new_CKA=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['benchmark','extract'],required=True)
    parser.add_argument('--protocol',type=Path,default=DEFAULT_PROTOCOL)
    args=parser.parse_args()
    {'benchmark':benchmark,'extract':extract}[args.stage](args.protocol)
