"""Fresh independent data worlds with within-world orbit-disjoint splits."""
import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from .core import *


def old_archive_inputs(path):
    with np.load(path,allow_pickle=False) as a:
        if 'permutations' in a.files and 'lengths' in a.files:
            for row,n in zip(a['permutations'],a['lengths']):
                n=int(n)
                if 1<=n<=30:
                    for flat in np.asarray(row).reshape(-1,row.shape[-1]):
                        p=tuple(map(int,flat[:n]))
                        if sorted(p)==list(range(1,n+1)):yield p
        for k in a.files:
            if k=='input' or k.endswith('_input'):
                nk='lengths' if k=='input' else k[:-6]+'_lengths'
                if nk not in a.files:continue
                for row,n in zip(a[k],a[nk]):
                    n=int(n)
                    if not 1<=n<=30:continue
                    for flat in np.asarray(row).reshape(-1,row.shape[-1]):
                        try:p=decode(flat[:2*n+4],'one_line')
                        except (ValueError,TypeError,KeyError):continue
                        yield p


def contains(index,digests):
    q=np.array(digests,dtype='V16');pos=np.searchsorted(index,q)
    valid=pos<len(index);result=np.zeros(len(q),bool)
    result[valid]=index[pos[valid]]==q[valid]
    return result


def build_history(config,root):
    """One fixed historical hash index, reused by all independently sampled worlds."""
    path=root/'historical_input_hashes.npy';meta=path.with_suffix('.json')
    signature=hashlib.sha256(json.dumps(config['historical_exclusions'],sort_keys=True).encode()).hexdigest()
    if path.exists():
        m=json.loads(meta.read_text());assert m['signature']==signature and m['sha256']==sha(path)
        return path,m
    start=time.monotonic();raw=root/'historical_hashes.building';count=0
    with raw.open('wb') as out:
        hist=config['historical_exclusions'];parent=Path(hist['parent_manifest']);assert sha(parent)==hist['parent_manifest_sha256']
        manifest=json.loads(parent.read_text())
        for j,shard in enumerate(manifest['shards']):
            p=parent.parent/shard['filename'];assert sha(p)==shard['sha256']
            hashes=[]
            with gzip.open(p,'rb') as f:
                for line in f:
                    begin=line.index(b'[',line.index(b'"primary"'));end=line.index(b']',begin)+1
                    key=line[begin:end].replace(b' ',b'');hashes.append(hashlib.blake2b(key,digest_size=16).digest());count+=1
                    if len(hashes)==8192:out.write(b''.join(hashes));hashes=[]
            if hashes:out.write(b''.join(hashes))
            if j%20==0:print({'history_shards':j+1,'records':count},flush=True)
        for entry in hist['archives']:
            p=Path(entry['path']);assert sha(p)==entry['sha256']
            hashes=[]
            for permutation in old_archive_inputs(p):
                hashes.append(fingerprint(permutation));count+=1
                if len(hashes)==8192:out.write(b''.join(hashes));hashes=[]
            if hashes:out.write(b''.join(hashes))
    mm=np.memmap(raw,mode='r',dtype='V16');unique=np.unique(mm);np.save(path,unique);del mm;raw.unlink()
    m=dict(signature=signature,sha256=sha(path),records=count,unique_inputs=len(unique),elapsed_seconds=time.monotonic()-start,
        policy='128-bit hash matches conservatively reject candidate orbits. Collisions can reject extra candidates but cannot admit an identical historical input.')
    json_write(meta,m);return path,m


def generate_world(config,unit,index,history_metadata,root):
    root.mkdir(parents=True,exist_ok=True);target=root/'data.npz';auditpath=root/'data_audit.json'
    if target.exists():
        a=json.loads(auditpath.read_text());assert a['data_sha256']==sha(target) and a['data_seed']==unit['data_seed']
        return
    rng=np.random.default_rng(unit['data_seed']);data={};seen=set();audit=[];started=time.monotonic()
    width=64
    for split,per_length in config['data']['examples_per_length'].items():
        perms=[];ns=[];candidate_attempts=0
        for n in config['lengths']:
            accepted=0
            while accepted<per_length:
                candidate_attempts+=1
                if candidate_attempts>config['data']['maximum_attempts_per_world']:raise RuntimeError('Fixed fresh-orbit pool exhausted; do not silently change lengths or historical exclusion.')
                p=tuple(map(int,rng.permutation(n)+1));members=orbit(p)
                if len(set(members))!=8:continue
                key=min(input_key(q) for q in members)
                if key in seen or contains(index,[fingerprint(q) for q in members]).any():continue
                seen.add(key);perms.append(p);ns.append(n);accepted+=1
        padded=np.zeros((len(perms),30),np.uint8)
        for i,p in enumerate(perms):padded[i,:len(p)]=p
        data[f'{split}_permutations']=padded;data[f'{split}_lengths']=np.asarray(ns,np.int16)
        acts=ACTIONS[:1] if split=='train' else ACTIONS
        y=np.zeros((len(perms),len(acts),len(TASKS)),np.uint8)
        inputs={e:np.full((len(perms),len(acts),width),TOKEN_TO_ID['<PAD>'],np.uint16) for e in ENCODINGS}
        for i,p in enumerate(perms):
            for k,action in enumerate(acts):
                q=transform(p,action);y[i,k]=labels(q)
                for e in ENCODINGS:
                    ids=encode(q,e);inputs[e][i,k,:len(ids)]=ids
            for relation in RELATIONS:
                if len(acts)==3:
                    k=ACTIONS.index(relation['input_transform'])
                    assert y[i,k,TASKS.index(relation['right'])]==int(y[i,0,TASKS.index(relation['left'])])+relation['right_label_offset']
        data[f'{split}_labels']=y
        for e in ENCODINGS:data[f'{split}_{e}_input']=inputs[e]
        audit.append(dict(split=split,anchors=len(perms),per_length=per_length,attempts=candidate_attempts,actions=list(acts)))
        print({'world':unit['unit'],'split':split,'anchors':len(perms)},flush=True)
    schedule_rng=np.random.default_rng(unit['batch_seed']);updates=config['training']['updates'];effective=config['training']['effective_batch']
    period=len(config['lengths'])
    schedule_lengths=np.concatenate([schedule_rng.permutation(config['lengths']) for _ in range((updates+period-1)//period)])[:updates]
    train_ns=data['train_lengths'];by_n={n:np.flatnonzero(train_ns==n) for n in config['lengths']}
    schedule=np.stack([schedule_rng.choice(by_n[int(n)],effective,replace=False) for n in schedule_lengths]).astype(np.int32)
    np.save(root/'batch_indices.npy',schedule);np.save(root/'batch_lengths.npy',schedule_lengths)
    np.savez_compressed(target,**data)
    json_write(auditpath,dict(unit=unit['unit'],data_seed=unit['data_seed'],batch_seed=unit['batch_seed'],data_sha256=sha(target),
        batch_indices_sha256=sha(root/'batch_indices.npy'),batch_lengths_sha256=sha(root/'batch_lengths.npy'),
        historical_index=history_metadata,split_audit=audit,orbit_count=len(seen),within_world_orbit_overlap=0,
        across_encodings_identical_objects_labels_splits_batches=True,cross_world_disjointness_enforced=False,
        independent_world_policy='Independent seeded sampling conditional on the fixed historical exclusion universe. Different worlds may overlap; no cross-world model pairs are primary endpoints.',
        elapsed_seconds=time.monotonic()-started))


def prepare(config_path,units=None):
    config=verify_frozen(config_path);root=WORKSPACE/config['output'];root.mkdir(parents=True,exist_ok=True)
    path,meta=build_history(config,root);index=np.load(path,mmap_mode='r')
    for unit in config['units']:
        if units and unit['unit'] not in units:continue
        generate_world(config,unit,index,meta,root/unit['unit'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--unit',action='append');a=p.parse_args();prepare(a.config,a.unit)
