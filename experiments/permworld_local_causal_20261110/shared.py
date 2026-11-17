from pathlib import Path
import csv
import gzip
import hashlib
import json
import os

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PACKAGE=Path(__file__).resolve().parent
DIAG=ROOT/'diagnostics/permworld_local_causal_20261110'
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')
TASKS=('peaks','valleys')
RANDOM_SEEDS=tuple(range(1026,1034))
CONDITIONS=('unedited','local_projection','same_label_projection',
            *[f'random_direction_{s}' for s in RANDOM_SEEDS],
            'full_state_swap','actual_counterfactual')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)


def write_csv(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def families(unit):
    receipt=read(DIAG/'data'/unit/'completion.json')
    path=OUT/unit/'selected_families.json.gz'
    assert receipt['status']=='complete'
    assert sha(path)==receipt['output_sha256'][str(path)]
    with gzip.open(path,'rt') as stream:
        return json.load(stream)


def ids_for(permutations,task):
    from experiments.permworld_peaks_probe_20261109.vendor.passage import TOKEN_TO_ID as vocab
    n=len(permutations[0])
    assert all(len(p)==n for p in permutations)
    assert all(vocab[f'{k:02d}']==k for k in range(31))
    body=[]
    for p in permutations:
        seq=[vocab['<BOS>'],vocab['<SIZE>'],vocab[f'{n:02d}'],vocab['<ONE_START>']]
        for i,value in enumerate(p):
            if i:seq.append(vocab[','])
            seq.append(vocab[f'{int(value):02d}'])
        seq.extend([vocab['<ONE_END>'],vocab[f'<{task.upper()}>'],vocab['=']])
        assert len(seq)==2*n+6
        body.append(seq)
    return np.asarray(body,dtype=np.int64)


def transformed(p,task):
    from .pairs import complement
    return tuple(p) if task=='peaks' else complement(p)


def catalog(selected,task,pools):
    """Unique natural prefixes; no new outcome-dependent sample selection."""
    from .pairs import extrema_mask,resolved_count
    nodes={}
    for f in selected:
        if f['split'] not in pools:continue
        if f['split']=='fit':
            records=[r for r in f['pairs'] if r['kind']=='count_matched' and r['direction']==1]
        elif f['split']=='dev':
            records=[r for r in f['pairs'] if r['kind'] in ('simple','count_matched')]
        else:records=f['pairs']
        for r in records:
            sides=['recipient','donor']
            if f['split']=='test':sides.append('actual_local_counterfactual')
            for side in sides:
                p=transformed(r[side],task);n,i=f['n'],f['center']
                key=(n,i,p)
                if key in nodes:
                    assert nodes[key]['pool']==f['split']
                    if f['split']=='fit':assert nodes[key]['fit_partition']==f['fit_partition']
                    continue
                mask=extrema_mask(p,task)
                node_id=hashlib.sha256((f['unit']+':'+task+':'+str(i)).encode()+bytes(p)).hexdigest()[:24]
                nodes[key]=dict(node_id=node_id,n=n,position=i,permutation=p,pool=f['split'],
                    fit_partition=f.get('fit_partition'),component=f['component_id'],
                    label=int(mask[i-1]),resolved_count=resolved_count(p,i,task),total_count=sum(mask))
    ordered=[nodes[k] for k in sorted(nodes)]
    return ordered,{(r['n'],r['position'],tuple(r['permutation'])):i for i,r in enumerate(ordered)}
