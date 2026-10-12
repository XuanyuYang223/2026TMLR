"""Lossless matched-length encodings and shared mathematical-object protocol."""
import hashlib
import json
from pathlib import Path
import sys
import tomllib
import numpy as np
import torch

WORKSPACE=Path(__file__).resolve().parents[2]
UPSTREAM=WORKSPACE/'external/neurips/src'
sys.path.insert(0,str(UPSTREAM))
from neurips_permutations.passage import VOCABULARY,TOKEN_TO_ID,ID_TO_TOKEN,one_line_tokens,cycle_tokens,canonical_cycles,validate_permutation
from neurips_permutations.math_ops import PROPERTY_FUNCTIONS
from neurips_permutations.models import CausalTransformer
from experiments.permutation_audit import transform,TRANSFORMS

ENCODINGS=('one_line','canonical_cycle')
RELATIONS=tomllib.loads((WORKSPACE/'external/neurips/configs/property_task_geometry.toml').read_text())['relation_pairs']
TASKS=tuple(sorted({r[s] for r in RELATIONS for s in ['left','right']}))
ACTIONS=('identity','complement','inverse')


def encode(p,encoding):
    p=validate_permutation(p)
    if not 1<=len(p)<=30:raise ValueError('Frozen protocol supports n=1..30, one atomic number each.')
    if encoding=='one_line':body=one_line_tokens(p)
    elif encoding=='canonical_cycle':body=cycle_tokens(p)
    else:raise ValueError(encoding)
    tokens=('<BOS>','<SIZE>',f'{len(p):02d}',*body)
    assert len(tokens)==2*len(p)+4
    return tuple(TOKEN_TO_ID[t] for t in tokens)


def decode(ids,encoding):
    try:tokens=tuple(ID_TO_TOKEN[int(t)] for t in ids)
    except (KeyError,TypeError):raise ValueError('Invalid token IDs') from None
    if len(tokens)<6 or tokens[:2]!=('<BOS>','<SIZE>'):raise ValueError('Invalid prefix')
    try:n=int(tokens[2])
    except ValueError:raise ValueError('Invalid size') from None
    if not 1<=n<=30 or len(tokens)!=2*n+4:raise ValueError('Invalid length')
    start,end=('<ONE_START>','<ONE_END>') if encoding=='one_line' else ('<CYCLE_START>','<CYCLE_END>')
    if encoding not in ENCODINGS or tokens[3]!=start or tokens[-1]!=end:raise ValueError('Invalid boundaries')
    numbers=[];groups=[[]]
    for i,t in enumerate(tokens[4:-1]):
        if i%2==0:
            if t not in VOCABULARY[:100]:raise ValueError('Expected atomic number')
            v=int(t);numbers.append(v);groups[-1].append(v)
        else:
            if t not in (',',';') or (encoding=='one_line' and t!=','):raise ValueError('Invalid separator')
            if t==';':groups.append([])
    validate_permutation(numbers)
    if encoding=='one_line':p=tuple(numbers)
    else:
        if any(g[0]!=min(g) for g in groups) or [g[0] for g in groups]!=sorted(g[0] for g in groups):raise ValueError('Noncanonical cycles')
        p=[0]*n
        for g in groups:
            for a,b in zip(g,g[1:]+g[:1]):p[a-1]=b
        p=tuple(p)
    # Reject alternate numeric spellings or noncanonical serialized representations.
    if encode(p,encoding)!=tuple(map(int,ids)):raise ValueError('Noncanonical token sequence')
    return p


def labels(p):return tuple(int(PROPERTY_FUNCTIONS[t](p)) for t in TASKS)


def input_key(p):return b'['+b','.join(str(int(v)).encode() for v in p)+b']'


def fingerprint(p):return hashlib.blake2b(input_key(p),digest_size=16).digest()


def orbit(p):return tuple(transform(p,a) for a in TRANSFORMS)


def orbit_key(p):return min(input_key(q) for q in orbit(p))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def parameter_hash(model):
    h=hashlib.sha256()
    for name,v in sorted(model.state_dict().items()):
        a=v.detach().cpu().contiguous().numpy()
        for value in (name,str(a.dtype),str(a.shape)):h.update(value.encode())
        h.update(a.tobytes())
    return h.hexdigest()


def make_model(config,seed):
    import random
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    m=config['model']
    return CausalTransformer(vocab_size=len(VOCABULARY),max_seq_len=m['max_seq_len'],d_model=m['d_model'],layers=m['layers'],
        dropout=m['dropout'],n_heads=m['heads'],mlp_ratio=m['mlp_ratio'],tie_embeddings=m['tie_embeddings'])


def batch(prefixes,ns,task,answers=None,device='cpu'):
    """Prompt or answer+EOS supervised batch; prefix ends strictly before task."""
    ns=np.asarray(ns);width=2*int(ns.max())+6+(2 if answers is not None else 0)
    ids=torch.full((len(ns),width),TOKEN_TO_ID['<PAD>'],dtype=torch.long,device=device)
    target=torch.full_like(ids,-100);end=torch.as_tensor(2*ns+3,dtype=torch.long,device=device)
    rows=torch.arange(len(ns),device=device)
    prefix_width=2*int(ns.max())+4
    ids[:,:prefix_width]=torch.as_tensor(np.array(prefixes[:,:prefix_width],copy=True),device=device)
    ids[rows,end+1]=TOKEN_TO_ID[f'<{task.upper()}>'];ids[rows,end+2]=TOKEN_TO_ID['=']
    if answers is not None:
        answerids=torch.as_tensor([TOKEN_TO_ID[f'{int(v):02d}'] for v in answers],device=device)
        ids[rows,end+3]=answerids;ids[rows,end+4]=TOKEN_TO_ID['<EOS>']
        target[rows,end+3]=answerids;target[rows,end+4]=TOKEN_TO_ID['<EOS>']
    return ids,ids.ne(TOKEN_TO_ID['<PAD>']),target,end,end+2


def loss(logits,target):
    import torch.nn.functional as F
    return F.cross_entropy(logits[:,:-1].reshape(-1,logits.shape[-1]),target[:,1:].reshape(-1),ignore_index=-100)


def json_write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def load_config(path):return json.loads(Path(path).read_text())


def verify_frozen(config_path):
    config_path=Path(config_path);f=json.loads(config_path.with_name('freeze_manifest.json').read_text())
    if sha(config_path)!=f['config_sha256']:raise RuntimeError('Protocol changed after freeze')
    for path,h in f['code_sha256'].items():
        if sha(WORKSPACE/path)!=h:raise RuntimeError(f'Frozen code changed: {path}')
    return load_config(config_path)
