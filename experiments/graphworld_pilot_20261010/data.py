"""Fixed graph world, full<R,C,W> orbit exclusion and common batches."""
from collections import Counter
import time
import numpy as np
from .common import *

def wrong_witness(n):
    mapping=dict(zip(pairs(n),wrong_table(n)))
    for p in pairs(n):
        for q in pairs(n):
            if len(set(p)&set(q))!=len(set(mapping[p])&set(mapping[q])):
                return dict(first=list(p),second=list(q),mapped_first=list(mapping[p]),mapped_second=list(mapping[q]),
                    original_intersection=len(set(p)&set(q)),mapped_intersection=len(set(mapping[p])&set(mapping[q])))
    raise AssertionError('Frozen W has no incidence-violation witness; refuse unverified construction')

def mathematics():
    """No model calls: exact transformations and NetworkX3.6.1 cross-check."""
    import networkx as nx
    rng=np.random.default_rng(2026101003);records=[];count=0
    for n in (6,7,8):
        incidental=np.zeros(4,np.int64);examples={}
        for sample in range(3000):
            p=(.25,.5,.75)[sample%3];a=(rng.random((n,n))<p).astype(np.uint8);np.fill_diagonal(a,0)
            y=labels(a)
            for t in ('R','C','W'):assert np.array_equal(transform(transform(a,t),t),a)
            for t,u in (('R','C'),('R','W'),('C','W')):
                assert np.array_equal(transform(transform(a,t),u),transform(transform(a,u),t))
            assert int(transform(a,'W').sum())==int(a.sum())
            assert len(orbit(a))<=8
            for k,r in enumerate(RELATIONS):
                yc=mapped_right(labels(transform(a,r['correct']))[TASKS.index(r['right'])],n,r)
                yw=mapped_right(labels(transform(a,r['wrong']))[TASKS.index(r['right'])],n,r)
                yl=int(y[TASKS.index(r['left'])]);assert yc==yl
                incidental[k]+=int(yw==yl)
                if yw!=yl and r['relation'] not in examples:
                    examples[r['relation']]=dict(adjacency=a.tolist(),left_answer=yl,correct_mapped_right=int(yc),wrong_mapped_right=int(yw),
                                                wrong_adjacency=transform(a,r['wrong']).tolist())
            if sample<32:
                g=nx.from_numpy_array(a,create_using=nx.DiGraph)
                assert np.array_equal(nx.to_numpy_array(g.reverse(),dtype=np.uint8,nodelist=range(n)),transform(a,'R'))
                assert np.array_equal(nx.to_numpy_array(nx.complement(g),dtype=np.uint8,nodelist=range(n)),transform(a,'C'))
            if sample<128:
                assert np.array_equal(decode(encode(a)),a)
            count+=1
        assert len(examples)==4
        table=[dict(source=list(p),target=list(q)) for p,q in zip(pairs(n),wrong_table(n))]
        records.append(dict(n=n,seed=20261010+n,actual_wrong_table=table,wrong_table_sha256=payload_sha(table),
                            non_vertex_induced_witness=wrong_witness(n),counterexamples=examples,random_samples=3000,
                            accidental_relation_fraction={r['relation']:int(v)/3000 for r,v in zip(RELATIONS,incidental)}))
    # Exact exhaustive check of the structural identities at all4096 loopless n4 graphs.
    slots=[(i,j) for i in range(4) for j in range(4) if i!=j]
    for bits in range(1<<12):
        a=np.zeros((4,4),np.uint8)
        for k,(i,j) in enumerate(slots):a[i,j]=(bits>>k)&1
        for r in RELATIONS:
            assert labels(a)[TASKS.index(r['left'])]==mapped_right(labels(transform(a,r['correct']))[TASKS.index(r['right'])],4,r)
    out=dict(status='passed',random_graphs=count,exhaustive_n4_graphs=4096,networkx_version=nx.__version__,
             formulas_proved_by_degrees=True,all_involutions_and_commutations=True,all_W_non_vertex_induced=True,records=records,
             rng_seed=2026101003,accidental_matches_retained=True,source_updates=0,new_model_calls=0)
    json_write(DIAG/'mathematics_audit.json',out)
    return out

def prepare(c):
    directory=ROOT/'world';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'data_audit.json').exists():
        audit=read(directory/'data_audit.json')
        assert audit['protocol_sha256']==sha(CONFIG)
        assert sha(directory/'data.npz')==audit['data_sha256'] and sha(directory/'batch_indices.npy')==audit['batch_indices_sha256']
        return audit
    begun=time.monotonic();rng=np.random.default_rng(c['data_seed']);seen=set();arrays={};anchors=[];split_counts={};rejections=Counter();all_member_sets={}
    for split,size in c['data']['per_n'].items():
        graphs=[];ns=[];probs=[];ids=[];orbitids=[];edges=[];labels_all=[];prefixes=[];members=set()
        for n in c['lengths']:
            accepted=0
            while accepted<size:
                pi=int(rng.integers(3));p=c['data']['p'][pi]
                a=(rng.random((n,n))<p).astype(np.uint8);np.fill_diagonal(a,0)
                o=orbit(a);ok=min(map(key,o))
                if ok in seen:rejections[f'{split}_n{n}_p{p}']+=1;continue
                seen.add(ok);members.update(map(key,o));oid=hashlib.sha256(ok).hexdigest()
                actions=[transform(a,t) for t in ACTIONS];g=np.zeros((5,8,8),np.uint8);x=np.full((5,68),TOKEN_TO_ID['<PAD>'],np.int16)
                for ai,b in enumerate(actions):g[ai,:n,:n]=b;x[ai,:n*n+4]=encode(b)
                aid=f'{split}_n{n}_{accepted:05d}'
                graphs.append(g);prefixes.append(x);ns.append(n);probs.append(pi);ids.append(aid);orbitids.append(oid)
                edges.append([int(b.sum()) for b in actions]);labels_all.append([labels(b) for b in actions])
                anchors.append(dict(anchor_id=aid,split=split,n=n,p=p,m=int(a.sum()),orbit_id=oid,orbit_size=len(o)))
                accepted+=1
        arrays[split+'_graphs']=np.asarray(graphs,np.uint8);arrays[split+'_input']=np.asarray(prefixes,np.int16)
        arrays[split+'_lengths']=np.asarray(ns,np.int16);arrays[split+'_p_index']=np.asarray(probs,np.int8)
        arrays[split+'_anchor_id']=np.asarray(ids,dtype='U32');arrays[split+'_orbit_id']=np.asarray(orbitids,dtype='U64')
        arrays[split+'_edges']=np.asarray(edges,np.int16);arrays[split+'_labels']=np.asarray(labels_all,np.int16)
        split_counts[split]={str(n):dict(Counter(str(c['data']['p'][int(pi)]) for pi,nn in zip(probs,ns) if nn==n)) for n in c['lengths']}
        all_member_sets[split]=members
    assert len(seen)==sum(c['data']['per_n'].values())*3
    for s in all_member_sets:
        for t in all_member_sets:
            if s!=t:assert not (all_member_sets[s]&all_member_sets[t])
    np.savez_compressed(directory/'data.npz',**arrays)
    # Uniform permutations of the3 n values, then fixed64 distinct anchors/update.
    b_rng=np.random.default_rng(c['batch_seed']);updates=c['training']['updates']
    length_schedule=np.concatenate([b_rng.permutation(c['lengths']) for _ in range((updates+2)//3)])[:updates]
    pool={n:np.flatnonzero(arrays['train_lengths']==n) for n in c['lengths']}
    batches=np.asarray([b_rng.choice(pool[int(n)],size=64,replace=False) for n in length_schedule],np.int32)
    np.save(directory/'batch_indices.npy',batches);np.save(directory/'length_schedule.npy',length_schedule)
    csv_write(DIAG/'anchor_index.csv',anchors)
    baselines=[]
    for n in c['lengths']:
        for ti,t in enumerate(TASKS):
            y=arrays['train_labels'][arrays['train_lengths']==n,0,ti]
            mode=int(np.argmax(np.bincount(y,minlength=8)))
            row=dict(n=n,task=t,training_mode=mode,train_accuracy=float((y==mode).mean()))
            for split in ('validation','analysis'):
                yy=arrays[split+'_labels'][arrays[split+'_lengths']==n,0,ti]
                row[split+'_accuracy']=float((yy==mode).mean())
            baselines.append(row)
    csv_write(DIAG/'training_mode_baseline.csv',baselines)
    audit=dict(status='passed',protocol_sha256=sha(CONFIG),data_sha256=sha(directory/'data.npz'),
               batch_indices_sha256=sha(directory/'batch_indices.npy'),length_schedule_sha256=sha(directory/'length_schedule.npy'),
               anchor_index_sha256=sha(DIAG/'anchor_index.csv'),p_counts=split_counts,rejections=dict(rejections),
               selected_anchors=len(seen),full_orbit_cross_split_overlap=0,orbit_member_counts={s:len(v) for s,v in all_member_sets.items()},
               length_schedule_counts={str(n):int((length_schedule==n).sum()) for n in c['lengths']},
               independent_world_count=1,graph_isomorphisms_not_excluded=True,wall_seconds=time.monotonic()-begun,
               data_bytes=(directory/'data.npz').stat().st_size+(directory/'batch_indices.npy').stat().st_size,
               no_gpu_calls=True,no_training=True)
    json_write(directory/'data_audit.json',audit);json_write(DIAG/'data_audit.json',audit)
    return audit
