"""Deterministic counterfactual local-extremum pairs and component isolation.

No neural model, activation, prediction, or test outcome is accessed. Positions
are one-based; extrema masks include two false endpoint entries. Counterfactual
permutations derive only from the supplied existing anchor permutations.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import combinations
from numbers import Integral

import numpy as np

VERSION = 'local-causal-pairs/v1'


def permutation(values):
    p=tuple(values)
    if not p or any(isinstance(x,(bool,np.bool_)) or not isinstance(x,Integral) for x in p):
        raise ValueError('A nonempty integer permutation is required')
    p=tuple(int(x) for x in p)
    if sorted(p)!=list(range(1,len(p)+1)):
        raise ValueError('Expected a permutation of 1..n')
    return p


def complement(values):
    p=permutation(values)
    return tuple(len(p)+1-x for x in p)


def extrema_mask(values,task='peaks'):
    """Independent direct-comparison oracle, including false endpoints."""
    p=permutation(values)
    if task not in ('peaks','valleys'):
        raise ValueError(task)
    out=[False]*len(p)
    for at in range(1,len(p)-1):
        a,b,c=p[at-1:at+2]
        out[at]=(a<b>c) if task=='peaks' else (a>b<c)
    return tuple(out)


def resolved_count(values,center,task='peaks'):
    """Extrema through center i, resolved after reading pi_(i+1)."""
    p=permutation(values)
    if not 2<=center<=len(p)-2:
        raise ValueError('Quartet center requires 2 <= i <= n-2')
    return sum(extrema_mask(p,task)[:center])


def _center(p,i):
    if not isinstance(i,Integral) or not 2<=i<=len(p)-2:
        raise ValueError('Quartet center requires 2 <= i <= n-2')
    return int(i)


def _ascending(p,i):
    q=list(p)
    q[i-2:i+2]=sorted(q[i-2:i+2])
    return tuple(q)


def _activate(p,i):
    q=list(p)
    q[i-1],q[i]=q[i],q[i-1]
    return tuple(q)


def motif_pair(anchor,center):
    p=permutation(anchor);i=_center(p,center)
    inactive=_ascending(p,i)
    active=_activate(inactive,i)
    audit_pair(inactive,active,i,'simple')
    return inactive,active


def count_matched_pair(anchor,center,earlier_center=None):
    p=permutation(anchor);i=_center(p,center)
    if i<6:
        raise ValueError('Count-matched disjoint quartets require i>=6')
    if earlier_center is None:
        earlier_center=2
    k=int(earlier_center)
    if not 2<=k<=i-4:
        raise ValueError('Earlier center must satisfy 2 <= k <= i-4')
    both_inactive=_ascending(_ascending(p,i),k)
    recipient=_activate(both_inactive,k)
    donor=_activate(both_inactive,i)
    audit_pair(recipient,donor,i,'count_matched',k)
    return recipient,donor,k


def same_label_donor(recipient,center,seed=0,max_attempts=64,with_kind=False):
    """Earlier-only variation first, then permute the full read prefix1..i+1.

    Both variants preserve local label, resolved count, total count, and the
    complete future suffix. Earlier-only additionally fixes the local triple.
    Each variant tries bounded seeded permutations then sorted transpositions.
    Missing controls are returned as None, never imputed.
    """
    p=permutation(recipient);i=_center(p,center)
    wanted=resolved_count(p,i)
    peak=extrema_mask(p)
    def acceptable(prefix,width):
        q=tuple(prefix)+p[width:]
        if q==p or resolved_count(q,i)!=wanted:
            return None
        own=extrema_mask(q)
        if own[i-1]!=peak[i-1] or sum(own)!=sum(peak):
            return None
        audit_pair(p,q,i,'same_label')
        return q
    for kind,width in [('earlier_only',i-2),('full_read_prefix',i+1)]:
        if width<2:
            continue
        code=hashlib.sha256(f'same-label-v1:{seed}:{kind}:'.encode()+bytes(p)+bytes([i])).digest()
        rng=np.random.default_rng(int.from_bytes(code[:8],'big'))
        for _ in range(max_attempts):
            q=acceptable(rng.permutation(p[:width]).tolist(),width)
            if q is not None:
                return (q,kind) if with_kind else q
        for a,b in combinations(range(width),2):
            prefix=list(p[:width]);prefix[a],prefix[b]=prefix[b],prefix[a]
            q=acceptable(prefix,width)
            if q is not None:
                return (q,kind) if with_kind else q
    return (None,None) if with_kind else None


def audit_pair(recipient,donor,center,kind,earlier_center=None,task='peaks'):
    """Reject any off-target extremum/count/suffix violation independently."""
    a,b=permutation(recipient),permutation(donor)
    if len(a)!=len(b):
        raise ValueError('Pair lengths differ')
    i=_center(a,center)
    ma,mb=extrema_mask(a,task),extrema_mask(b,task)
    diff={at+1:int(y)-int(x) for at,(x,y) in enumerate(zip(ma,mb)) if x!=y}
    assert a[i+1:]==b[i+1:], 'Post-read suffix mismatch'
    ra,rb=resolved_count(a,i,task),resolved_count(b,i,task)
    ta,tb=sum(ma),sum(mb)
    local=int(mb[i-1])-int(ma[i-1])
    if kind=='simple':
        assert set(diff)=={i} and abs(local)==1
        assert rb-ra==local and tb-ta==local
    elif kind=='count_matched':
        k=int(earlier_center)
        assert 2<=k<=i-4 and set(diff)=={k,i}
        assert abs(local)==1 and diff[k]==-local
        assert ra==rb and ta==tb
    elif kind=='same_label':
        assert a!=b
        assert local==0 and ra==rb and ta==tb
    else:
        raise ValueError(kind)
    return dict(n=len(a),center=i,read_after_position=i+1,token_index=4+2*i,
                recipient_label=int(ma[i-1]),donor_label=int(mb[i-1]),local_delta=local,
                recipient_prefix_count=ra,donor_prefix_count=rb,
                recipient_total_count=ta,donor_total_count=tb,
                changed_positions=sorted(diff),changed_deltas=[diff[k] for k in sorted(diff)],
                suffix_equal=True,local_triple_equal=a[i-2:i+1]==b[i-2:i+1])


def actual_local_counterfactual(recipient,center):
    """Only swap the local middle values; all other peak labels stay fixed."""
    p=permutation(recipient);i=_center(p,center)
    counterfactual=_activate(p,i)
    checked=audit_pair(p,counterfactual,i,'simple')
    return dict(actual_local_counterfactual=counterfactual,
                actual_local_counterfactual_label=checked['donor_label'],
                actual_local_counterfactual_prefix_count=checked['donor_prefix_count'],
                actual_local_counterfactual_total_count=checked['donor_total_count'])


def _digest(*pieces):
    h=hashlib.sha256()
    for piece in pieces:
        if isinstance(piece,str):
            piece=piece.encode()
        h.update(len(piece).to_bytes(8,'big'));h.update(piece)
    return h.hexdigest()


def build_family(anchor,center,*,unit,root_split,root_index,control_seed=0,max_control_attempts=64):
    """Both directions, count-matched pair where possible, same-label controls."""
    p=permutation(anchor);i=_center(p,center)
    if root_split not in ('validation','test'):
        raise ValueError('Only existing validation/test anchors are authorized')
    family_id=_digest(VERSION,unit,root_split,str(root_index),bytes(p),bytes([i]))[:24]
    family=dict(family_id=family_id,unit=unit,root_split=root_split,root_index=int(root_index),
                n=len(p),center=i,anchor=p,pairs=[],missing_controls=[])
    a,b=motif_pair(p,i)
    base=[('simple',a,b,None)]
    if i>=6:
        c,d,k=count_matched_pair(p,i)
        base.append(('count_matched',c,d,k))
    else:
        family['missing_controls'].append(dict(kind='count_matched',reason='disjoint_earlier_quartet_requires_i>=6'))
    for kind,left,right,k in base:
        for direction,(recipient,donor) in [(1,(left,right)),(-1,(right,left))]:
            identity=f'{family_id}:{kind}:{direction:+d}'
            record=dict(pair_id=identity,family_id=family_id,kind=kind,contrast=kind,
                        direction=direction,earlier_center=k,recipient=recipient,donor=donor,
                        **audit_pair(recipient,donor,i,kind,k),
                        **actual_local_counterfactual(recipient,i))
            assert record['actual_local_counterfactual_label']==record['donor_label']
            assert record['actual_local_counterfactual_prefix_count']-record['recipient_prefix_count']==direction
            assert record['actual_local_counterfactual_total_count']-record['recipient_total_count']==direction
            family['pairs'].append(record)
            same,variant=same_label_donor(recipient,i,control_seed,max_control_attempts,with_kind=True)
            if same is None:
                family['missing_controls'].append(dict(kind='same_label',control_for_pair_id=identity,
                     reason='no_changed_prefix_with_exact_label_and_count_under_bounded_search'))
            else:
                family['pairs'].append(dict(pair_id=identity+':same_label',family_id=family_id,
                    kind='same_label',contrast=kind,direction=0,intended_direction=direction,
                    same_label_variant=variant,
                    earlier_center=k,control_for_pair_id=identity,recipient=recipient,donor=same,
                    **audit_pair(recipient,same,i,'same_label'),
                    **actual_local_counterfactual(recipient,i)))
    return family


def task_pair(record,task):
    """Complement the complete pair for valleys; peak/valley positions agree."""
    if task not in ('peaks','valleys'):
        raise ValueError(task)
    result=dict(record)
    if task=='valleys':
        result['recipient']=complement(record['recipient'])
        result['donor']=complement(record['donor'])
        result['actual_local_counterfactual']=complement(record['actual_local_counterfactual'])
    result.update(audit_pair(result['recipient'],result['donor'],record['center'],
                             record['kind'],record['earlier_center'],task))
    result['task']=task
    return result


def complement_key(values):
    p=permutation(values)
    return min(bytes(p),bytes(complement(p)))


def family_keys(family):
    return {complement_key(family['anchor']),
            *[complement_key(r[side]) for r in family['pairs']
              for side in ('recipient','donor','actual_local_counterfactual')]}


def build_candidates(data,*,unit,candidate_roots_per_cell,lengths=range(10,31),
                     root_indices=None,control_seed=0,max_control_attempts=64):
    """Bounded hash-selected roots for every existing (split,n,i) cell.

    root_indices may provide permitted global anchor indices per split. It is
    the caller's responsibility to preserve any stronger original orbit rules.
    Component partitioning below then excludes identical/complement leakage
    among *all generated* inputs. Root selection never uses model outcomes.
    """
    if candidate_roots_per_cell<1:
        raise ValueError('A positive explicit engineering bound is required')
    records=[]
    for split in ('validation','test'):
        allowed=None if root_indices is None else set(map(int,root_indices[split]))
        for n in lengths:
            indices=np.flatnonzero(np.asarray(data[f'{split}_lengths'])==n)
            if allowed is not None:
                indices=np.asarray([k for k in indices if int(k) in allowed],dtype=np.int64)
            for i in range(2,n-1):
                ranked=sorted(indices,key=lambda idx:_digest('root-rank-v1',unit,split,str(n),str(i),
                              str(int(idx)),bytes(np.asarray(data[f'{split}_permutations'][idx,:n],dtype=np.uint8))))
                for index in ranked[:candidate_roots_per_cell]:
                    records.append(build_family(data[f'{split}_permutations'][index,:n],i,unit=unit,
                        root_split=split,root_index=int(index),control_seed=control_seed,
                        max_control_attempts=max_control_attempts))
    return records


def partition_components(families,*,train_permutations=None,dev_fraction=.2,namespace='local-causal-split-v1'):
    """Join complete families under equality/complement before assigning pools.

    A component touching a test root keeps only its test-root families; its
    validation families are excluded. Remaining validation components hash to
    fit/dev. An optional original-training collision excludes the *entire*
    component. No pair direction or same-label donor can cross pools.
    """
    if not 0<dev_fraction<1:
        raise ValueError('Expected a nondegenerate pre-fixed dev fraction')
    if len({f['unit'] for f in families})>1:
        raise ValueError('Partition each independent unit separately')
    parent=list(range(len(families)))
    def find(k):
        while parent[k]!=k:
            parent[k]=parent[parent[k]];k=parent[k]
        return k
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:
            parent[max(a,b)]=min(a,b)
    owner={};keys=[]
    for index,family in enumerate(families):
        own=family_keys(family);keys.append(own)
        for key in own:
            if key in owner:
                union(index,owner[key])
            else:
                owner[key]=index
    groups=defaultdict(list)
    for index in range(len(families)):
        groups[find(index)].append(index)
    training=set() if train_permutations is None else {complement_key(p) for p in train_permutations}
    assigned=[];components=[]
    for members in groups.values():
        all_keys=set().union(*(keys[k] for k in members))
        component=_digest(namespace,*sorted(all_keys))
        has_test=any(families[k]['root_split']=='test' for k in members)
        train_collision=bool(training&all_keys)
        draw=int(component[:16],16)/2**64
        pool='dev' if draw<dev_fraction else 'fit'
        for index in members:
            family=dict(families[index])
            if train_collision:
                split,reason='excluded','original_training_equality_or_complement_collision'
            elif has_test and family['root_split']!='test':
                split,reason='excluded','validation_family_component_touches_test_root'
            else:
                split,reason=('test' if has_test else pool),None
            family.update(component_id=component,split=split,exclusion_reason=reason)
            family['fit_partition']=('A' if int(_digest('fit-halves-v1',component)[:2],16)<128 else 'B') if split=='fit' else None
            assigned.append(family)
        components.append(dict(component_id=component,families=len(members),unique_complement_groups=len(all_keys),
            test_root_present=has_test,training_collision=train_collision,
            validation_pool=None if has_test or train_collision else pool))
    assigned.sort(key=lambda f:f['family_id'])
    # Verify direct equality/complement support is disjoint after exclusions.
    support=defaultdict(set)
    for f in assigned:
        if f['split']!='excluded':
            support[f['split']].update(family_keys(f))
    for a,b in combinations(support,2):
        assert not support[a]&support[b],(a,b,'generated-input split leakage')
    return assigned,components


def select_quotas(families,quotas):
    """Fixed per-(pool,n,i) family quotas, retaining all directions/controls."""
    grouped=defaultdict(list)
    for f in families:
        if f['split'] in quotas:
            grouped[(f['split'],f['n'],f['center'])].append(f)
    cells={(f['n'],f['center']) for f in families}
    for split in quotas:
        for n,i in cells:
            grouped.setdefault((split,n,i),[])
    selected=[];coverage=[]
    for (split,n,i),group in sorted(grouped.items()):
        wanted=int(quotas[split])
        if wanted<0:
            raise ValueError('Quotas must be nonnegative')
        chosen=sorted(group,key=lambda f:_digest('quota-rank-v1',f['family_id']))[:wanted]
        selected.extend(chosen)
        coverage.append(dict(split=split,n=n,center=i,available_families=len(group),
            requested_families=wanted,selected_families=len(chosen),shortfall=max(0,wanted-len(chosen)),
            count_matched_available=i>=6,
            same_label_directions=sum(r['kind']=='same_label' for f in chosen for r in f['pairs'])))
    return selected,coverage
