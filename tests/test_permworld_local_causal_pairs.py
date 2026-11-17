"""Independent permutation math and split-leakage checks, no neural outputs."""
from itertools import permutations

import numpy as np
import pytest

from experiments.permworld_local_causal_20261110.pairs import (
    permutation, complement, extrema_mask, resolved_count, motif_pair,
    count_matched_pair, same_label_donor, audit_pair, build_family, task_pair,
    complement_key, family_keys, build_candidates, partition_components, select_quotas,
)


def independent_positions(p,task='peaks'):
    # Deliberately no call to production extrema or audit code.
    if task=='peaks':
        return {i for i in range(2,len(p)) if p[i-2]<p[i-1] and p[i-1]>p[i]}
    return {i for i in range(2,len(p)) if p[i-2]>p[i-1] and p[i-1]<p[i]}


def test_all_simple_pairs_n4_through8_independent_local_and_prefix_counts():
    for n in range(4,9):
        for p in permutations(range(1,n+1)):
            for i in range(2,n-1):
                before,after=motif_pair(p,i)
                a,b=independent_positions(before),independent_positions(after)
                assert b-a=={i} and a-b==set()
                assert after[i+1:]==before[i+1:]
                assert sum(k<=i for k in b)-sum(k<=i for k in a)==1
                assert len(b)-len(a)==1
                assert independent_positions(complement(before),'valleys')==a
                assert independent_positions(complement(after),'valleys')==b


def test_all_count_matched_n8_anchors_independent_two_position_differences():
    for p in permutations(range(1,9)):
        a,b,k=count_matched_pair(p,6,2)
        x,y=independent_positions(a),independent_positions(b)
        assert y-x=={6} and x-y=={2} and k==2
        assert len(x)==len(y)
        assert sum(i<=6 for i in x)==sum(i<=6 for i in y)
        assert a[7:]==b[7:]


def test_disjoint_boundaries_all_centers_random_larger_permutations():
    rng=np.random.default_rng(112)
    for n in (10,15,30):
        for _ in range(10):
            p=tuple(rng.permutation(n)+1)
            for i in range(6,n-1):
                for k in range(2,i-3):
                    a,b,got=count_matched_pair(p,i,k)
                    x,y=independent_positions(a),independent_positions(b)
                    assert x-y=={k} and y-x=={i} and got==k
                    assert a[i+1:]==b[i+1:]
                    assert len(x)==len(y)


@pytest.mark.parametrize('n,i',[(4,2),(8,5),(10,2),(30,5)])
def test_count_matching_unavailable_before_i6(n,i):
    with pytest.raises(ValueError,match='i>=6'):
        count_matched_pair(tuple(range(1,n+1)),i)


def test_same_label_donors_keep_local_triple_suffix_and_counts():
    rng=np.random.default_rng(41)
    available,missing=0,0
    for n in (10,15,30):
        for _ in range(6):
            p=tuple(rng.permutation(n)+1)
            for i in range(2,n-1):
                for recipient in motif_pair(p,i):
                    donor,variant=same_label_donor(recipient,i,117,max_attempts=16,with_kind=True)
                    assert (donor,variant)==same_label_donor(recipient,i,117,max_attempts=16,with_kind=True)
                    if donor is None:
                        missing+=1
                        continue
                    available+=1
                    a,b=independent_positions(recipient),independent_positions(donor)
                    assert donor!=recipient
                    assert recipient[i+1:]==donor[i+1:]
                    if variant=='earlier_only':assert recipient[i-2:]==donor[i-2:]
                    assert (i in a)==(i in b)
                    assert len(a)==len(b)
                    assert sum(k<=i for k in a)==sum(k<=i for k in b)
    assert available>400 and missing==0


def test_early_same_label_control_uses_full_read_prefix():
    for i in (2,3):
        family=build_family(tuple(range(1,11)),i,unit='synthetic',root_split='test',root_index=i)
        assert len(family['pairs'])==4
        controls=[r for r in family['pairs'] if r['kind']=='same_label']
        assert len(controls)==2 and all(r['same_label_variant']=='full_read_prefix' for r in controls)
        assert any(r['kind']=='count_matched' for r in family['missing_controls'])


def test_both_directions_and_complement_task_labels_are_exact():
    p=(4,1,9,2,8,5,10,3,7,6)
    family=build_family(p,7,unit='u',root_split='test',root_index=15)
    assert family==build_family(p,7,unit='u',root_split='test',root_index=15)
    simple=[r for r in family['pairs'] if r['kind']=='simple']
    assert {r['direction'] for r in simple}=={-1,1}
    assert simple[0]['recipient']==simple[1]['donor']
    for record in family['pairs']:
        peak=task_pair(record,'peaks');valley=task_pair(record,'valleys')
        assert peak['local_delta']==valley['local_delta']
        assert peak['recipient_prefix_count']==valley['recipient_prefix_count']
        assert independent_positions(valley['recipient'],'valleys')==independent_positions(peak['recipient'])
        assert valley['read_after_position']==8 and valley['token_index']==18
        assert independent_positions(valley['actual_local_counterfactual'],'valleys')==independent_positions(peak['actual_local_counterfactual'])
        actual=independent_positions(peak['actual_local_counterfactual'])
        wanted=record.get('intended_direction',record['direction'])
        assert len(actual)-peak['recipient_total_count']==wanted
        if record['kind']=='count_matched':
            assert record['earlier_center']==2
            assert record['actual_local_counterfactual']!=record['donor']


def test_actual_one_line_token_point_is_second_local_endpoint():
    from experiments.permworld_exceedance_alignment_20261110.oracles import encoded_tokens
    p=(4,1,9,2,8,5,10,3,7,6)
    for i in range(2,9):
        a,b=motif_pair(p,i)
        tokens=encoded_tokens(a,'one_line')
        at=4+2*i
        assert tokens[at]==f'{a[i]:02d}'
        # Exactly the unread suffix follows this value and its next separator.
        assert tokens[at+2]==(f'{a[i+1]:02d}' if i+1<len(a) else '<ONE_END>')


def test_component_test_priority_keeps_entire_derived_family_together():
    p=(4,1,9,2,8,5,10,3,7,6)
    a=build_family(p,6,unit='u',root_split='validation',root_index=1)
    # A test root equal to a generated donor creates a component bridge.
    b=build_family(complement(a['pairs'][0]['donor']),5,unit='u',root_split='test',root_index=2)
    assigned,components=partition_components([a,b])
    assert len(components)==1
    assert {f['split'] for f in assigned}=={'test','excluded'}
    assert next(f for f in assigned if f['root_split']=='validation')['exclusion_reason']=='validation_family_component_touches_test_root'
    again,_=partition_components([b,a])
    assert assigned==again


def test_original_training_collision_excludes_complete_component():
    p=(4,1,9,2,8,5,10,3,7,6)
    f=build_family(p,7,unit='u',root_split='test',root_index=1)
    donor=next(r['donor'] for r in f['pairs'] if r['kind']=='same_label')
    assigned,components=partition_components([f],train_permutations=[complement(donor)])
    assert assigned[0]['split']=='excluded'
    assert components[0]['training_collision']


def test_fit_dev_test_support_disjoint_and_all_missing_cells_reported():
    rng=np.random.default_rng(19)
    families=[]
    for index in range(50):
        families.append(build_family(rng.permutation(10)+1,6,unit='u',
                         root_split='test' if index<10 else 'validation',root_index=index))
    assigned,_=partition_components(families,dev_fraction=.5)
    support={split:set() for split in ('fit','dev','test')}
    for f in assigned:
        if f['split']!='excluded':support[f['split']].update(family_keys(f))
    assert all(support.values())
    assert not support['fit']&support['dev']
    assert not support['fit']&support['test']
    assert not support['dev']&support['test']
    selected,coverage=select_quotas(assigned,{'fit':5,'dev':5,'test':12})
    assert len(coverage)==3
    test=next(r for r in coverage if r['split']=='test')
    assert test['selected_families']==10 and test['shortfall']==2
    _,empty=select_quotas([dict(families[0],split='excluded')],{'fit':8,'dev':8,'test':32})
    assert len(empty)==3 and all(r['selected_families']==0 and r['shortfall']>0 for r in empty)


def test_candidate_selection_bound_and_order_are_deterministic():
    rng=np.random.default_rng(93)
    data={f'{s}_lengths':np.full(20,10) for s in ('validation','test')}
    data.update({f'{s}_permutations':np.array([rng.permutation(10)+1 for _ in range(20)]) for s in ('validation','test')})
    kwargs=dict(unit='u',candidate_roots_per_cell=2,lengths=[10],control_seed=14,max_control_attempts=8)
    a=build_candidates(data,**kwargs);b=build_candidates(data,**kwargs)
    assert a==b and len(a)==2*7*2
    limited=build_candidates(data,root_indices={'validation':[0],'test':[1]},**kwargs)
    assert len(limited)==2*7
    assert {f['root_index'] for f in limited if f['root_split']=='validation'}=={0}


@pytest.mark.parametrize('bad',[[],[1,1],[0,1],[True,2],[1.0,2.0]])
def test_malformed_permutation_rejected(bad):
    with pytest.raises(ValueError):permutation(bad)


def test_false_endpoint_and_prefix_count_convention():
    p=(1,5,2,6,3,7,4,8)
    assert extrema_mask(p)==(False,True,False,True,False,True,False,False)
    assert resolved_count(p,4)==2
    assert complement_key(p)==complement_key(complement(p))
