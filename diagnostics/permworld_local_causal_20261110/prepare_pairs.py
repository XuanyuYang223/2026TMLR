"""Prepare authorized counterfactual inputs only; never load a neural model."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.permworld_local_causal_20261110.pairs import build_candidates,partition_components,select_quotas

DIAG=Path(__file__).resolve().parent
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')
SOURCE_AUDIT=ROOT/'diagnostics/permworld_lemma_composition_20261110/cache_audit.json'
PAIRS=ROOT/'experiments/permworld_local_causal_20261110/pairs.py'
TESTS=ROOT/'tests/test_permworld_local_causal_pairs.py'
EXPECTED_PAIRS_SHA='61182a2ed5653de4a5c972913431ad909b900aa4f9c614706dd9442da7e871c4'
LENGTHS=tuple(range(10,31))
QUOTAS={'fit':12,'dev':6,'test':8}


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def csv_rows(path,records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    handle=gzip.open(path,'wt',newline='') if path.suffix=='.gz' else path.open('w',newline='')
    with handle as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)


def freeze(units):
    assert sha(PAIRS)==EXPECTED_PAIRS_SHA
    config=dict(version='local-causal-pair-generation/v1',source_audit_sha256=sha(SOURCE_AUDIT),
        source_sha256={str(p):sha(p) for p in [PAIRS,TESTS,Path(__file__),DIAG/'math_audit.json']},
        candidate_roots_per_cell=64,control_seed=611201,max_control_attempts=64,lengths=list(LENGTHS),
        centers='all i2..n-2',quotas=QUOTAS,dev_fraction=.2,split_namespace='local-causal-split-v1',
        earlier_center=2,training_exclusion='whole component touching original training p or Cp',
        components='original anchor and ALL recipient/donor/actual_local_counterfactual nodes under equality/complement',
        fit_partition='component hash A/B; only count-matched i>=6 intended for direction/checker fits',
        units=[dict(unit=u['unit'],historical=u['historical'],data_path=u['data_path'],data_sha256=u['data_sha256']) for u in units],
        output=str(OUT),no_neural_forward=True,no_model_outcome_access=True,no_model_training=True)
    path=DIAG/'pair_generation_protocol.json'
    if path.exists():assert read(path)==config,'Generation protocol changed after freeze'
    else:write(path,config)
    return config


def independent_selected_audit(selected,train_rows):
    """Direct raw comparisons, independent of production extrema/audit code."""
    def labels(p,task='peaks'):
        if task=='peaks':return {j for j in range(2,len(p)) if p[j-2]<p[j-1]>p[j]}
        return {j for j in range(2,len(p)) if p[j-2]>p[j-1]<p[j]}
    def key(p):return min(bytes(p),bytes(len(p)+1-x for x in p))
    training={key(p) for p in train_rows}
    pools=defaultdict(set);fit_halves=defaultdict(set);counts=Counter()
    for family in selected:
        own={key(family['anchor'])}
        for row in family['pairs']:
            r,d,c=[tuple(row[k]) for k in ('recipient','donor','actual_local_counterfactual')]
            n,i=row['n'],row['center']
            for p in (r,d,c):
                assert sorted(p)==list(range(1,n+1));own.add(key(p))
            a,b,z=labels(r),labels(d),labels(c)
            assert r[i+1:]==d[i+1:]==c[i+1:]
            prefix=lambda q:sum(j<=i for j in q)
            sign=row.get('intended_direction',row['direction'])
            assert z-a==({i} if sign==1 else set())
            assert a-z==({i} if sign==-1 else set())
            assert len(z)-len(a)==sign and prefix(z)-prefix(a)==sign
            assert row['recipient_prefix_count']==prefix(a) and row['donor_prefix_count']==prefix(b)
            assert row['recipient_total_count']==len(a) and row['donor_total_count']==len(b)
            assert row['actual_local_counterfactual_total_count']==len(z)
            assert row['actual_local_counterfactual_prefix_count']==prefix(z)
            if row['kind']=='simple':
                assert b-a==({i} if sign==1 else set()) and a-b==({i} if sign==-1 else set())
                assert d==c
            elif row['kind']=='count_matched':
                k=row['earlier_center'];assert k==2 and i>=6
                assert b-a==({i} if sign==1 else {k})
                assert a-b==({k} if sign==1 else {i})
                assert prefix(a)==prefix(b) and len(a)==len(b) and d!=c
            else:
                assert row['kind']=='same_label' and (i in a)==(i in b)
                assert prefix(a)==prefix(b) and len(a)==len(b)
                if row['same_label_variant']=='earlier_only':assert r[i-2:]==d[i-2:]
                else:assert row['same_label_variant']=='full_read_prefix'
            assert labels(tuple(n+1-v for v in r),'valleys')==a
            assert labels(tuple(n+1-v for v in d),'valleys')==b
            assert labels(tuple(n+1-v for v in c),'valleys')==z
            assert row['read_after_position']==i+1 and row['token_index']==4+2*i
            counts[row['kind']]+=1
        assert not own&training
        pools[family['split']].update(own)
        if family['split']=='fit':fit_halves[family['fit_partition']].update(own)
    for a in pools:
        for b in pools:
            if a<b:assert not pools[a]&pools[b]
    assert not fit_halves['A']&fit_halves['B']
    return dict(status='passed',families=len(selected),directed_records=sum(counts.values()),
        record_kinds=dict(counts),permutation_endpoint_checks=3*sum(counts.values()),
        complement_valley_identity_checks=3*sum(counts.values()),
        support_complement_groups={k:len(v) for k,v in pools.items()},
        fit_AB_complement_groups={k:len(v) for k,v in fit_halves.items()},
        all_selected_disjoint_from_training=True,all_selected_pools_disjoint=True,
        method='Direct extrema comparisons on all selected raw endpoints, independent of production oracle; own canonical complement keys for leakage.')


def prepare_unit(unit,config):
    target=OUT/unit['unit'];receipt=DIAG/'data'/unit['unit']/'completion.json'
    if receipt.exists():
        old=read(receipt)
        assert old['generation_protocol_sha256']==sha(DIAG/'pair_generation_protocol.json')
        for p,h in old['output_sha256'].items():assert sha(p)==h
        print(f'PAIR REUSE {unit["unit"]}',flush=True);return old
    started=time.perf_counter();target.mkdir(parents=True,exist_ok=True)
    assert sha(unit['data_path'])==unit['data_sha256']
    with np.load(unit['data_path'],allow_pickle=False) as archive:
        data={f'{s}_{field}':archive[f'{s}_{field}'].copy()
              for s in ('train','validation','test') for field in ('lengths','permutations')}
    train_rows=[tuple(map(int,p[:n])) for p,n in zip(data['train_permutations'],data['train_lengths'])]
    candidates=build_candidates(data,unit=unit['unit'],candidate_roots_per_cell=64,lengths=LENGTHS,
                                control_seed=611201,max_control_attempts=64)
    print(f'PAIR CANDIDATES {unit["unit"]}: {len(candidates)} in {time.perf_counter()-started:.1f}s',flush=True)
    assigned,components=partition_components(candidates,train_permutations=train_rows,dev_fraction=.2,
                                            namespace='local-causal-split-v1')
    selected,coverage=select_quotas(assigned,QUOTAS)
    independent=independent_selected_audit(selected,train_rows)
    selected_ids={f['family_id'] for f in selected}
    roster=[]
    for f in assigned:
        kinds=Counter(r['kind'] for r in f['pairs'])
        variants=Counter(r.get('same_label_variant') for r in f['pairs'] if r['kind']=='same_label')
        roster.append(dict(family_id=f['family_id'],unit=f['unit'],root_split=f['root_split'],root_index=f['root_index'],
             n=f['n'],center=f['center'],component_id=f['component_id'],split=f['split'],fit_partition=f['fit_partition'],
             exclusion_reason=f['exclusion_reason'],selected=f['family_id'] in selected_ids,
             simple_directions=kinds['simple'],count_matched_directions=kinds['count_matched'],
             same_label_directions=kinds['same_label'],same_label_earlier_only=variants['earlier_only'],
             same_label_full_read_prefix=variants['full_read_prefix'],
             missing_controls=json.dumps(f['missing_controls'],separators=(',',':'))))
    csv_rows(target/'candidate_roster.csv.gz',roster)
    csv_rows(target/'components.csv.gz',components)
    coverage_path=DIAG/'data'/unit['unit']/'coverage.csv';csv_rows(coverage_path,coverage)
    selected_path=target/'selected_families.json.gz'
    with selected_path.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf-8') as stream:
                json.dump(selected,stream,sort_keys=True,separators=(',',':'),allow_nan=False)
    ab=[]
    for split in ('fit','dev','test'):
        for n in LENGTHS:
            for i in range(2,n-1):
                group=[f for f in selected if f['split']==split and f['n']==n and f['center']==i]
                ab.append(dict(split=split,n=n,center=i,families=len(group),
                    fit_A_families=sum(f['fit_partition']=='A' for f in group),
                    fit_B_families=sum(f['fit_partition']=='B' for f in group),
                    simple_directions=sum(r['kind']=='simple' for f in group for r in f['pairs']),
                    count_matched_directions=sum(r['kind']=='count_matched' for f in group for r in f['pairs']),
                    same_label_directions=sum(r['kind']=='same_label' for f in group for r in f['pairs'])))
    ab_path=DIAG/'data'/unit['unit']/'split_AB_coverage.csv';csv_rows(ab_path,ab)
    audit_path=DIAG/'data'/unit['unit']/'independent_math_audit.json';write(audit_path,independent)
    assert sha(unit['data_path'])==unit['data_sha256']
    assert sha(PAIRS)==EXPECTED_PAIRS_SHA
    result=dict(status='complete',unit=unit['unit'],historical=unit['historical'],
        role='engineering_reference' if unit['historical'] else 'formal_four_unit',
        data_path=unit['data_path'],data_sha256_before=unit['data_sha256'],data_sha256_after=unit['data_sha256'],
        generation_protocol_sha256=sha(DIAG/'pair_generation_protocol.json'),
        candidate_families=len(candidates),selected_families=len(selected),
        selected_by_split=dict(Counter(f['split'] for f in selected)),
        selected_fit_AB=dict(Counter(f['fit_partition'] for f in selected if f['split']=='fit')),
        candidate_exclusions=dict(Counter(f['exclusion_reason'] for f in assigned if f['split']=='excluded')),
        components=len(components),maximum_component_families=max(c['families'] for c in components),
        training_collision_components=sum(c['training_collision'] for c in components),
        cells_with_shortfall=sum(row['shortfall']>0 for row in coverage),
        missing_family_quota=sum(row['shortfall'] for row in coverage),
        selected_path=str(selected_path),output_sha256={str(p):sha(p) for p in [selected_path,target/'candidate_roster.csv.gz',
              target/'components.csv.gz',coverage_path,ab_path,audit_path]},
        no_neural_forward=True,no_checkpoint_load=True,no_model_outcome_access=True,no_training=True,
        elapsed_seconds=time.perf_counter()-started)
    write(receipt,result)
    print(f'PAIR COMPLETE {unit["unit"]}: {len(selected)} families, shortfall {result["missing_family_quota"]}, {result["elapsed_seconds"]:.1f}s',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--units',nargs='*');args=parser.parse_args()
    units=read(SOURCE_AUDIT)['units'];config=freeze(units)
    selected=[u for u in units if args.units is None or u['unit'] in args.units]
    assert selected
    start=DIAG/'pair_generation_start.json'
    if not start.exists():
        write(start,dict(created_utc=datetime.now(timezone.utc).isoformat(),
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            protocol_sha256=sha(DIAG/'pair_generation_protocol.json'),no_model_outcomes=True))
    receipts=[]
    for unit in selected:
        for p,h in config['source_sha256'].items():assert sha(p)==h
        receipts.append(prepare_unit(unit,config))
    write(DIAG/'pair_generation_completion.json',dict(status='complete',units=[r['unit'] for r in receipts],
        protocol_sha256=sha(DIAG/'pair_generation_protocol.json'),
        receipts={r['unit']:sha(DIAG/'data'/r['unit']/'completion.json') for r in receipts},
        no_neural_forward=True,no_model_outcome_access=True,no_training=True))


if __name__=='__main__':main()
