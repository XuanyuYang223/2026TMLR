"""Fixed saved-vector/CSV arithmetic audit; never run or differentiate a model."""
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TIMING=ROOT/'diagnostics/permworld_local_timing_20261110'
OLD=Path('/mnt/yangx-projects-f/20261110/permworld_local_causal')
OUT=Path('/mnt/yangx-projects-f/20261110/permworld_closing_audit/gradient')
UNITS=('archdev001','unit02','unit03','unit04','unit05')
SEEDS=tuple(range(1026,1034))
EPS=1e-12


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def rows(path):
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'rt',newline='') as handle:return list(csv.DictReader(handle))


def num(value):
    if value is None or str(value).lower() in ('','nan','none','null'):return None
    result=float(value)
    return result if math.isfinite(result) else None


def boolean(value):
    if isinstance(value,bool):return value
    assert str(value) in ('True','False'),value
    return str(value)=='True'


class Checks:
    def __init__(self):self.n=0;self.maxima=defaultdict(float)
    def ok(self,value,label):
        self.n+=1
        assert bool(value),label
    def same(self,a,b,label,category='scalar'):
        a,b=num(a),num(b);self.n+=1
        if a is None or b is None:
            assert a is None and b is None,(label,a,b)
            return
        error=abs(a-b);self.maxima[category]=max(self.maxima[category],error)
        assert error<=1e-10+1e-10*abs(b),(label,a,b,error)
    def array(self,a,b,label,exact=False):
        self.n+=1;a,b=np.asarray(a),np.asarray(b)
        assert a.shape==b.shape,(label,a.shape,b.shape)
        if exact:assert np.array_equal(a,b),label
        else:
            delta=np.max(np.abs(a.astype(float)-b.astype(float)))
            self.maxima['vector']=max(self.maxima['vector'],float(delta))
            assert np.allclose(a,b,atol=1e-10,rtol=1e-10,equal_nan=True),(label,delta)


def open_arrays(path,expected,bindings,checks):
    path=Path(path);actual=sha(path);checks.ok(actual==expected,'source SHA '+str(path))
    bindings[str(path)]=actual
    with np.load(path,allow_pickle=False) as source:return {k:source[k].copy() for k in source.files}


def norm(value):return math.sqrt(math.fsum(float(x)*float(x) for x in value))
def dot(a,b):return math.fsum(float(x)*float(y) for x,y in zip(a,b))


def angles(g,delta):
    gn,dn=norm(g),norm(delta);product=dot(g,delta)
    valid=gn>EPS and dn>EPS
    cosine=max(-1.,min(1.,product/gn/dn)) if valid else None
    return gn,dn,product,cosine,math.degrees(math.acos(cosine)) if valid else None,valid


def main():
    spec=load(HERE/'verification_spec.json');checks=Checks();bindings={}
    for path,expected in spec['source_bindings'].items():
        checks.ok(sha(path)==expected,'preselected original input unchanged');bindings[path]=expected
    for path,expected in load(HERE/'run_freeze.json')['sources'].items():
        checks.ok(sha(path)==expected,'frozen production source unchanged');bindings[path]=expected
    cases=spec['cases'];checks.ok(len(cases)==120,'fixed case inventory')
    selected={(r['unit'],r['task'],r['pair_id']):r for r in cases}
    old_audit=load(TIMING/'input_audit.json')
    old_units={u['unit']:u for u in old_audit['units']}
    all_case_rows={};all_random_rows={};fd_rows=[];raw_counts={};receipts={};original_records={}
    for unit in UNITS:
        with gzip.open(OLD/unit/'selected_families.json.gz','rt') as handle:
            families=json.load(handle)
        original_records[unit]={r['pair_id']:r for family in families if family['split']=='test'
                                for r in family['pairs'] if r['kind'] in ['simple','count_matched']}
        receipt=load(HERE/'results'/unit/'completion.json');receipts[unit]=receipt
        checks.ok(receipt['status']=='completed' and receipt['pairs']==20160,unit+' completed pair count')
        bindings[str(HERE/'results'/unit/'completion.json')]=sha(HERE/'results'/unit/'completion.json')
        for task in receipt['task_reports']:
            checks.ok(task['parameter_sha256_before']==task['parameter_sha256_after'],unit+'/'+task['task']+' parameters unchanged')
        count=0
        path=HERE/'results'/unit/'gradient_per_pair.csv.gz'
        with gzip.open(path,'rt',newline='') as handle:
            for row in csv.DictReader(handle):
                count+=1;key=(unit,row['task'],row['pair_id'])
                if key in selected:
                    checks.ok(key not in all_case_rows,'unique fixed row');all_case_rows[key]=row
        checks.ok(count==20160,unit+' raw CSV rows');raw_counts[unit]=count
        path=HERE/'results'/unit/'random_directions.csv.gz';random_count=0
        with gzip.open(path,'rt',newline='') as handle:
            for row in csv.DictReader(handle):
                random_count+=1;key=(unit,row['task'],row['pair_id'])
                if key in selected:
                    full=(*key,int(row['random_seed']))
                    checks.ok(full not in all_random_rows,'unique fixed random row');all_random_rows[full]=row
        checks.ok(random_count==20160*8,unit+' random CSV rows')
        fd_rows.extend(rows(HERE/'results'/unit/'finite_difference.csv'))
    checks.ok(len(all_case_rows)==120 and len(all_random_rows)==960,'selected CSV coverage')
    checked_rows=[]
    grouped=defaultdict(list)
    for case in cases:grouped[case['unit'],case['task'],case['length']].append(case)
    for (unit,task,n),cell_cases in grouped.items():
        old_desc=next(c for c in old_units[unit]['cells'][task] if (c['n'],c['i'])==(n,6))
        timing_receipt=load(TIMING/'results'/unit/'completion.json')
        timing_desc=next(c for t in timing_receipt['task_reports'] if t['task']==task for c in t['cells'] if (c['n'],c['i'])==(n,6))
        old=open_arrays(old_desc['path'],old_desc['sha256'],bindings,checks)
        timing=open_arrays(timing_desc['path'],timing_desc['sha256'],bindings,checks)
        path=OUT/unit/'cells'/task/f'n{n}_i6.npz'
        saved=open_arrays(path,receipts[unit]['files'][str(path)],bindings,checks)
        probe_path=OLD/unit/'probes'/task/'layer1_point1.npz'
        probe=open_arrays(probe_path,old_units[unit]['probes'][task]['sha256'],bindings,checks)
        v=probe['edit_ridge_coef']/probe['edit_scaler_scale'];v=v/norm(v)
        checks.array(v,probe['direction'],'raw scaler conversion')
        checks.array(v,saved['raw_unit_axis'],'saved raw axis')
        checks.same(norm(v),1,'raw axis unit norm')
        checks.array(probe['edit_ridge_coef']/probe['edit_scaler_scale'],probe['edit_weight'],'raw weight equals converted coefficient')
        checks.ok(str(saved['spec_sha256'].item())==sha(HERE/'spec.json'),'spec binding')
        checks.ok(str(saved['source_timing_cell_sha256'].item())==timing_desc['sha256'],'timing binding')
        checks.ok(str(saved['source_old_cell_sha256'].item())==old_desc['sha256'],'old cell binding')
        source_model=next(u for u in load(HERE/'spec.json')['units'] if u['unit']==unit)
        checks.ok(str(saved['checkpoint_sha256'].item())==source_model['models'][task]['checkpoint_sha256'],'checkpoint binding')
        checks.ok(str(saved['data_sha256'].item())==source_model['data_sha256'],'data binding')
        checks.array(saved['pair_ids'],old['pair_ids'],'pair ID order',exact=True)
        injected=list(map(str,timing['injected_conditions']))
        times=list(map(str,timing['timings']));conditions=list(map(str,timing['conditions']))
        baseline=next(k for k,pair in enumerate(zip(times,conditions)) if pair==('early','unedited'))
        edited=next(k for k,pair in enumerate(zip(times,conditions)) if pair==('early','local_projection'))
        for case in cell_cases:
            j=case['row'];key=(unit,task,case['pair_id']);row=all_case_rows[key]
            checks.ok(str(saved['pair_ids'][j])==case['pair_id'],'fixed row identity')
            checks.ok((int(row['length']),int(row['position']),row['group'],int(row['sign']))==(n,6,case['group'],case['sign']),'fixed metadata')
            h=old['recipient_states'][j,0].astype(float)
            d=old['donor_states'][j,0].astype(float)-h
            actual=timing['injected_layer1_states'][injected.index('local_projection'),j]
            delta=actual.astype(float)-h
            ideal=dot(d,v)*v
            checks.array(d,saved['donor_difference'][j],'raw donor difference',exact=True)
            checks.array(delta,saved['actual_displacement'][j],'actual FP32 delta',exact=True)
            checks.array(ideal,saved['ideal_orthogonal_projection'][j],'ideal projection')
            checks.array((h+ideal).astype(np.float32),actual,'actual rounded displacement',exact=True)
            g=saved['raw_gradient'][j];gn,dn,product,cosine,angle,defined=angles(g,delta)
            checks.ok(np.isfinite(g).all(),'finite saved raw gradient')
            dn_donor=norm(d);energy=dot(d,v)**2/dn_donor**2 if dn_donor>EPS else None
            for field,value in [('donor_difference_norm',dn_donor),('actual_displacement_norm',dn),
                                ('ideal_projection_norm',abs(dot(d,v))),('projection_energy_fraction',energy),
                                ('actual_vs_ideal_norm',norm(delta-ideal)),('gradient_norm',gn),('g_dot_delta',product),
                                ('cosine',cosine),('angle_degrees',angle)]:checks.same(row[field],value,field)
            checks.ok(boolean(row['angle_defined'])==defined,'near-zero angle flag')
            checks.ok(boolean(row['gradient_near_zero'])==(gn<=EPS),'near-zero gradient flag')
            checks.ok(boolean(row['gradient_exact_zero'])==(gn==0),'exact-zero gradient flag')
            c=int(saved['true_recipient_count'][j]);s=int(saved['local_sign'][j])
            checks.ok(c==int(row['true_recipient_count']) and c+s==int(row['desired_local_count']) and s==case['sign'],'recipient c+s endpoint')
            record=original_records[unit][case['pair_id']]
            checks.ok(c==int(record['recipient_total_count']) and s==int(record['direction']), 'original oracle c and direction')
            checks.ok(int(row['donor_final_count'])==int(record['donor_total_count']),'original donor count')
            if case['group']=='matched':
                checks.ok(int(record['donor_total_count'])==c and c+s!=int(record['donor_total_count']), 'matched local goal differs from donor final count')
            original=float(timing['logits'][baseline,j,c+s])-float(timing['logits'][baseline,j,c])
            changed=float(timing['logits'][edited,j,c+s])-float(timing['logits'][edited,j,c])
            for field,value in [('baseline_logit_margin',original),('edited_logit_margin',changed),('actual_logit_margin_change',changed-original),
                                ('linear_prediction_error',changed-original-product)]:checks.same(row[field],value,field)
            checks.same(saved['cached_baseline_margin'][j],original,'saved baseline margin')
            checks.same(saved['cached_edited_margin'][j],changed,'saved edited margin')
            checks.ok(int(row['baseline_argmax_token'])==int(timing['logits'][baseline,j].argmax()),'original mode')
            checks.ok(int(row['edited_argmax_token'])==int(timing['logits'][edited,j].argmax()),'edited mode')
            random_products=[];equal_products=[];random_changes=[]
            for k,seed in enumerate(SEEDS):
                random_row=all_random_rows[*key,seed]
                rd=timing['injected_layer1_states'][injected.index(f'random_direction_{seed}'),j].astype(float)-h
                checks.array(rd,saved['random_displacements'][k,j],'actual saved random displacement',exact=True)
                _,rn,rp,rc,ra,rv=angles(g,rd)
                equal=rp*dn/rn if rn>EPS else None
                ri=next(k for k,pair in enumerate(zip(times,conditions)) if pair==('early',f'random_direction_{seed}'))
                rm=float(timing['logits'][ri,j,c+s])-float(timing['logits'][ri,j,c])
                for field,value in [('displacement_norm',rn),('matched_local_norm',dn),('norm_error',abs(rn-dn)),
                                    ('g_dot_delta',rp),('equal_local_norm_g_dot_direction',equal),('cosine',rc),('angle_degrees',ra),
                                    ('actual_logit_margin_change',rm-original)]:checks.same(random_row[field],value,'random '+field)
                checks.ok(abs(rn-dn)<=1e-5+1e-5*dn,'random original norm tolerance')
                checks.ok(boolean(random_row['angle_defined'])==rv,'random undefined flag')
                random_products.append(rp);equal_products.append(equal);random_changes.append(rm-original)
            for field,value in [('random_mean_g_dot_delta',math.fsum(random_products)/8),
                                ('random_equal_norm_mean_g_dot_delta',math.fsum(equal_products)/8),
                                ('random_mean_actual_margin_change',math.fsum(random_changes)/8),
                                ('local_minus_random_equal_norm_g_dot_delta',product-math.fsum(equal_products)/8)]:checks.same(row[field],value,field)
            checked_rows.append(dict(unit=unit,task=task,length=n,position=6,pair_id=case['pair_id'],group=case['group'],sign=s,
                                     raw_coordinate_count=len(g),angle_defined=defined,source_gradient_cell=str(path)))
    expected_fd={(u,t,p,d) for (u,t,p) in selected for d in ['local','random_1026']}
    actual_fd={(r['unit'],r['task'],r['pair_id'],r['direction']) for r in fd_rows}
    checks.ok(len(fd_rows)==240 and actual_fd==expected_fd,'all240fixedFDrows coverage')
    for row in fd_rows:
        analytic=float(row['analytic']);numerical=float(row['finite_difference'])
        error=abs(numerical-analytic);allowed=1e-7+1e-3*abs(analytic)
        checks.same(row['absolute_error'],error,'saved FD arithmetic')
        checks.same(row['allowed_error'],allowed,'saved fixed FD tolerance')
        checks.ok(boolean(row['passed'])==(error<=allowed),'saved FD classification')
        checks.ok(error<=allowed,'fixed FD passed')
    weighting_checks=0
    for unit in UNITS:
        length_rows=rows(HERE/'results'/unit/'by_length.csv');unit_rows=rows(HERE/'results'/unit/'by_unit.csv')
        grouped_length=defaultdict(list)
        for row in length_rows:
            n=int(row['length']);expected=n-(3 if row['group']=='main' else 7)
            checks.ok(int(row['registered_centers'])==expected,'registered center coverage')
            checks.ok(int(row['finite_samples'])+int(row['undefined_samples'])==int(row['samples']),'finite/undefined length denominator')
            grouped_length[row['task'],row['group'],row['sign'],row['metric']].append(row)
        for row in unit_rows:
            source=grouped_length[row['task'],row['group'],row['sign'],row['metric']]
            checks.ok(len(source)==21 and {int(r['length']) for r in source}==set(range(10,31)),'fixed21length inventory')
            values=[num(r['mean']) for r in source]
            expected=math.fsum(values)/21 if all(v is not None for v in values) else None
            checks.same(row['mean'],expected,'fixed1/21 unit aggregate')
            checks.ok(row['fixed_length_weight']=='1/21','unchanged length weighting')
            checks.ok(int(row['finite_lengths'])==sum(v is not None for v in values),'unit finite denominator')
            weighting_checks+=1
    for path,expected in bindings.items():checks.ok(sha(path)==expected,'end selected source unchanged')
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks.n,
                maximum_differences=dict(checks.maxima),coordinate='raw unnormalized256-dimensional recurrent L1 Euclidean space',
                fixed_cases=120,fixed_random_rows=960,finite_difference_rows_checked=240,unit_weighting_checks=weighting_checks,
                saved_pair_rows_streamed=sum(raw_counts.values()),selected_cases=checked_rows,
                finite_difference_scope='Saved240FP64-shadow analytic/numerical scalar rows checked for fixed subset, arithmetic/tolerance/pass/coverage only; no model or derivative was rerun here.',
                limitations=['Input-only FP32 derivatives and previously cachedCuDNN actual margin changes are different numerical executions. Their1e-4logit agreement does not validate tiny delta_m as curvature/nonlinearity or a mechanistic residual.',
                             'The fixed FP64 FD shadow verifies differentiation of the graph in its own precision, not the precision of every primaryFP32gradient.',
                             'Saved-vector arithmetic validation does not prove that an axis carries an abstract predicate or is naturally used.'],
                no_forward=True,no_autograd=True,no_model_loading=True,no_fits=True,no_new_metrics_or_conditions=True,
                verification_spec_sha256=sha(HERE/'verification_spec.json'),source_sha256=sha(__file__),source_bindings=bindings)
    (HERE/'independent_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['selected_cases','source_bindings']},indent=2))


if __name__=='__main__':main()
