"""Independent table-only audit of all eight relations; never remeasure.

Reads CSV/JSON scientific tables only. Does not load model, data, activations,
or test arrays. Decimal arithmetic checks decompositions and all21 summaries.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import struct

DIAG = Path(__file__).resolve().parents[1]
FIELDS = tuple(f'{p}_{r}' for p in ('initial', 'trained') for r in ('correct', 'wrong', 'identity')) + ('B0','Bt','dc','dw','di','Delta')
ERRORS = tuple(f'error_{p}_{r}' for p in ('initial', 'trained') for r in ('correct', 'wrong', 'identity'))
KEYS = ('architecture','encoding','relation','control','branch','layer','metric')


def require(value, message):
    if not value:
        raise AssertionError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def js(path):
    return json.loads(Path(path).read_text())


def val(value):
    return float(value) if value not in (None,'','None','nan','NaN') else float('nan')


def same(first, second, tolerance=1e-10):
    a, b = val(first), val(second)
    require((math.isnan(a) and math.isnan(b)) or (math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tolerance),
            f'Arithmetic mismatch: {first} vs {second}')


def exact(first, second):
    a, b = val(first), val(second)
    require((math.isnan(a) and math.isnan(b)) or struct.pack('d',a)==struct.pack('d',b), 'Original numeric bytes not reused')


def mean21(records, field):
    values = [val(r.get(field)) for r in records]
    require(len(values)==21, 'Cannot reweight missing lengths')
    if not all(map(math.isfinite, values)):
        return float('nan')
    with localcontext() as c:
        c.prec=60
        return float(sum((Decimal.from_float(v) for v in values), Decimal(0))/Decimal(21))


def audit():
    protocol=js(DIAG/'protocol.json')
    for filename in ('training_freeze.json','analysis_freeze.json'):
        frozen=js(DIAG/filename)
        require(sha(DIAG/'protocol.json')==frozen['protocol_sha256'], 'Protocol freeze mismatch')
        for path,digest in frozen['source_sha256'].items():
            require(sha(path)==digest,'Frozen source mismatch: '+path)
    source=read(DIAG/'per_length_results.csv')
    require(len(source)==24192,'Expected complete24192 cells')
    expected=set()
    for a in protocol['architectures']:
        layers=[f'block{i}' if a=='transformer' else f'layer{i}' for i in range(1,5)]
        branchmetrics=[('hidden','final_norm',('linear_cka','debiased_cka','knn_overlap','procrustes')),
                       ('output_prob','output',('linear_cka','debiased_cka','knn_overlap','procrustes'))]
        branchmetrics += [('hidden', l, ('linear_cka','debiased_cka')) for l in layers+['embedding']]
        for e in protocol['encodings']:
            for relation in protocol['relations']:
                for control in protocol['controls']:
                    for branch,layer,names in branchmetrics:
                        for name in names:
                            for n in range(10,31):
                                expected.add((a,e,relation['pair_id'],control,branch,layer,name,n))
    keys=[tuple(r[k] for k in KEYS)+(int(r['length']),) for r in source]
    require(len(keys)==len(set(keys)) and set(keys)==expected,'Missing/duplicate fixed metric layer crossing')
    arithmetic=0
    identities=0
    constants=0
    groups=defaultdict(list)
    metric_hashes=defaultdict(set)
    bykey=dict(zip(keys,source))
    for row in source:
        key=tuple(row[k] for k in KEYS)
        groups[key].append(row)
        require(row['metric'] in ('linear_cka','debiased_cka') or row['layer'] in ('final_norm','output'), 'Metrics added to new layers')
        if row['layer']=='embedding':
            require(all(math.isnan(val(row[k])) for k in FIELDS),'Constant embedding turned into a numeric finding')
            require('structurally_constant' in row['nan_reason'],'Constant undefined reason missing')
            constants+=1
        else:
            metric_hashes[key[:-1]+(int(row['length']),)].add(row['raw_feature_array_hashes'])
        for dest,first,second in [('B0','initial_correct','initial_wrong'),('Bt','trained_correct','trained_wrong'),
                                  ('dc','trained_correct','initial_correct'),('dw','trained_wrong','initial_wrong'),
                                  ('di','trained_identity','initial_identity'),('Delta','Bt','B0'),('Delta','dc','dw')]:
            same(row[dest],val(row[first])-val(row[second]));arithmetic+=1
        for field,alias in [('dc','change_correct'),('dw','change_wrong'),('di','change_identity')]:
            same(row[field],row[alias]);arithmetic+=1
        for phase in ('initial','trained'):
            for role in ('correct','wrong','identity'):
                field='error_'+phase+'_'+role
                if row['metric']=='procrustes':
                    same(row[field],-val(row[phase+'_'+role]));arithmetic+=1
                else:
                    require(math.isnan(val(row[field])),'Non-Procrustes numerical error invented')
        if row['branch']=='hidden' and row['layer']!='embedding' and math.isfinite(val(row['initial_identity'])):
            same(row['initial_identity'],0 if row['metric']=='procrustes' else 1,
                 tolerance=float(protocol['validation']['identity_decomposition_abs_tolerance']))
            identities+=1
    require(constants==2688,'All constant embedding cells must remain visible')
    require(all(len(v)==1 for v in metric_hashes.values()),'Different metric variants use different raw arrays/selected anchors')
    # With raw within-length centering, task-independent shared initialization
    # and identical anchors must give identical initial geometry for each
    # transformation. Answer strata deliberately do not obey this comparison:
    # relation-specific joint labels change their measurement preprocessing.
    relation_transforms={r['pair_id']:r['input_transform'] for r in protocol['relations']}
    rawinit=defaultdict(list)
    for row in source:
        if row['control']=='raw_within_length' and row['branch']=='hidden':
            key=(row['architecture'],row['encoding'],relation_transforms[row['relation']],row['layer'],row['metric'],int(row['length']))
            rawinit[key].append(row)
    rawinit_checks=0
    for key,group in rawinit.items():
        expected_size=sum(r['input_transform']==key[2] for r in protocol['relations'])
        require(len(group)==expected_size,'Shared-init same-transform raw grouping incomplete')
        for field in ('initial_correct','initial_wrong','initial_identity'):
            for row in group[1:]:
                same(row[field],group[0][field]);rawinit_checks+=1
    olddir=Path(protocol['old_analysis_dir'])
    for filename,digest in protocol['old_table_sha256'].items():
        require(sha(olddir/filename)==digest,'Original source table changed: '+filename)
    old=read(olddir/'per_length_results.csv')
    exact_count=0
    for row in old:
        new=bykey[tuple(row[k] for k in KEYS)+(int(row['length']),)]
        for field in FIELDS+ERRORS:
            exact(row.get(field),new.get(field));exact_count+=1
        for field in ('raw_feature_array_hashes','fit_samples','test_samples','nan_reason'):
            require(new[field]==row[field],'Original proven row metadata changed: '+field)
        require(new['inherited_original_values']=='True','Original scientific scores recomputed')
    require(len(old)==6048,'Original proven complete rows missing')
    summaries=read(DIAG/'contrast_summary.csv')
    summaries_index={tuple(r[k] for k in KEYS):r for r in summaries if r['contrast_type']=='condition_summary'}
    require(len(groups)==len(summaries_index)==1152,'Summary coverage mismatch')
    undefined=0
    for key,g in groups.items():
        require(sorted(int(r['length']) for r in g)==list(range(10,31)),'Summary group lacks one of21 fixed lengths')
        for field in FIELDS+ERRORS:
            expectedvalue=mean21(g,field)
            same(summaries_index[key][field],expectedvalue)
            undefined+=not math.isfinite(expectedvalue)
    contrasts=0
    for row in summaries:
        if row['contrast_type'] in ('condition_summary','metric_direction'):
            continue
        leftmeta,rightmeta=json.loads(row['left_condition']),json.loads(row['right_condition'])
        if row['contrast_type']=='interaction':
            eindex={tuple(r[k] for k in KEYS):r for r in summaries if r['contrast_type']=='encoding_difference'}
            left=eindex[tuple(leftmeta[k] for k in KEYS)]
            right=eindex[tuple(rightmeta[k] for k in KEYS)]
        else:
            left=summaries_index[tuple(leftmeta[k] for k in KEYS)]
            right=summaries_index[tuple(rightmeta[k] for k in KEYS)]
        for field in FIELDS:
            same(row[field],val(right[field])-val(left[field]));contrasts+=1
        if row['contrast_type']=='encoding_difference':
            for name,field in [('Gamma0','B0'),('Gammat','Bt'),('Gammac','dc'),('Gammaw','dw'),('GammaDelta','Delta')]:
                same(row[name],row[field]);contrasts+=1
    cross=read(DIAG/'cross_arch_same_task.csv')
    require(len(cross)==2688,'Cross-task coverage incomplete')
    crosskeys=('task','encoding','control','metric')
    crossgroups=defaultdict(list)
    crossindex={tuple(r[k] for k in crosskeys)+(int(r['length']),):r for r in cross}
    require(len(crossindex)==2688,'Repeated same-task cells')
    for row in cross:
        crossgroups[tuple(row[k] for k in crosskeys)].append(row)
        same(row['similarity_change'],val(row['trained_similarity'])-val(row['initial_similarity']))
    crosssum={tuple(r[k] for k in crosskeys):r for r in read(DIAG/'cross_arch_same_task_summary.csv')}
    require(len(crossgroups)==len(crosssum)==128,'32pairs×2controls×2metrics expected')
    for key,g in crossgroups.items():
        require(sorted(int(r['length']) for r in g)==list(range(10,31)),'Same-task missing length')
        for dest,name in [('initial','initial_similarity'),('trained','trained_similarity'),('change','similarity_change')]:
            same(crosssum[key][dest],mean21(g,name))
    for row in read(olddir/'cross_arch_same_task.csv'):
        new=crossindex[tuple(row[k] for k in crosskeys)+(int(row['length']),)]
        for name in ('initial_similarity','trained_similarity','similarity_change'):
            exact(row[name],new[name]);exact_count+=1
    gold=read(DIAG/'gold_baseline_length.csv')
    require(len(gold)==32256,'Static gold missing cells')
    gkeys=('architecture','encoding','relation','length','control','baseline','role','metric')
    goldindex={tuple(r[k] for k in gkeys):r for r in gold}
    require(len(goldindex)==32256,'Static gold duplicate cells')
    for row in read(olddir/'gold_baseline_length.csv'):
        new=goldindex[tuple(row[k] for k in gkeys)]
        for name in ('value','error'):
            exact(row[name],new[name]);exact_count+=1
    learning=read(DIAG/'source_learning.csv')
    require(len(learning)==16128,'Source learning missing original or negative-learning endpoints')
    lkeys=('architecture','encoding','task','condition','split','action')
    lgroups=defaultdict(list)
    lindex={tuple(r[k] for k in lkeys)+(int(r['length']),):r for r in learning}
    require(len(lindex)==16128,'Source learning duplicate cells')
    for row in learning:
        lgroups[tuple(row[k] for k in lkeys)].append(row)
        same(row['accuracy_minus_length_mode'],val(row['answer_accuracy'])-val(row['length_majority_accuracy']))
    lsummaries={tuple(r[k] for k in lkeys):r for r in read(DIAG/'source_learning_summary.csv')}
    require(len(lgroups)==len(lsummaries)==768,'64models×2conditions×2splits×3actions expected')
    for key,g in lgroups.items():
        require(sorted(int(r['length']) for r in g)==list(range(10,31)),'Source learning incomplete lengths')
        for field in ('answer_accuracy','answer_ce','length_majority_accuracy','accuracy_minus_length_mode'):
            same(lsummaries[key][field],mean21(g,field))
        same(lsummaries[key]['minimum_length_accuracy'],min(val(r['answer_accuracy']) for r in g))
    for row in read(olddir/'source_learning.csv'):
        new=lindex[tuple(row[k] for k in lkeys)+(int(row['length']),)]
        for field in ('answer_accuracy','answer_ce','length_majority_accuracy','accuracy_minus_length_mode'):
            exact(row[field],new[field]);exact_count+=1
    actual=js(DIAG/'metric_review/independent_metric_actual.json')
    require(actual['status']=='passed' and actual['original_fixture_checks']==48,'Fixed original48 independent checks missing')
    actual_original=[r for r in actual['records'] if r['relation']=='descent_inverse']
    expected_actual={(c,p,r,m) for c in protocol['controls'] for p in ('initial','trained')
                     for r in ('correct','wrong','identity')
                     for m in ('linear_cka','debiased_cka','knn_overlap','procrustes')}
    require(len(actual_original)==48 and {(r['control'],r['condition'],r['role'],r['metric']) for r in actual_original}==expected_actual,
            'Fixed48 independent numerical crossing incomplete')
    for row in actual['records']:
        if row.get('passed'):
            require(row['absolute_error'] is None or row['absolute_error']<=1e-8,'Independent actual metric tolerance breached')
        else:
            require(row.get('status')=='unavailable_fixed64','Undeclared skipped fixed fixture')
    key_coordinate_checks=0
    if not protocol.get('fixture_only',False):
        key_coordinates=read(DIAG/'reused_key_coordinate_audit.csv')
        require(len(key_coordinates)==2016,'Original two-relation key-coordinate audit incomplete')
        for row in key_coordinates:
            require(row['original_key_dtype']=='int16' and row['current_key_dtype']=='int64',
                    'Unexpected unregistered answer-key conversion')
            require(all(row[k]=='True' for k in ('values_equal','partitions_equal',
                'initial_left_centered_values_bitwise_equal','original_CSV_key_hash_preserved',
                'no_scientific_metric_recomputed')),'Old key transformation lacks exact grouping/centering evidence')
            key_coordinate_checks+=1
        alignment=read(DIAG/'relation_label_alignment.csv')
        require(len(alignment)==1008,'Relation-answer alignment coverage incomplete')
        for row in alignment:
            require(0<=int(row['mapped_answer_matches'])<=int(row['samples']),'Invalid relation-label counts')
            same(row['mapped_answer_match_rate'],int(row['mapped_answer_matches'])/int(row['samples']))
            if row['role']=='correct':
                same(row['mapped_answer_match_rate'],1,tolerance=0)
    result=dict(status='passed',fixture_only=bool(protocol.get('fixture_only',False)),
        auditor_sha256=sha(__file__),per_length_rows=24192,full21_groups=1152,
        original_scientific_rows_reused=6048,exact_binary64_comparisons=exact_count,
        per_row_arithmetic_checks=arithmetic,contrast_arithmetic_checks=contrasts,
        initial_hidden_identity_checks=identities,constant_embedding_undefined_rows=constants,
        raw_shared_initial_same_transform_cross_relation_checks=rawinit_checks,
        original_key_dtype_mapping_checks=key_coordinate_checks,
        answer_strata_not_assumed_equal_across_relationships=True,
        undefined21_component_aggregates_preserved=undefined,no_NaN_reweight=True,
        cross_arch_same_task_rows=2688,cross_arch_same_task_pairs=32,gold_rows=32256,source_learning_rows=16128,
        primary_real_independent_fixture_checks=48,actual_independent_metric_max_error=actual['maximum_absolute_error'],
        different_metrics_same_raw_arrays_and_anchor_keys=True,
        no_geometry_metrics_recalculated=True,no_raw_data_or_test_arrays_loaded=True,no_training=True,
        one_world=True,relation_instances=8,transform_types=2,
        main_final_CKA_answer_control=[r for r in summaries if r['contrast_type']=='condition_summary'
            and r['layer']=='final_norm' and r['branch']=='hidden' and r['control']=='answer_strata' and r['metric']=='linear_cka'],
        input_sha256={name:sha(DIAG/name) for name in ['per_length_results.csv','contrast_summary.csv',
            'cross_arch_same_task.csv','cross_arch_same_task_summary.csv','gold_baseline_length.csv',
            'source_learning.csv','source_learning_summary.csv']})
    (DIAG/'metric_review/final_reader_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (DIAG/'metric_review/final_reader_audit.txt').write_text(
        'Independent read-only table review of eight relations passed.24192 per-length rows,1152 complete21-length groups,and6048 earlier scientific results reused value for value.\n'
        'Original S,B0,Bt,correct/wrong changes,and Delta/Gamma/configuration/layer decompositions all verified;undefined components are not reweighted.\n'
        'Checks passed for32 same-task cross-configuration pairings,static answers,all64 source-learning records,and original-value reuse.\n'
        'This review computes no metrics,loads no raw input/test arrays,and runs no models/training;one world,eight instances,and only two transformation types.\n'
        'Interpret Delta versus Bt reversals,and correct-score rises versus wrong-score falls,separately;unmatched source mastery/capacity precludes pure mechanism inference.\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('main_final_CKA_answer_control','input_sha256')}))


if __name__=='__main__':
    audit()
