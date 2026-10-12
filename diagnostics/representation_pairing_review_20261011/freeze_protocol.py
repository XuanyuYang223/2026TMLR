"""Freeze exploratory task-pair controls before their actual CKA is computed."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[1]
PERM=WORKSPACE/'diagnostics/permworld_eight_relations_20261010'
GRAPH=WORKSPACE/'diagnostics/graphworld_pilot_20261010'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def run():
    assert not (HERE/'freeze.json').exists(), 'Never refresh a frozen exploration protocol.'
    assert read(HERE/'protection_start.json')['status']=='passed'
    pp=read(PERM/'protocol.json');gp=read(GRAPH/'protocol.json')
    with (HERE/'classification_manifest.csv').open() as f:math_pairs=list(csv.DictReader(f))
    assert len(math_pairs)==132 and sum(r['domain']=='permworld' for r in math_pairs)==120
    categories={'designated','other_known_exact','not_designated'}
    assert all(r['category'] in categories for r in math_pairs)
    pairs=[]
    for row in math_pairs:
        if row['domain']=='permworld':
            rr=next(r for r in pp['relations'] if r['pair_id']==row['relation'])
            conditions=[(a,e) for a in pp['architectures'] for e in pp['encodings']]
            label_map='subtract original relation offset '+str(rr['right_label_offset'])+' from each candidate right answer; a bijective coordinate map only'
            designated=rr['right']
        else:
            assert row['domain']=='graphworld'
            rr=next(r for r in gp['relations'] if r['relation']==row['relation'])
            conditions=[(gp['reference_architecture'],gp['encodings'][0])]
            label_map=rr['mapping'];designated=rr['right']
        for architecture,encoding in conditions:
            pair=dict(row);pair.update(architecture=architecture,encoding=encoding,
                designated_right_task=designated,original_right_label_mapping=label_map,
                pair_id='__'.join((row['domain'],row['relation'],architecture,encoding,row['right_task'])))
            assert (pair['category']=='designated')==(pair['right_task']==designated)
            pairs.append(pair)
    assert len(pairs)==492 and len({r['pair_id'] for r in pairs})==492
    assert sum(r['domain']=='permworld' for r in pairs)==480
    with (HERE/'candidate_pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in pairs for k in r)));w.writeheader();w.writerows(pairs)
    systems={}
    for name,old,protocol,index,data,results,split,lengths,controls in (
        ('permworld',PERM,PERM/'protocol.json',Path(pp['evaluation_index_path']),Path(pp['source_data_path']),PERM/'per_length_results.csv','test',pp['lengths'],['raw_within_length','answer_strata']),
        ('graphworld',GRAPH,GRAPH/'protocol.json',Path('/mnt/yangx-projects-f/20261010/graphworld_pilot/evaluation_cache/index.json'),Path('/mnt/yangx-projects-f/20261010/graphworld_pilot/world/data.npz'),GRAPH/'geometry_per_n.csv','analysis',gp['lengths'],['raw_within_n','answer_strata','answer_density_strata'])):
        systems[name]=dict(protocol_path=str(protocol),protocol_sha256=sha(protocol),
            index_path=str(index),index_sha256=sha(index),data_path=str(data),data_sha256=sha(data),
            original_results_path=str(results),original_results_sha256=sha(results),
            analysis_split=split,lengths=lengths,weights=[1/len(lengths)]*len(lengths),
            controls=controls,source_models=64 if name=='permworld' else 4,
            conditions=32 if name=='permworld' else 4,expected_candidate_pairs=480 if name=='permworld' else 12)
    metric=Path(pp['metrics_source_path'])
    protocol=dict(version='representation-pairing-review/2026-10-11-v1',
        registered_utc=datetime.now(timezone.utc).isoformat(),exploratory=True,
        prior_training_and_geometry_results_seen=True,confirmation_preregistration=False,
        new_actual_cross_task_metrics_computed_before_freeze=False,
        purpose='Distinguish the old fixed-task/vary-input-role comparison from fixed-input-transform/vary-right-task selectivity.',
        systems=systems,pair_manifest_path=str(HERE/'candidate_pairs.csv'),pair_manifest_sha256=sha(HERE/'candidate_pairs.csv'),
        classification_manifest_path=str(HERE/'classification_manifest.csv'),classification_manifest_sha256=sha(HERE/'classification_manifest.csv'),
        classification_audit_path=str(HERE/'classification_audit.json'),classification_audit_sha256=sha(HERE/'classification_audit.json'),
        categories=['designated','other_known_exact','not_designated'],
        classification_limit='A frozen catalogue of known exact directed task correspondences, not a proof that other tasks have no mathematical relation. Finite counterexamples reject only a tested specific formula.',
        metric_source_path=str(metric),metric_source_sha256=sha(metric),cpu_threads=2,existing_tolerance=1e-8,
        metric='ordinary linear CKA using frozen metrics.center and metrics.cka',
        endpoint='Saved prefix_final, final norm at input-end before task token. Not packed hook final norm.',
        feature_dtype='Preserve recorded float32/float64 cache; original centering/CKA converts to float64.',
        conditions=['initial','trained'],new_score_fields=['fixed_transform_score_initial','fixed_transform_score_trained'],
        inputs='All original analysis anchors and their fixed correct transforms; all candidate specialists excluding left itself. No new inputs/world/selection by result.',
        controls=dict(permworld='Old joint[n,f(identity),g(identity)-offset,g(complement)-offset,g(inverse)-offset], minimum count3 per length, center each retained stratum independently. Right offset is a bijection for every candidate; no assumption of equal numeric answers.',
                      graphworld='Old joint[n,f(identity),mapped g(identity),mapped g(T),mapped g(WT)], minimum count3; original complement map n-1-y is bijective. Density sensitivity adds exact anchor edge count m; no threshold relaxation.',
                      interpretation='Raw anchor pool shared across all candidates. Answer masks/groups may differ by right task; record counts, residual df, retained IDs/key hashes and undefined reasons. Do not pretend equal support or a pure mathematical-relation causal intervention.'),
        aggregation='Compute each fixed length independently then equal1/21 (Perm) or1/3 (Graph). Any undefined length makes that candidate/phase aggregate undefined; never fill0 or reweight.',
        category_summaries='Per f/T/config/encoding/control/phase/category: finite-value mean and median plus total,finite,undefined counts; include empty categories explicitly.',
        category_difference_formula='D_category,phase = S_designated,phase - mean(S_candidates_in_category,phase with finite complete-length aggregates). Phase-specific supports listed.',
        relative_initial_comparison_formula='G_category = [S_designated,trained - mean_C(S_g,trained)] - [S_designated,initial - mean_C(S_g,initial)], C is the prespecified category candidates with complete finite aggregates in both phases; designated must be finite in both. This is a new contrast, not original Bt or Delta.',
        fixed_check_plan=dict(perm_relation='descent_inverse',perm_right_tasks=['recoils','anti_fixed_points'],perm_conditions='all four configuration/encoding conditions',perm_length=10,
                             graph_relation='reverse_max',graph_right_tasks=['max_in','min_in'],graph_length=6,
                             controls=['raw','answer_strata'],phases=['initial','trained'],checks=40,
                             path='Saved activations only, full retained anchor cells; independent centered cross-covariance vs frozen Gram formula. No new forward.'),
        validation=dict(metric_abs_tolerance=1e-8,arithmetic_abs_tolerance=1e-10,
                        original_designated_comparison='All32 Perm and4 Graph designated pairs, all original controls/lengths/conditions, against saved initial_correct/trained_correct; no tolerance relaxation.',
                        label_checks='Saved exact labels, original transforms/maps, finite catalog validation and actual W accidental formula diagnostics.',
                        pooling_or_layer_search=False,additional_metrics=False),
        W_diagnostic='Actual original analysis anchors, original wrong role (WR/WC or opposite Perm transform), exact mapped-answer equality, preserve every boolean. Full raw and original answer/density retained counts separate, fixed-length equal-rate and pooled count reported separately, unique-anchor count separate from row count. Random mathematical audit is a different input population.',
        dependence='Shared specialists, inputs, worlds, saved initialization and transformations;492 pairs are not492 independent replicates; no p-values, causal/functional or universal mechanism claims.',
        proposals_only='2–4 bounded future comparisons from already seen exploratory results, paired within new true initialization/data-world units; no new data generation, models, probes or future execution.',
        protection_start_path=str(HERE/'protection_start.json'),protection_start_sha256=sha(HERE/'protection_start.json'),
        limits=dict(new_training=0,optimizers=0,new_models=0,new_worlds=0,new_forward=0,metric_search=0,new_all_layer_metrics=0,
                    old_artifact_writes=0,Git_commit=0,upload=0,delete=0,move=0),
        environment=dict(python=sys.version,packages={n:importlib.metadata.version(n) for n in ('numpy','matplotlib','torch','threadpoolctl')}),
        stop='After the fixed review, all492 task-pair controls, actual W diagnostic, report/figures/independent checks and start/end protection. No automatic next experiment.')
    write(HERE/'protocol.json',protocol)
    scripts=['protection_audit.py','freeze_protocol.py','classification_and_W.py','existing_results_review.py','fixed_transform_analysis.py','engineering_fixture.py']
    for n in scripts:assert (HERE/n).is_file(),n
    freeze=dict(status='frozen_before_new_actual_task_pair_CKA',registered_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=sha(HERE/'protocol.json'),source_sha256={str(HERE/n):sha(HERE/n) for n in scripts},
        candidate_pairs_sha256=sha(HERE/'candidate_pairs.csv'),classification_manifest_sha256=sha(HERE/'classification_manifest.csv'),
        classification_audit_sha256=sha(HERE/'classification_audit.json'),
        protection_start_sha256=sha(HERE/'protection_start.json'),new_actual_task_pair_CKA_seen=False,confirmation_preregistration=False)
    write(HERE/'freeze.json',freeze)
    print(json.dumps(dict(status=freeze['status'],candidate_pairs=492,protocol_sha256=freeze['protocol_sha256'])))

if __name__=='__main__':run()
