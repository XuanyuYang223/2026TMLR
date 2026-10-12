"""Synthetic complete-table/reader smoke test; no real caches or new outcomes.

Creates all eight-relation CSV shapes in a temporary directory, exercises
fixed21 contrasts, independent table audit, and optionally the full HTML/plot
report. No model, raw data, checkpoint, or real representation array is loaded.
"""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import tomllib

WORKSPACE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(WORKSPACE))
from experiments.permworld_eight_relations_20261010 import analysis_helpers as h
from experiments.permworld_eight_relations_20261010 import report


def main(make_report=False):
    original_diag=h.DIAG
    with tempfile.TemporaryDirectory(prefix='permworld-eight-relations-fixture-') as directory:
        d=Path(directory);old=d/'old';old.mkdir();(d/'metric_review').mkdir()
        relation=tomllib.loads((WORKSPACE/'external/neurips/configs/property_task_geometry.toml').read_text())['relation_pairs']
        tasks=sorted({r[k] for r in relation for k in ('left','right')})
        p=dict(fixture_only=True,lengths=list(range(10,31)),relations=relation,label_task_names=tasks,
            architectures=['transformer','gru'],encodings=['one_line','canonical_cycle'],
            controls=['raw_within_length','answer_strata'],actions=['identity','complement','inverse'],
            validation={'identity_decomposition_abs_tolerance':1e-10},old_analysis_dir=str(old))
        data=[]
        for a in p['architectures']:
            layers=[f'block{i}' if a=='transformer' else f'layer{i}' for i in range(1,5)]
            branches=[('hidden','final_norm',h.ALL_METRICS),('output_prob','output',h.ALL_METRICS)]
            branches += [('hidden',l,h.LAYER_METRICS) for l in layers+['embedding']]
            for e in p['encodings']:
                for ri,r in enumerate(relation):
                    for c in p['controls']:
                        for b,l,names in branches:
                            for metric in names:
                                for n in p['lengths']:
                                    const=l=='embedding'
                                    # Raw shared-init inputs do not depend on
                                    # which task pair names the transformation.
                                    wrong=.2+(ri*.003 if c=='answer_strata' else 0)+n*.0001
                                    correct=wrong+.08+(0.01 if e=='canonical_cycle' else 0)
                                    values={phase:{role:(float('nan') if const else
                                        (0 if metric=='procrustes' else 1) if phase=='initial' and role=='identity' and b=='hidden' else
                                        (correct if role=='correct' else wrong if role=='wrong' else .35)
                                        +(0 if phase=='initial' else .013 if role=='correct' else -.019))
                                        for role in h.ROLES} for phase in h.PHASES}
                                    if metric=='procrustes' and not const:
                                        for phase in h.PHASES:
                                            for role in h.ROLES:
                                                if not (phase=='initial' and role=='identity' and b=='hidden'):
                                                    values[phase][role]-=1.4
                                    err={f'error_{phase}_{role}':-values[phase][role] if metric=='procrustes' else float('nan')
                                         for phase in h.PHASES for role in h.ROLES}
                                    data.append(dict(architecture=a,encoding=e,relation=r['pair_id'],control=c,branch=b,layer=l,
                                        metric=metric,length=n,fit_samples=256,test_samples=512,
                                        **h.component_values(values),**err,
                                        nan_reason='structurally_constant_fixture' if const else '{}',
                                        raw_feature_array_hashes='not_extracted_structural_constant' if const else 'synthetic_same_raw_arrays',
                                        inherited_original_values=r['pair_id'] in h.OLD_RELATIONS))
        h.require(len(data)==24192,'Synthetic fixed coverage wrong')
        summary,index=h.summarize(data,p['lengths'])
        h.require(len(index)==1152,'Synthetic summary coverage wrong')
        olddata=[r for r in data if r['relation'] in h.OLD_RELATIONS]
        h.csv_write(d/'per_length_results.csv',data);h.csv_write(d/'contrast_summary.csv',summary)
        h.csv_write(old/'per_length_results.csv',olddata)
        cross=[]
        for t in tasks:
            for e in p['encodings']:
                for c in p['controls']:
                    for metric in h.LAYER_METRICS:
                        for n in p['lengths']:
                            first=.15+n*.001;last=first+.1
                            cross.append(dict(task=t,encoding=e,control=c,metric=metric,length=n,
                                initial_similarity=first,trained_similarity=last,similarity_change=last-first))
        h.csv_write(d/'cross_arch_same_task.csv',cross)
        h.csv_write(d/'cross_arch_same_task_summary.csv',h.cross_summary(cross,p['lengths']))
        oldtasks={r[k] for r in relation[:2] for k in ('left','right')}
        h.csv_write(old/'cross_arch_same_task.csv',[r for r in cross if r['task'] in oldtasks])
        gold=[]
        for r in relation:
            for a in p['architectures']:
                for e in p['encodings']:
                    for c in p['controls']:
                        for baseline in ('true_answer_onehot','true_answer_scalar'):
                            for role in h.ROLES:
                                for metric in h.ALL_METRICS:
                                    for n in p['lengths']:
                                        value=float('nan') if c=='answer_strata' else .1
                                        if metric=='procrustes':value=-value
                                        gold.append(dict(architecture=a,encoding=e,relation=r['pair_id'],control=c,baseline=baseline,
                                            role=role,metric=metric,length=n,value=value,error=-value if metric=='procrustes' else float('nan')))
        h.csv_write(d/'gold_baseline_length.csv',gold)
        h.csv_write(old/'gold_baseline_length.csv',[r for r in gold if r['relation'] in h.OLD_RELATIONS])
        learning=[];learning_groups={}
        for a in p['architectures']:
            for e in p['encodings']:
                for t in tasks:
                    for phase in h.PHASES:
                        for split in ('validation','test'):
                            for action in p['actions']:
                                group=[]
                                for n in p['lengths']:
                                    acc=.81-n*.001 if phase=='trained' else .01
                                    ce=1.1 if phase=='trained' else float('nan')
                                    row=dict(architecture=a,encoding=e,task=t,condition=phase,split=split,action=action,length=n,
                                        answer_accuracy=acc,answer_ce=ce,length_majority_accuracy=.2,accuracy_minus_length_mode=acc-.2)
                                    group.append(row);learning.append(row)
                                learning_groups[a,e,t,phase,split,action]=group
        learning_summary=[]
        for key,group in learning_groups.items():
            row=dict(zip(('architecture','encoding','task','condition','split','action'),key))
            row.update({field:h.equal_length([r[field] for r in group]) for field in
                ('answer_accuracy','answer_ce','length_majority_accuracy','accuracy_minus_length_mode')})
            row['minimum_length_accuracy']=min(r['answer_accuracy'] for r in group)
            learning_summary.append(row)
        h.csv_write(d/'source_learning.csv',learning);h.csv_write(d/'source_learning_summary.csv',learning_summary)
        h.csv_write(old/'source_learning.csv',[r for r in learning if r['task'] in oldtasks])
        h.csv_write(old/'source_validation_eos_original.csv',[{'run_id':'synthetic_unused_placeholder'}])
        p['old_table_sha256']={f:h.sha(old/f) for f in ['per_length_results.csv','cross_arch_same_task.csv',
            'gold_baseline_length.csv','source_learning.csv','source_validation_eos_original.csv']}
        h.write_json(d/'protocol.json',p)
        freeze=dict(protocol_sha256=h.sha(d/'protocol.json'),source_sha256={})
        h.write_json(d/'training_freeze.json',freeze);h.write_json(d/'analysis_freeze.json',freeze)
        checks=[dict(relation='descent_inverse',control=c,condition=phase,role=role,metric=metric,
                     absolute_error=0,passed=True) for c in p['controls'] for phase in h.PHASES for role in h.ROLES for metric in h.ALL_METRICS]
        h.write_json(d/'metric_review/independent_metric_actual.json',dict(status='passed',original_fixture_checks=48,
                                                                       records=checks,maximum_absolute_error=0))
        h.write_json(d/'metric_review/analysis_audit.json',dict(status='passed',fixture_only=True))
        alignment=[dict(relation=r['pair_id'],split=split,length=n,role=role,samples=512,mapped_answer_match_rate=1.0,
                        right_label_offset=r['right_label_offset']) for r in relation for split in ('validation','test')
                   for n in p['lengths'] for role in h.ROLES]
        h.csv_write(d/'relation_label_alignment.csv',alignment)
        spec=importlib.util.spec_from_file_location('eight_relation_fixture_auditor',Path(__file__).with_name('final_reader_audit.py'))
        auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor);auditor.DIAG=d
        auditor.audit()
        if make_report:
            h.DIAG=d;report.DIAG=d
            report.main()
            h.require((d/'report.html').exists(),'Synthetic full report failed')
        result=dict(status='passed',synthetic_only=True,real_new_outcomes_read=False,trained=False,
            synthetic_per_length_rows=len(data),summary_rows=len(summary),condition_groups=len(index),
            contrast_counts=dict(Counter(r['contrast_type'] for r in summary)),
            cross_arch_rows=len(cross),gold_rows=len(gold),source_learning_rows=len(learning),
            independent_table_audit=h.js(d/'metric_review/final_reader_audit.json')['status'],
            full_report_and_plots_exercised=make_report,
            report_plots=h.js(d/'report_audit.json')['plots'] if make_report else 0,
            fixture_source_sha256=h.sha(__file__))
        h.DIAG=original_diag
        h.write_json(original_diag/'metric_review/complete_table_fixture.json',result)
        print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',action='store_true')
    main(p.parse_args().report)
