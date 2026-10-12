"""Validate the saved delivery and seal its manifest; no scientific computation."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
def read(name): return json.loads((HERE/name).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(name, value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rows(name):
    with (HERE/name).open() as f:return list(csv.DictReader(f))
class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self, tag, attrs):
        for k,v in attrs:
            if k in ('href','src') and v and '://' not in v and not v.startswith('#'):self.paths.append(v)

def verify():
    checks=[]
    def check(ok, detail):checks.append({'passed':bool(ok),'detail':detail})
    freeze=read('freeze.json')
    check(sha(HERE/'protocol.json')==freeze['protocol_sha256'],'Scientific protocol SHA unchanged')
    check(sha(HERE/'candidate_pairs.csv')==freeze['candidate_pairs_sha256'],'Frozen 492 candidate list unchanged')
    check(sha(HERE/'classification_manifest.csv')==freeze['classification_manifest_sha256'],'Pre-score classification unchanged')
    check(sha(HERE/'classification_audit.json')==freeze['classification_audit_sha256'],'Pre-score proof audit unchanged')
    for path,expected in freeze['source_sha256'].items():check(sha(path)==expected,'Frozen source '+path)
    aud=read('analysis_audit.json')
    for path,expected in aud['output_sha256'].items():check(sha(path)==expected,'Analysis output '+path)
    old=read('existing_results_review.json')
    for name,expected in old['output_sha256'].items():check(sha(HERE/name)==expected,'Existing table review '+name)
    wa=read('W_actual_audit.json')
    for name,expected in wa['outputs'].items():check(sha(HERE/name)==expected,'Actual W output '+name)
    rep=read('report_audit.json')
    for name,expected in rep['outputs_sha256'].items():check(sha(HERE/name)==expected,'Reader output '+name)
    for name in ('protection_start.json','protection_end.json','analysis_audit.json',
                 'existing_results_review.json','result_reader_audit.json','W_actual_audit.json',
                 'W_reader_validation.json','engineering_fixture.json'):
        check(read(name)['status']=='passed','Completed audit '+name)
    expect={'candidate_pairs.csv':492,'classification_manifest.csv':132,'recomputed_main32.csv':32,
            'six_negative_bt.csv':6,'six_negative_bt_by_length.csv':126,
            'pair_per_length.csv':20268,'pair_masks.csv':20268,'pair_summary.csv':996,
            'category_summary.csv':456,'designated_category_comparisons.csv':304,
            'common_support_training_comparisons.csv':228,'W_actual_per_anchor.csv':92160,
            'W_actual_per_length.csv':372,'W_actual_summary.csv':28,
            'designated_endpoint_verification.csv':2760,'designated_aggregate_verification.csv':152,
            'independent_covariance_verification.csv':40}
    counts={}
    for name,n in expect.items():counts[name]=len(rows(name));check(counts[name]==n,'Complete row count '+name)
    candidates=rows('candidate_pairs.csv')
    check(Counter(r['domain'] for r in candidates)=={'permworld':480,'graphworld':12},'All frozen domain candidates')
    check(len({r['pair_id'] for r in candidates})==492,'No duplicate candidate identities')
    negative=rows('six_negative_bt.csv')
    check(all(r['encoding']=='canonical_cycle' and float(r['Bt'])<0 for r in negative),'All six old negative primary cells identified')
    check(all(r['left_test_majority_source'] and r['right_test_majority_source'] for r in negative),'Length-mode source retained for both tasks')
    missing=sum(any(not r['fixed_transform_score_'+s] for s in ('initial','trained')) for r in rows('pair_summary.csv'))
    check(missing==0,'Complete new score support; empty mathematical class remains separately undefined')
    for r in rows('pair_summary.csv'):
        check(all(math.isfinite(float(r['fixed_transform_score_'+s])) for s in ('initial','trained')),'Finite saved score '+r['pair_id']+'/'+r['control'])
    links=Links();links.feed((HERE/'reader_report.html').read_text())
    for name in links.paths:check((HERE/name).is_file(),'Reader link/image '+name)
    for name in ('REPRODUCE.txt','next_phase_candidates.txt','replication_candidates.txt','results_paragraph_zh_en.txt'):
        check((HERE/name).is_file(),'Delivery entry '+name)
    end=read('protection_end.json')
    check(end['all_protected_bytes_match_start'] and end['tracked_Git_snapshots_equal'],'Named2580 protected bytes and tracked Git snapshots match')
    check(end['files']==read('protection_start.json')['files']==2580,'Fixed concrete protection scope count')
    result={'status':'passed' if all(c['passed'] for c in checks) else 'failed',
            'checks':checks,'check_count':len(checks),'csv_row_counts':counts,
            'pair_summary_rows_missing_phase_scores':missing,'report_links_checked':len(links.paths),
            'new_scientific_metrics':0,'new_forward':0,'new_training':0,
            'scope':'Saved delivery hashes, row coverage and report links only. Does not rerun CKA or training.'}
    write('delivery_verification.json',result)
    assert result['status']=='passed',[c for c in checks if not c['passed']]
    print(json.dumps({'status':result['status'],'checks':len(checks),'links':len(links.paths)}))

def finalize():
    assert not (HERE/'completion.json').exists(),'Final completion manifest is not silently refreshed'
    verify()
    visual=read('visual_review.json');assert visual['status']=='passed'
    aud=read('analysis_audit.json');old=read('existing_results_review.json')
    end=read('protection_end.json');report=read('report_audit.json')
    bindings={}
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and path.name!='completion.json':
            bindings[str(path.relative_to(HERE))]={'sha256':sha(path),'bytes':path.stat().st_size}
    out=dict(status='completed_and_stopped',finished_utc=datetime.now(timezone.utc).isoformat(),
        task='2026-10-11 PermWorld / GraphWorld existing-result review supplement',
        analysis_level='Exploratory, cached existing inputs and genuine saved step0/20k endpoints',
        source_scope=read('source_scope.json')['expected_and_actual'],
        source_scope_differences=[],new_models=0,new_worlds=0,new_training_updates=0,
        new_optimizer_calls=0,new_forward=0,new_probe_training=0,new_metric_layer_expansion=0,
        cache_only_cpu_analysis=True,GPU_execution_called=False,
        new_pairs={'permworld':480,'graphworld':12,'independent_replication_units_added':0},
        new_score_rows={'per_length_control':aud['pair_length_control_rows'],'per_pair_control':aud['pair_summary_rows']},
        primary_old_Bt_signs=old['primary_Bt_signs'],primary_old_Delta_signs=old['primary_Delta_signs'],
        old_three_metric_directions=old['metric_direction_agreement'],
        new_designated_minus_not_designated_mean_trained=report['summary'],
        old_arithmetic={'per_length_checks':old['saved_length_arithmetic_checks'],
                        'aggregate_checks':old['saved_summary_arithmetic_checks'],'maximum_difference':old['maximum_saved_value_difference']},
        new_numerical_verification={'original_designated_per_length_checks':aud['original_designated_per_length_phase_checks'],
            'original_aggregate_checks':aud['original_designated_aggregate_phase_checks'],
            'independent_covariance_checks':aud['independent_covariance_checks'],
            'maximum_original_per_length_error':aud['designated_maximum_absolute_error'],
            'maximum_independent_error':aud['independent_maximum_absolute_error'],'original_tolerance':1e-8},
        W_actual={'relation_anchor_exposures':92160,'unique_base_inputs':{'permworld':10752,'graphworld':1536},
                  'original_analysis_count_checks':372,'original_wrong_transform_retained':True},
        protection={'concrete_files':end['files'],'start_end_status':end['status'],
                    'all_named_bytes_equal':end['all_protected_bytes_match_start'],
                    'tracked_Git_snapshots_equal':end['tracked_Git_snapshots_equal'],
                    'limitation':end['limitation']},
        actual_new_CKA_CPU_wall_seconds=aud['wall_seconds'],
        resource_record='fixed_transform_resources.txt (existing GNUtime receipt; not GPU time)',
        future_training_proposals='Only three main comparisons and12/24-model budget scenarios; not executed or power guarantees',
        protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(HERE/'freeze.json'),
        git_HEAD=read('git_start.json')['HEAD'],
        saved_engineering_exceptions=['W_engineering_repair.json','engineering_repairs/freeze_restore.json'],
        no_auto_followup=True,no_new_training=True,no_new_world=True,no_delete=True,no_move=True,
        no_upload_or_publish=True,no_F17_edit=True,no_old_frozen_artifact_edit=True,
        artifact_bindings=bindings)
    write('completion.json',out)
    print(json.dumps({'status':out['status'],'bound_new_artifacts':len(bindings),'new_training':0,'new_forward':0}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['verify','finalize'],required=True)
    {'verify':verify,'finalize':finalize}[parser.parse_args().stage]()
