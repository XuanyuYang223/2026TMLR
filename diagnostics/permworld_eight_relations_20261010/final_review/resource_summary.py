"""Summarize completed receipts and saved resource snapshots without inference."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

OUT=Path(__file__).resolve().parent
DIAG=OUT.parent
ROOT=Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p):return json.loads(p.read_text())
def utc_epoch(s):return dt.datetime.fromisoformat(s).timestamp()
audit=js(DIAG/'resource_monitor_audit.json')
state=js(ROOT/'queue_state.json')
protocol=js(DIAG/'protocol.json')
rows=list(csv.DictReader((DIAG/'resource_monitor.csv').open()))
assert audit['status']=='completed' and audit['completed_verified']==state['completed_models']==48
assert state['status']=='completed' and len(rows)==audit['samples']==795
assert sha(DIAG/'resource_monitor.csv')==audit['csv_sha256']
verified={r['run_id']:r for r in audit['completion_records']}
endpoints=[]
for job in protocol['training_manifest']:
    p=ROOT/'models'/job['run_id']/'completed.json'
    r=js(p);e=r['curve'][-1]
    assert r['step']==e['step']==20000 and r['checkpoint_sha256']==verified[job['run_id']]['checkpoint_sha256']
    assert r['actual_initial_identity_verified'] is True and r['finite_gradient_updates']==20000
    endpoints.append({k:r[k] for k in ('run_id','architecture','encoding','task','step','elapsed_seconds','peak_cuda_allocated_bytes','checkpoint_sha256')}
        |{k:e[k] for k in ('answer_accuracy','numeric_accuracy','eos_accuracy','samples')}
        |dict(receipt_sha256=sha(p),endpoint_origin='saved20k_identity_validation_curve'))
assert len(endpoints)==48
with (OUT/'source_endpoint_validation.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(endpoints[0]));w.writeheader();w.writerows(endpoints)
gaps=[]
for a,b in zip(rows,rows[1:]):
    utc_delta=utc_epoch(b['utc'])-utc_epoch(a['utc'])
    monotonic_delta=float(b['elapsed_monitor_seconds'])-float(a['elapsed_monitor_seconds'])
    if abs(utc_delta-monotonic_delta)>10:
        gaps.append(dict(before_sample=int(a['sample']),after_sample=int(b['sample']),
            before_utc=a['utc'],after_utc=b['utc'],calendar_seconds=utc_delta,
            monotonic_seconds=monotonic_delta,difference_seconds=utc_delta-monotonic_delta))
def gpu_summary(records):
    out={}
    for k in ('utilization_percent','memory_used_mib','power_watts'):
        vals=[d[k] for r in records for d in json.loads(r['gpu_snapshot_json']) if d[k] is not None]
        out[k]=dict(samples=len(vals),minimum=min(vals),median=statistics.median(vals),maximum=max(vals))
    return out
running=[r for r in rows if r['queue_status']=='running']
groups=[]
for arch in ('transformer','gru'):
    for encoding in ('one_line','canonical_cycle'):
        rr=[r for r in endpoints if r['architecture']==arch and r['encoding']==encoding]
        assert len(rr)==12
        groups.append(dict(architecture=arch,encoding=encoding,source_models=len(rr),
            sum_recorded_source_monotonic_seconds=math.fsum(r['elapsed_seconds'] for r in rr),
            endpoint_answer_accuracy_min=min(r['answer_accuracy'] for r in rr),
            endpoint_answer_accuracy_max=max(r['answer_accuracy'] for r in rr)))
receipt_total=math.fsum(r['elapsed_seconds'] for r in endpoints)
assert abs(receipt_total-state['source_process_wall_seconds'])<1e-8
review=dict(status='completed_readonly_monitoring',scientific_geometry_not_evaluated_by_this_review=True,
    source_models_verified=48,source_updates=960000,groups=groups,
    protocol_sha256=state['protocol_sha256'],source_endpoint_csv_sha256=sha(OUT/'source_endpoint_validation.csv'),
    resource_csv_sha256=sha(DIAG/'resource_monitor.csv'),monitor_source_sha256=audit['monitor_source_sha256'],
    receipt_sum_monotonic_seconds=receipt_total,queue_calendar_wall_seconds=state['queue_wall_seconds'],
    first_monitor_utc=rows[0]['utc'],last_monitor_utc=rows[-1]['utc'],samples=len(rows),
    monitor_monotonic_elapsed_seconds=audit['elapsed_monitor_seconds'],
    monitor_calendar_elapsed_seconds=utc_epoch(rows[-1]['utc'])-utc_epoch(rows[0]['utc']),
    observed_process_user_delta_seconds=audit['observed_user_delta_seconds'],
    observed_process_system_delta_seconds=audit['observed_system_delta_seconds'],
    cpu_total_training_core_hours_available=False,cpu_scope=audit['cpu_scope'],
    source_completion_anomalies=audit['anomalies'],clock_domain_discrepancies=gaps,
    clock_cause_identified=False,gpu_snapshot_all=gpu_summary(rows),
    gpu_snapshot_queue_running_only=gpu_summary(running),
    gpu_scope='Instantaneous device-wide samples only; no GPU-active time or energy estimate. Final completed-state sample may include authorized Perm extraction begun09:55:18UTC, after queue completion09:55:01.986533UTC.',
    no_training_or_inference_launched=True,no_training_process_signalled=True,
    observer_and_health_polling_closed=True)
(OUT/'resource_summary.json').write_text(json.dumps(review,indent=2)+'\n')
text=[
 'Read-only queue supervision complete:48/48 newly authorized sources finished exactly20k updates.',
 'Frozen source/protocol identity,20-stage receipts and actual checkpoint SHA256 were checked as each source completed. No failed queue, non20k endpoint or frozen/weight hash anomaly was observed. Clock-domain divergence is a separate unresolved observation, not counted as a hash/endpoint failure.',
 'No training/inference was launched by this observer and no training process was stopped, signalled or changed. The observer exited normally after the48-source completion; its tool polling and health readers have also ended.',
 '',
 f"Recorded source monotonic time sum={receipt_total:.6f}s ({receipt_total/3600:.6f}h). Queue calendar-clock wall={state['queue_wall_seconds']:.6f}s ({state['queue_wall_seconds']/3600:.6f}h), including the preserved pre-wrapper elapsed. These are different clock domains and are not exact GPU-active time.",
 f"Resource snapshots:795, {rows[0]['utc']} to {rows[-1]['utc']}; monotonic observation elapsed={audit['elapsed_monitor_seconds']:.6f}s. Target sampling cadence was45 monotonic seconds.",
 f"Observed adjacent-snapshot process CPU increments: user={audit['observed_user_delta_seconds']:.2f}s, system={audit['observed_system_delta_seconds']:.2f}s. They are incomplete monitoring-window increments, not full training CPU core-hours. Source1 and the failed first metadata writer precede the monitor; each process's pre-first-observation and post-last-observation CPU tail is also omitted.",
 '',
 'Clock discrepancy: samples310->311 cross UTC02:04:59.338823->03:52:36.957526 with6457.619 calendar seconds but46.145 recorded monotonic seconds. The approximately106.9-minute UTC/monotonic offset shift is retained in timing_discrepancy_review.txt/json. Clock adjustment versus system/VM clock behavior cannot be distinguished; no human pause or actual GPU idle interval is inferred. Exact source start/end UTC are unavailable because queue stdout has no UTC fields and queue_state overwrites prior events.',
 '',
 'GPU statistics below describe samples with saved queue_status=running only. They are instantaneous device-wide values, not per-source active time, integrated energy or guaranteed maxima. The last completed-state sample (09:55:37UTC) follows the authorized extraction start09:55:18UTC, so it is excluded from these training-window descriptive ranges; the complete raw CSV is preserved.'
]
for k,v in review['gpu_snapshot_queue_running_only'].items():
    text.append(f"  {k}:N={v['samples']}, min={v['minimum']}, median={v['median']}, max={v['maximum']}")
text+=['','Saved20k identity-validation endpoints are listed for all48 sources in source_endpoint_validation.csv. No task, encoding or source was dropped for poor accuracy; this table contains saved validation statistics, not new forward passes, held-out test results or CKA conclusions.']
for g in groups:
    text.append(f"  {g['architecture']}/{g['encoding']}:12 sources, monotonic sum={g['sum_recorded_source_monotonic_seconds']:.6f}s, full-vocabulary answer-accuracy range={100*g['endpoint_answer_accuracy_min']:.4f}%..{100*g['endpoint_answer_accuracy_max']:.4f}%.")
text+=['','Resources are now closed. Subsequent Perm extraction/analysis and Graph training belong to the fixed bridge, not this supervisor. No further monitoring/GPU work is initiated here.']
(OUT/'resource_summary.txt').write_text('\n'.join(text)+'\n')
print(json.dumps({k:review[k] for k in ('status','source_models_verified','samples','receipt_sum_monotonic_seconds','queue_calendar_wall_seconds','source_endpoint_csv_sha256','resource_csv_sha256')},indent=2))
