"""Read-only evidence snapshot for a calendar/monotonic clock discrepancy."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path

DIAG = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations')
SCIENCE = DIAG.parents[1] / 'experiments/permworld_eight_relations_20261010'

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def stamp(s): return dt.datetime.fromisoformat(s).timestamp()
def js(p): return json.loads(p.read_text())

raw = (DIAG / 'resource_monitor.csv').read_bytes()
rows = list(csv.DictReader(raw.decode().splitlines()))
gaps = []
for a, b in zip(rows, rows[1:]):
    cal = stamp(b['utc']) - stamp(a['utc'])
    mono = float(b['elapsed_monitor_seconds']) - float(a['elapsed_monitor_seconds'])
    if abs(cal - mono) > 10:
        gaps.append(dict(before_sample=int(a['sample']), after_sample=int(b['sample']),
                         before_utc=a['utc'], after_utc=b['utc'],
                         calendar_seconds=cal, monitor_monotonic_seconds=mono,
                         clock_offset_change_seconds=cal-mono,
                         before_row=a, after_row=b))
selected = [r for r in rows if 301 <= int(r['sample']) <= 321]
receipts = []
for run in ('eightrel001_transformer_canonical_cycle_longest_increasing_run',
            'eightrel001_gru_one_line_anti_fixed_points'):
    p = ROOT / 'models' / run / 'completed.json'
    r = js(p)
    receipts.append(dict(run_id=run, completed_json_path=str(p),
                         completed_json_sha256=sha_bytes(p.read_bytes()),
                         file_mtime_utc=dt.datetime.fromtimestamp(p.stat().st_mtime,dt.timezone.utc).isoformat(),
                         file_mtime_epoch=p.stat().st_mtime, step=r['step'],
                         elapsed_seconds=r['elapsed_seconds'], checkpoint_sha256=r['checkpoint_sha256']))
state = js(ROOT / 'queue_state.json')
queue_log = (ROOT / 'queue.log').read_text().splitlines()
evidence = dict(reviewed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                status='clock_domain_discrepancy_located_cause_not_identified',
                source_readonly=True, no_model_or_test_launched=True,
                csv_snapshot_sha256=sha_bytes(raw), csv_rows_at_snapshot=len(rows),
                sample_gaps=gaps, selected_resource_rows=selected, receipts=receipts,
                receipt_mtime_calendar_gap_seconds=receipts[1]['file_mtime_epoch']-receipts[0]['file_mtime_epoch'],
                queue_state_at_review=state,
                queue_log_source_sha256=sha_bytes((ROOT / 'queue.log').read_bytes()),
                queue_log_relevant_lines=[dict(line_number=i+1,text=l) for i,l in enumerate(queue_log)
                    if any(t in l for t in ('canonical_cycle_longest_increasing_run','gru_one_line_anti_fixed_points','gru_one_line_components'))],
                queue_code_sha256=sha_bytes((SCIENCE/'queue.py').read_bytes()),
                train_code_sha256=sha_bytes((SCIENCE/'train.py').read_bytes()),
                timebase=dict(queue_wall='time.time (calendar clock), with elapsed_before from prior queue_state',
                              source_elapsed='time.monotonic (fixed source training loop plus prior resumed elapsed)',
                              observer_utc='datetime.now(timezone.utc) near sample start',
                              observer_elapsed='time.monotonic at row write after hashes/GPU query'),
                exact_start_end_utc_unavailable=True,
                exact_start_end_reason='Queue stdout has ordered starting/completed events without UTC fields. queue_state only keeps the latest event; receipt mtime is an observed file-write timestamp, not an instrumented exact training boundary.',
                causal_limit='The observed UTC-versus-monotonic offset shift cannot distinguish a civil-clock adjustment from a system/VM suspension or other clock-domain behavior. It does not establish a human pause or GPU idle/active duration.')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'timing_discrepancy_review.json').write_text(json.dumps(evidence,indent=2)+'\n')
g = gaps[0]
before = json.loads(g['before_row']['process_snapshot_json'])
after = json.loads(g['after_row']['process_snapshot_json'])
b = next(p for p in before if p['role']=='queue_descendant')
a = next(p for p in after if p['role']=='queue_descendant')
cpu = a['user_seconds']+a['system_seconds']-b['user_seconds']-b['system_seconds']
text = f'''Timing discrepancy review (read-only; no science code or model changed)

The calendar/monotonic discrepancy is localized to adjacent resource samples {g['before_sample']} and {g['after_sample']}:
  {g['before_utc']} -> {g['after_utc']}
  UTC separation: {g['calendar_seconds']:.6f} seconds.
  Recorded monotonic separation: {g['monitor_monotonic_seconds']:.6f} seconds.
  Difference: {g['clock_offset_change_seconds']:.6f} seconds, approximately106.9 minutes.
Observer elapsed values are captured after hashes/GPU-query work, whereas UTC is captured near sample start, so small processing offsets must not be treated as exact clock-step magnitudes. Across samples310->312 the calendar-minus-monotonic change is about6412.624s.

The same queue PID363553 and source25 PID417551, with the same /proc start_ticks, appear on both sides. Source25 user+system CPU increased by {cpu:.2f}s. This is consistent with the roughly46s monotonic interval and does not supply CPU/GPU measurements over the unsampled calendar span.

queue.py uses time.time() for queue_wall_seconds and preserves elapsed_before across the metadata-wrapper queue restart. train.py uses time.monotonic() for elapsed_seconds. The observer records both UTC and monotonic elapsed, with the processing-boundary caveat above. These costs are different clock domains and should remain separate.

Source25 completed at step20000 with elapsed_seconds={receipts[1]['elapsed_seconds']:.9f}. Its completed.json mtime is {receipts[1]['file_mtime_utc']}; source24's mtime is {receipts[0]['file_mtime_utc']}. Their file-write calendar separation is {evidence['receipt_mtime_calendar_gap_seconds']:.6f}s. Queue stdout confirms the ordered source24 completed -> source25 starting -> source25 completed -> source26 starting events, but contains no UTC timestamps; queue_state retains only its latest event. Exact per-source start/end UTC cannot be reconstructed from those artifacts.

The evidence rules out interpreting the entire calendar span as870s of instrumented training. It also does not support attributing the missing calendar span solely to observer descheduling while its monotonic clock advanced normally. The underlying cause remains unidentified: civil-clock change and clock behavior during system/VM suspension are possible examples, not diagnoses. No human pause, actual GPU idleness, GPU-active time or energy is inferred. There are no GPU measurements inside the large UTC gap.

Read-only monitoring continued at its fixed45s monotonic cadence. No training process was launched, stopped, signalled or altered for this review. The original CSV and frozen queue/train scripts remain unchanged. JSON retains the exact selected resource rows, receipt byte hashes, queue state, ordered log evidence and clock-domain limits.
'''
(OUT / 'timing_discrepancy_review.txt').write_text(text)
print(json.dumps({k:evidence[k] for k in ('status','csv_rows_at_snapshot','receipt_mtime_calendar_gap_seconds')},indent=2))
