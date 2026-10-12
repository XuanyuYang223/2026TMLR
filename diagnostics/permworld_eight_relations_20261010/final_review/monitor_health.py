"""Read the existing observer's latest snapshot; no model or /proc mutation."""
import csv
import datetime as dt
import json
from pathlib import Path

DIAG = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((DIAG/'resource_monitor.csv').open()))
r = rows[-1]
a = json.loads((DIAG/'resource_monitor_audit.json').read_text())
issues = []
if r['queue_status'] == 'running' and not json.loads(r['queue_pids']):
    issues.append('Queue status still running but no queue process in latest saved /proc snapshot.')
if r['queue_status'] == 'failed':
    issues.append('Saved queue status failed.')
issues += json.loads(r['anomalies_json'])
age = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(r['utc'])).total_seconds()
if age > 120:
    issues.append('Observer latest UTC stamp over120s old; clock shift or observation gap remains possible.')
if issues:
    print(json.dumps(dict(event='monitor_health_issue', utc=r['utc'], details=issues,
                          queue_status=r['queue_status'], current_run=r['current_run'],
                          completed_verified=a['completed_verified'])))
