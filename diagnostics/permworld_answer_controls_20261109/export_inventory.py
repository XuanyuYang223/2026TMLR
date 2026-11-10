"""Export the already audited source inventory without reading model weights."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
inventory=json.loads((HERE/'source_inventory.json').read_text())
rows=inventory['models']
fields=list(dict.fromkeys(k for row in rows for k in row))
with (HERE/'checkpoint_inventory.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields)
    writer.writeheader();writer.writerows(rows)
(HERE/'missing_checkpoints.json').write_text(json.dumps(dict(
    missing_required_artifacts=inventory['missing_required_artifacts'],
    hash_mismatches=inventory['hash_mismatches'],
    scope='All eligible completed 20k main-experiment specialists in the five audited units; development runs are not added as repetitions.'),indent=2)+'\n')
print('Exported',len(rows),'audited specialists; no model forward or new metric.')
