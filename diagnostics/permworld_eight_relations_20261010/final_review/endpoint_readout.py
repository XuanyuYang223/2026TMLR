"""Print an existing 20k validation curve endpoint; never load a model."""
import json
from pathlib import Path
import sys

ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations')
for run_id in sys.argv[1:]:
    assert run_id.startswith('eightrel001_') and '/' not in run_id
    path = ROOT / 'models' / run_id / 'completed.json'
    r = json.loads(path.read_text())
    endpoint = r['curve'][-1]
    assert r['step'] == endpoint['step'] == 20000
    print(json.dumps(dict(run_id=run_id, curve_endpoint=endpoint,
                          endpoint_scope='Saved source validation identity samples; full-vocabulary answer accuracy; no model re-evaluation.',
                          receipt_path=str(path))))
