"""Boolean intersection of prespecified lengthwise comparisons; no new fit."""
from pathlib import Path
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent
FOLDER=HERE/'summary'
rows=list(csv.DictReader((FOLDER/'per_length_contrasts.csv').open(newline='')))
result=[]
for unit in ['archdev001','unit02','unit03','unit04','unit05']:
    for encoding in ['canonical_cycle','one_line']:
        subset=[r for r in rows if r['unit']==unit and r['encoding']==encoding]
        assert len(subset)==21
        count=sum(float(r['complement_minus_best_alternative'])>0 and
                  float(r['trained_complement_minus_initial_complement'])>0 for r in subset)
        result.append(dict(unit=unit,encoding=encoding,registered_lengths=21,
            C_above_both_alternatives_and_initial_lengths=count,
            scope='Descriptive intersection of already-prespecified per-length comparisons; not independent replications'))
with (FOLDER/'conjunction_direction_counts.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
metadata=dict(source_sha256=hashlib.sha256((FOLDER/'per_length_contrasts.csv').read_bytes()).hexdigest(),
    definition='Within each length, trained C > trained I AND trained C > trained inverse AND trained C > separately fitted initial C. Strict original comparisons; no new threshold or inferential test.',results=result)
(FOLDER/'conjunction_direction_counts.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(result))
