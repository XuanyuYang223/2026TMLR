"""Aggregate the frozen ordinary-CKA bridge; no CKA or forward computation."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.permworld_answer_controls_20261109.common import load_config,csv_read,csv_write

HERE=Path(__file__).resolve().parent


def main():
    c=load_config()
    assert json.loads((HERE/'cka_completion.json').read_text())['status']=='completed'
    rows=[]
    for u in c['units']:
        unit=u['unit']
        for arch in c['architectures']:
            for enc in c['encodings']:
                phase_scores={}
                for phase in c['phases']:
                    by_key={}
                    base=Path(c['output'])/'scores'/f'{unit}_{arch}_{enc}_{phase}'
                    keys={(r['left'],r['right']) for r in c['relations']}
                    for n in c['lengths']:
                        for r in csv_read(base/f'n{n}.csv'):
                            if r['cka_version']!='within_length_debiased' or (r['source_task'],r['candidate_task']) not in keys:continue
                            by_key.setdefault((r['source_task'],r['candidate_task'],r['transform']),[]).append(float(r['ordinary_bridge']) if r['ordinary_bridge'] else np.nan)
                    aggregate={k:float(np.mean(v)) if len(v)==21 and np.isfinite(v).all() else None for k,v in by_key.items()}
                    phase_scores[phase]=aggregate
                for relation in c['relations']:
                    values={}
                    row=dict(unit=unit,architecture=arch,encoding=enc,relation=relation['pair_id'],version='within_length_ordinary_bridge')
                    for phase,aggregate in phase_scores.items():
                        scores={t:aggregate[relation['left'],relation['right'],t] for t in c['actions']}
                        row.update({phase+'_'+t+'_cka':v for t,v in scores.items()})
                        correct=scores[relation['input_transform']]
                        other=[v for t,v in scores.items() if t!=relation['input_transform']]
                        advantage=correct-max(other) if correct is not None and all(v is not None for v in other) else None
                        row[phase+'_transform_advantage']=advantage;values[phase]=advantage
                    row['advantage_training_increment']=values['trained']-values['initial'] if all(v is not None for v in values.values()) else None
                    rows.append(row)
    csv_write(HERE/'ordinary_bridge_by_unit.csv',rows)
    original=csv_read(HERE/'summary/transform_advantage_by_unit.csv')
    combined=rows+[dict(r,version=r['cka_version']) for r in original]
    summary=[]
    for cohort,units in [('new_units',[u['unit'] for u in c['units'] if u['unit']!='archdev001']),('all_units',[u['unit'] for u in c['units']])]:
        for version in [c['versions'][0],'within_length_ordinary_bridge',*c['versions'][1:]]:
            selected=[r for r in combined if r['unit'] in units and r['version']==version]
            for effect in ['trained_transform_advantage','advantage_training_increment']:
                vals=np.array([float(r[effect]) if r.get(effect) not in ('',None) else np.nan for r in selected])
                summary.append(dict(cohort=cohort,version=version,effect=effect,expected_values=len(units)*32,finite_values=int(np.isfinite(vals).sum()),
                    mean=float(vals.mean()) if len(vals)==len(units)*32 and np.isfinite(vals).all() else None,
                    interpretation='Descriptive fixed-grid mean; ordinary bridge separates length pooling from estimator change.'))
    csv_write(HERE/'ordinary_bridge_summary.csv',summary)
    print(json.dumps(dict(status='completed',bridge_unit_cell_rows=len(rows),summary_rows=len(summary),no_new_CKA=True)))


if __name__=='__main__':main()
