"""Historical-development geometry only: select one actual recurrent layer."""
from collections import defaultdict
import numpy as np
import torch

from .shared import ROOT,DIAG,OUT,TASKS,read,write,write_csv,sha,families,transformed
from .features import collect
from .probes import fit_bundle,predict,edit_state
from .recurrent import load_model,parameter_hash


def evaluate_dev(selected,task,states,lookup,bundle):
    groups=defaultdict(list)
    layer,point=bundle['layer'],bundle['point']
    for f in selected:
        if f['split']!='dev':continue
        for r in f['pairs']:
            if r['kind'] not in ('simple','count_matched'):continue
            key=lambda side:(f['n'],f['center'],transformed(r[side],task))
            hr=states[lookup[key('recipient')],point,layer]
            hd=states[lookup[key('donor')],point,layer]
            edited,coordinate=edit_state(hr,hd,bundle['direction'])
            before=float(predict(bundle['checker'],hr))
            after=float(predict(bundle['checker'],edited))
            donor=float(predict(bundle['checker'],hd))
            original=r['recipient_label'];desired=r['donor_label'];sign=r['direction']
            crossing=int((before>=.5)==original and (after>=.5)==desired)
            groups[r['kind'],sign].append(dict(crossing=crossing,shift=sign*(after-before),
                recipient_accuracy=int((before>=.5)==original),donor_accuracy=int((donor>=.5)==desired),
                counter_error=abs(float(predict(bundle['count'],hr))-(r['recipient_prefix_count']-(original if point==0 else 0)))))
    return [dict(task=task,group='main' if group=='simple' else 'matched',sign=sign,
                  layer=layer+1,point=point,timing='after_right_value' if point==1 else 'before_right_value',
                  pairs=len(values),true_boundary_crossing=float(np.mean([r['crossing'] for r in values])),
                  signed_checker_shift=float(np.mean([r['shift'] for r in values])),
                  recipient_checker_accuracy=float(np.mean([r['recipient_accuracy'] for r in values])),
                  donor_checker_accuracy=float(np.mean([r['donor_accuracy'] for r in values])),
                  prefix_count_probe_mae=float(np.mean([r['counter_error'] for r in values])))
            for (group,sign),values in sorted(groups.items())]


def run():
    from .run import verify_engineering
    verify_engineering()
    if (DIAG/'selection.json').exists():return read(DIAG/'selection.json')
    source=read(DIAG/'source_audit.json');unit=source['units'][0]
    selected=families(unit['unit'])
    records=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for task in TASKS:
        model=load_model(unit['models'][task],device,source['model'])
        states,nodes,lookup,receipt=collect(unit['unit'],task,model,('fit','dev'))
        for point in (0,1):
            for layer in range(4):
                bundle=fit_bundle(states,nodes,layer,point,unit['unit'],task)
                records.extend(evaluate_dev(selected,task,states,lookup,bundle))
        assert parameter_hash(model)==unit['models'][task]['parameter_sha256']
        del model,states
        if torch.cuda.is_available():torch.cuda.empty_cache()
    eligible=[]
    for layer in range(1,5):
        cells=[r for r in records if r['layer']==layer and r['point']==1]
        assert len(cells)==8
        eligible.append(dict(layer=layer,min_crossing=min(r['true_boundary_crossing'] for r in cells),
                             min_signed_shift=min(r['signed_checker_shift'] for r in cells)))
    choice=max(eligible,key=lambda r:(r['min_crossing'],r['min_signed_shift'],r['layer']))
    write_csv(DIAG/'development_geometry.csv',records)
    selection=dict(status='selected',historical_unit=unit['unit'],layer=choice['layer'],layer_zero_based=choice['layer']-1,
        timing='after_right_value',point=1,dose=1.,dimension=1,threshold=.5,
        minimum_historical_dev_true_crossing=choice['min_crossing'],
        manipulation_development_reliable=choice['min_crossing']>=.8 and choice['min_signed_shift']>0,
        min_signed_checker_shift=choice['min_signed_shift'],candidates=eligible,
        criterion='Maximize minimum independent-checker true boundary crossing across task/group/sign dev cells; then minimum directed score shift; then higher layer. BEFORE readouts are descriptive only. No count-output endpoint is inspected for selection.',
        no_formal_test_output_access=True,no_count_effect_selection=True,
        development_geometry_sha256=sha(DIAG/'development_geometry.csv'))
    write(DIAG/'selection.json',selection)
    print(selection,flush=True)
    return selection
