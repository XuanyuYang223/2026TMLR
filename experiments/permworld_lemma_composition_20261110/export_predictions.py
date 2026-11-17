"""Export saved per-anchor predictions; this performs no fitting or inference."""
from pathlib import Path
import argparse
import csv
import json

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DIAG=ROOT/'diagnostics/permworld_lemma_composition_20261110'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--unit',required=True);parser.add_argument('--route',default='aligned_complement');parser.add_argument('--variant',default='original');parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    receipt=json.loads((DIAG/'results'/args.unit/'completion.json').read_text())
    with np.load(receipt['predictions']['path'],allow_pickle=False) as z:
        ri=list(z['route_names']).index(args.route);vi=list(z['variant_names']).index(args.variant)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        fields=['unit','anchor_index','length','matched','truth_fixed_present','truth_constraint_changes','route','variant']
        fields += [prefix+key for key in ['longest','fixed_longest','turns_before_fixed'] for prefix in ['pred_','truth_','exact_']]
        with args.out.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for i,n in enumerate(z['lengths']):
                row=dict(unit=args.unit,anchor_index=int(z['anchor_indices'][i]),length=int(n),matched=bool(z['matched'][i]),truth_fixed_present=bool(z['truth_fixed_present'][i]),truth_constraint_changes=bool(z['truth_constraint_changes'][i]),route=args.route,variant=args.variant)
                for key in ['longest','fixed_longest','turns_before_fixed']:
                    p=z['pred_'+key][ri,vi,i];t=z['truth_'+key][i]
                    encode=lambda x:'NONE' if np.all(np.asarray(x)==-1) else str(int(x)) if np.asarray(x).ndim==0 else f'{int(x[0])}:{int(x[1])}'
                    row['pred_'+key]=encode(p);row['truth_'+key]=encode(t);row['exact_'+key]=bool(np.array_equal(p,t))
                writer.writerow(row)
    print(args.out)


if __name__=='__main__':main()
