"""Efficient per-anchor CSV export: materialize each saved array exactly once."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = ['longest', 'fixed_longest', 'turns_before_fixed']


def export(unit, route, variant, output):
    receipt = json.loads((HERE / 'results' / unit / 'completion.json').read_text())
    source = Path(receipt['predictions']['path'])
    with source.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == receipt['predictions']['sha256']
    with np.load(source, allow_pickle=False) as archive:
        ri = list(archive['route_names']).index(route)
        vi = list(archive['variant_names']).index(variant)
        scalars = {key: archive[key] for key in ['lengths', 'anchor_indices', 'matched', 'truth_fixed_present', 'truth_constraint_changes']}
        prediction = {task: archive['pred_'+task][ri, vi].copy() for task in TASKS}
        truth = {task: archive['truth_'+task].copy() for task in TASKS}
        # Check indexing/format equivalence on fixed representative anchors.
        for i in [0, len(scalars['lengths'])//2, len(scalars['lengths'])-1]:
            for task in TASKS:
                assert np.array_equal(prediction[task][i], archive['pred_'+task][ri, vi, i])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix+'.tmp')
    fields = ['unit', 'anchor_index', 'length', 'matched', 'truth_fixed_present', 'truth_constraint_changes', 'route', 'variant']
    fields += [prefix+task for task in TASKS for prefix in ['pred_', 'truth_', 'exact_']]

    def encode(value):
        value = np.asarray(value)
        if np.all(value == -1):
            return 'NONE'
        return str(int(value)) if value.ndim == 0 else f'{int(value[0])}:{int(value[1])}'

    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, n in enumerate(scalars['lengths']):
            row = dict(unit=unit, anchor_index=int(scalars['anchor_indices'][i]), length=int(n),
                       matched=bool(scalars['matched'][i]), truth_fixed_present=bool(scalars['truth_fixed_present'][i]),
                       truth_constraint_changes=bool(scalars['truth_constraint_changes'][i]), route=route, variant=variant)
            for task in TASKS:
                row['pred_'+task] = encode(prediction[task][i])
                row['truth_'+task] = encode(truth[task][i])
                row['exact_'+task] = bool(np.array_equal(prediction[task][i], truth[task][i]))
            writer.writerow(row)
    os.replace(temporary, output)
    return str(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unit')
    parser.add_argument('--route', default='aligned_complement')
    parser.add_argument('--variant', default='original')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--all-primary', action='store_true')
    args = parser.parse_args()
    if args.all_primary:
        base = Path('/mnt/yangx-projects-f/20261110/permworld_lemma_composition/exports')
        for unit in ['archdev001', 'unit02', 'unit03', 'unit04', 'unit05']:
            for route in ['aligned_complement', 'same_model_twice']:
                print(export(unit, route, 'original', base/f'{unit}-{route}.csv'), flush=True)
    else:
        if not args.unit or not args.out:
            parser.error('--unit and --out are required unless --all-primary is set')
        print(export(args.unit, args.route, args.variant, args.out))


if __name__ == '__main__':
    main()
