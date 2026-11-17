"""Read-only fixed-subset link from saved FP32 gradients to the FP64 FD check."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = Path('/mnt/yangx-projects-f/20261110/permworld_closing_audit/gradient')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    spec = json.loads((HERE / 'spec.json').read_text())
    reports = []; checks = []; maximum_arithmetic = 0.
    for unit in spec['units']:
        name = unit['unit']; folder = HERE / 'results' / name
        completion = json.loads((folder / 'completion.json').read_text())
        assert completion['status'] == 'completed'
        for path, digest in completion['files'].items():
            if Path(path).name in ['finite_difference.csv', 'gradient_per_pair.csv.gz']:
                assert sha(path) == digest
        with (folder / 'finite_difference.csv').open(newline='') as handle:
            fd_rows = list(csv.DictReader(handle))
        assert len(fd_rows) == 48
        selected_ids = {(row['task'], row['pair_id']) for row in fd_rows}
        selected_rows = {}
        with gzip.open(folder / 'gradient_per_pair.csv.gz', 'rt', newline='') as handle:
            for row in csv.DictReader(handle):
                if (row['task'], row['pair_id']) in selected_ids:
                    selected_rows[row['task'], row['pair_id']] = row
        assert len(selected_rows) == 24
        cached = {}
        for fd in fd_rows:
            task, n, i = fd['task'], int(fd['length']), int(fd['position'])
            key = task, n, i
            if key not in cached:
                path = OUT / name / 'cells' / task / f'n{n}_i{i}.npz'
                assert sha(path) == completion['files'][str(path)]
                with np.load(path, allow_pickle=False) as z:
                    cached[key] = {k: z[k].copy() for k in z.files}
            z = cached[key]
            index = list(map(str, z['pair_ids'])).index(fd['pair_id'])
            row = selected_rows[task, fd['pair_id']]
            gradient = z['raw_gradient'][index]
            delta = z['actual_displacement'][index]
            donor = z['donor_difference'][index]
            axis = z['raw_unit_axis']
            direction = delta if fd['direction'] == 'local' else z['random_displacements'][0, index]
            derivative32 = float(gradient @ (direction / np.linalg.norm(direction)))
            analytic64 = float(fd['analytic']); numerical64 = float(fd['finite_difference'])
            allowed = 1e-7 + 1e-3 * abs(analytic64)
            projected = float((donor @ axis) ** 2 / (donor @ donor))
            arithmetic = max(abs(float(row['g_dot_delta']) - float(gradient @ delta)),
                             abs(float(row['projection_energy_fraction']) - projected),
                             abs(float(row['actual_displacement_norm']) - np.linalg.norm(delta)),
                             abs(float(row['gradient_norm']) - np.linalg.norm(gradient)))
            maximum_arithmetic = max(maximum_arithmetic, arithmetic)
            assert arithmetic <= 1e-12
            assert int(z['true_recipient_count'][index]) == int(row['true_recipient_count'])
            assert int(z['true_recipient_count'][index] + z['local_sign'][index]) == int(row['desired_local_count'])
            item = dict(unit=name, task=task, group=fd['group'], sign=fd['sign'], length=n, position=i,
                        pair_id=fd['pair_id'], direction=fd['direction'], primary_fp32_directional=derivative32,
                        shadow_fp64_analytic=analytic64, shadow_fp64_finite_difference=numerical64,
                        primary_vs_shadow_absolute_error=abs(derivative32 - analytic64),
                        allowed_absolute_error=allowed,
                        primary_vs_shadow_passed=abs(derivative32 - analytic64) <= allowed,
                        shadow_fd_passed=fd['passed'] == 'True')
            checks.append(item)
        reports.append(dict(unit=name, selected_cases=len(selected_rows), directional_checks=len(fd_rows),
                            gradient_receipt_sha256=sha(folder / 'completion.json')))
    passed = all(row['primary_vs_shadow_passed'] and row['shadow_fd_passed'] for row in checks)
    with (HERE / 'primary_gradient_fd_link.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0])); writer.writeheader(); writer.writerows(checks)
    result = dict(status='passed' if passed else 'failed', fixed_cases=120, directional_checks=len(checks),
                  maximum_primary_fp32_vs_shadow_fp64_error=max(row['primary_vs_shadow_absolute_error'] for row in checks),
                  maximum_independent_arithmetic_error=maximum_arithmetic,
                  failures=[row for row in checks if not row['primary_vs_shadow_passed'] or not row['shadow_fd_passed']],
                  units=reports, source_sha256=sha(__file__), spec_sha256=sha(HERE / 'spec.json'),
                  no_new_forwards=True, no_model_construction=True, no_fits=True,
                  scope='Same pre-fixed FD subset only. FP64 validates the nominal smooth graph; primary FP32 autograd is not a derivative of floating-point rounding.',
                  precision_limitation='Actual endpoint delta_m is from original cached FP32 early logits, while g uses CuDNN-off execution. Their allowed logit guard (1e-4) may exceed tiny delta_m. The linear_prediction_error field is arithmetic, not evidence of curvature or failure of local linearity.')
    (HERE / 'saved_gradient_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['units', 'failures']}, indent=2))
    if not passed: raise SystemExit('Fixed derivative link failed; retain errors and original tolerance.')


if __name__ == '__main__':
    main()
