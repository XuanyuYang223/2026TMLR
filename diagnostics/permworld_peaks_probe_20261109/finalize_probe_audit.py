"""Endpoint hashes and independent saved-table arithmetic; never fit a probe.

Run only after all five supplied-protocol units finish and before publication.
The protected scope is the explicit start list, historical files actually read,
the frozen sources/protocol, and the supplied archive/vendor copies. This is
neither continuous monitoring nor a repository-wide immutability assertion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import time
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GROUPS = ('unit', 'encoding', 'phase', 'method', 'transform', 'label', 'support')
TOLERANCE = 1e-12


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


class Checks:
    def __init__(self):
        self.total = 0
        self.failures = []
        self.maximum_arithmetic_difference = 0.

    def require(self, condition, check_name, **context):
        self.total += 1
        if not condition:
            self.failures.append(dict(check=check_name, **context))

    def close(self, observed, expected, check_name, **context):
        a, b = finite(observed), finite(expected)
        if a is None or b is None:
            self.require(a is None and b is None, check_name, observed=a, expected=b, **context)
            return
        difference = abs(a-b)
        self.maximum_arithmetic_difference = max(self.maximum_arithmetic_difference, difference)
        self.require(difference <= TOLERANCE, check_name, observed=a, expected=b,
                     absolute_difference=difference, **context)


def check_file(item):
    path, expected = item
    result = dict(path=path, expected_sha256=expected)
    try:
        before = Path(path).stat()
        actual = sha(path)
        after = Path(path).stat()
        result.update(actual_sha256=actual, size_bytes=after.st_size,
                      passed=actual == expected and
                      (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
    except Exception as error:
        result.update(passed=False, error_type=type(error).__name__, error=str(error))
    return result


def audit_saved(config):
    check = Checks()
    units = [unit['unit'] for unit in config['units']]
    expected_units = ['archdev001', 'unit02', 'unit03', 'unit04', 'unit05']
    check.require(units == expected_units, 'five_fixed_units', units=units)
    check.require(sorted(p.name for p in (HERE/'results').iterdir() if p.is_dir()) == sorted(units),
                  'no_missing_or_extra_result_units')
    mathematical = {u['unit']: u for u in read(HERE/'math_split_audit.json')['units']}
    legacy_learning = rows(ROOT/'diagnostics/permworld_answer_controls_20261109/learning_summary.csv')
    legacy = {(r['unit'], r['encoding'], r['task']): r for r in legacy_learning
              if r['architecture'] == 'gru'}
    template = {tuple(row[key] for key in GROUPS if key != 'unit')
                for row in rows(HERE/'supplied_historical/results/summary.csv')}
    output = []
    completion_files = {}
    for unit in units:
        base = HERE/'results'/unit
        receipt = read(base/'completion.json')
        check.require(receipt['status'] == 'completed' and receipt['unit'] == unit,
                      'unit_completed', unit=unit)
        check.require(receipt['protocol_sha256'] == sha(HERE/'protocol.json'),
                      'completion_protocol_binding', unit=unit)
        check.require(receipt.get('no_neural_forward') is True and
                      receipt.get('no_base_training') is True and
                      receipt.get('fitted_only_linear_ridge_probes') is True,
                      'bounded_execution_receipt', unit=unit)
        for filename, digest in receipt['result_sha256'].items():
            completion_files[str(base/filename)] = digest
        for filename, digest in receipt['imported_feature_sha256'].items():
            completion_files[str(Path(config['output'])/'features'/unit/filename)] = digest
        coverage = rows(base/'coverage.csv')
        by_length = rows(base/'by_length.csv')
        strata = rows(base/'by_length_and_count.csv')
        summaries = rows(base/'summary.csv')
        learning = rows(base/'learning_by_length.csv')
        cov = {int(r['length']): r for r in coverage}
        check.require(len(cov) == len(coverage) == 21 and set(cov) == set(config['lengths']),
                      'complete_unique_coverage_lengths', unit=unit)
        if unit == 'archdev001':
            reference_coverage = rows(HERE/'supplied_historical/results/coverage.csv')
            expected_coverage = {int(r['length']): r for r in reference_coverage}
            expected_matched = sum(int(r['matched_n']) for r in reference_coverage)
        else:
            expected_coverage = {int(r['length']): r for r in mathematical[unit]['per_length']}
            expected_matched = mathematical[unit]['matched_total']
        for n, row in cov.items():
            check.require(row['unit'] == unit, 'coverage_unit', unit=unit, length=n)
            for key in ('fit_n', 'dev_n', 'test_n', 'matched_n',
                        'validation_excluded_n', 'test_excluded_n'):
                check.require(int(row[key]) == int(expected_coverage[n][key]),
                              'coverage_matches_frozen_data_audit', unit=unit, length=n, field=key)
        matched = sum(int(r['matched_n']) for r in coverage)
        check.require(matched == expected_matched,
                      'unit_specific_matched_total', unit=unit, observed=matched)
        groups = defaultdict(list)
        count_groups = defaultdict(list)
        for row in by_length:
            key = tuple(row[k] for k in GROUPS)
            groups[key].append(row)
            n = int(row['length'])
            context = dict(unit=unit, encoding=row['encoding'], phase=row['phase'],
                           method=row['method'], transform=row['transform'],
                           label=row['label'], support=row['support'], length=n)
            check.require(row['unit'] == unit and n in cov, 'length_row_identity', **context)
            expected_n = int(cov[n]['matched_n' if row['support'] == 'matched' else 'test_n'])
            check.require(int(row['n']) == expected_n and 0 < int(row['defined_n']) <= expected_n,
                          'length_support_counts', **context)
            auc = finite(row['auc'])
            check.require(auc is not None and 0 <= auc <= 1, 'finite_bounded_length_auc', **context)
            if row['support'] == 'matched':
                check.require(int(row['defined_n']) == expected_n, 'all_matched_aucs_defined', **context)
            alpha = finite(row['alpha'])
            check.require(alpha is None if row['method'] == 'position_prior' else alpha in config['alphas'],
                          'registered_alpha_or_prior', **context)
        for row in strata:
            count_groups[tuple(row[k] for k in GROUPS)+(int(row['length']),)].append(row)
        observed_template = {tuple(row[k] for k in GROUPS if k != 'unit') for row in summaries}
        check.require(observed_template == template and len(summaries) == len(template) == 104,
                      'complete_fixed_score_inventory', unit=unit)
        for row in summaries:
            key = tuple(row[k] for k in GROUPS)
            selected = groups[key]
            check.require(len(selected) == 21 and
                          {int(r['length']) for r in selected} == set(config['lengths']),
                          'twenty_one_unique_components', unit=unit, key=list(key))
            values = [finite(r['auc']) for r in selected]
            check.require(int(row['finite_lengths']) == 21 and all(v is not None for v in values),
                          'no_nan_length_omission', unit=unit, key=list(key))
            if len(values) == 21 and all(v is not None for v in values):
                check.close(row['auc'], math.fsum(values)/21, 'equal_length_aggregate',
                            unit=unit, key=list(key))
            check.require(int(row['n']) == sum(int(r['n']) for r in selected) and
                          int(row['defined_n']) == sum(int(r['defined_n']) for r in selected),
                          'aggregate_sample_counts', unit=unit, key=list(key))
            for length_row in selected:
                n = int(length_row['length'])
                bins = count_groups[key+(n,)]
                check.require(len(bins) == len({r['source_peak_count'] for r in bins}),
                              'unique_count_strata', unit=unit, length=n, key=list(key))
                check.require(sum(int(r['n']) for r in bins) == int(length_row['n']) and
                              sum(int(r['defined_n']) for r in bins) == int(length_row['defined_n']),
                              'count_stratum_support_reconstruction', unit=unit, length=n, key=list(key))
                defined = [(finite(r['auc']), int(r['defined_n'])) for r in bins if int(r['defined_n'])]
                check.require(all(v is not None for v, _ in defined),
                              'defined_strata_have_scores', unit=unit, length=n, key=list(key))
                if defined and all(v is not None for v, _ in defined):
                    reconstructed = math.fsum(v*weight for v, weight in defined)/sum(w for _, w in defined)
                    check.close(length_row['auc'], reconstructed, 'defined_example_weighted_strata',
                                unit=unit, length=n, key=list(key))
        for row in by_length:
            if row['method'] not in ('transfer', 'descents', 'recoils'):
                continue
            source_key = (unit, row['encoding'], row['phase'], 'source_peaks',
                          'identity', 'peak', row['support'])
            source = next(r for r in groups[source_key] if r['length'] == row['length'])
            check.close(row['alpha'], source['alpha'], 'same_source_probe_alpha_across_targets',
                        unit=unit, length=int(row['length']))
        learning_groups = defaultdict(list)
        for row in learning:
            learning_groups[row['encoding'], row['task']].append(row)
            check.require(row['unit'] == unit and int(row['n']) == int(cov[int(row['length'])]['test_n']),
                          'learning_same_original_test_support', unit=unit)
            check.require(all(finite(row[k]) is not None and 0 <= float(row[k]) <= 1
                              for k in ('accuracy', 'baseline')), 'valid_learning_accuracy', unit=unit)
        check.require(len(learning) == 84 and len(learning_groups) == 4,
                      'complete_learning_inventory', unit=unit)
        for (enc, task), group in learning_groups.items():
            check.require(len(group) == 21 and {int(r['length']) for r in group} == set(config['lengths']),
                          'complete_learning_lengths', unit=unit, encoding=enc, task=task)
            prior = legacy[unit, enc, task]
            check.close(math.fsum(float(r['accuracy']) for r in group)/21,
                        prior['exact_match_accuracy'], 'learning_matches_prior_saved_accuracy',
                        unit=unit, encoding=enc, task=task)
            check.close(math.fsum(float(r['baseline']) for r in group)/21,
                        prior['train_length_mode_accuracy'], 'learning_matches_prior_length_mode',
                        unit=unit, encoding=enc, task=task)
        output.append(dict(unit=unit,summary_rows=len(summaries),per_length_rows=len(by_length),
                           count_strata_rows=len(strata),learning_rows=len(learning),matched_n=matched,
                           fit_n=sum(int(r['fit_n']) for r in coverage),
                           dev_n=sum(int(r['dev_n']) for r in coverage)))
    return check, output, completion_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    started, timer = utc(), time.monotonic()
    config = read(HERE/'protocol.json')
    freeze = read(HERE/'freeze.json')
    completion = read(HERE/'run_completion.json')
    assert completion['status'] == 'completed' and completion['stage'] == 'all', 'Wait for all five units'
    assert read(HERE/'historical_reproduction.json')['status'] == 'passed'
    start = read(HERE/'protection_start.json')
    historic = read(Path(config['output'])/'historical_cache_read.json')
    archive_manifest = read(HERE/'imported_archive_manifest.json')
    expected = dict(start['files'])
    scopes = dict(protected_old_files=set(expected), historical_NPY=set(historic['files']),
                  frozen_sources=set(), supplied_files=set(), saved_outputs=set())
    def add(path, digest, scope):
        path = str(Path(path))
        assert path not in expected or expected[path] == digest, ('Conflicting expected SHA', path)
        expected[path] = digest
        scopes[scope].add(path)
    for path, digest in historic['files'].items():
        add(path, digest, 'historical_NPY')
    add(HERE/'protocol.json', freeze['protocol_sha256'], 'frozen_sources')
    for relative, digest in freeze['source_sha256'].items():
        add(ROOT/relative, digest, 'frozen_sources')
    assert archive_manifest['archive_sha256'] == config['supplied_archive_sha256']
    add(archive_manifest['archive_path'], archive_manifest['archive_sha256'], 'supplied_files')
    downloaded = Path(config['output'])/'downloaded'
    zip_member_checks = []
    with ZipFile(archive_manifest['archive_path']) as archive:
        assert set(archive.namelist()) == set(archive_manifest['files'])
        for name, desc in archive_manifest['files'].items():
            content = archive.read(name)
            passed = hashlib.sha256(content).hexdigest() == desc['sha256'] and len(content) == desc['size_bytes']
            zip_member_checks.append(dict(member=name, passed=passed))
            add(downloaded/name, desc['sha256'], 'supplied_files')
            relative = PurePosixPath(name).relative_to('permworld_peaks_probe')
            target = ROOT/'experiments/permworld_peaks_probe_20261109/vendor'/relative.name if relative.parts[0] == 'code' else HERE/'supplied_historical'/relative
            add(target, desc['sha256'], 'supplied_files')
    check, per_unit, output_files = audit_saved(config)
    for path, digest in output_files.items():
        add(path, digest, 'saved_outputs')
    results = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        jobs = [pool.submit(check_file, item) for item in sorted(expected.items())]
        for future in as_completed(jobs):
            result = future.result()
            results[result['path']] = result
            if len(results) % 50 == 0:
                print(json.dumps(dict(event='probe_endpoint_hash', files=len(results), total=len(expected))), flush=True)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    old_good = all(results[path]['passed'] for name in scopes if name != 'saved_outputs' for path in scopes[name])
    saved_good = all(results[path]['passed'] for path in scopes['saved_outputs'])
    protected = dict(status='passed' if old_good and head == start['git_head'] and all(r['passed'] for r in zip_member_checks) else 'failed',
        started_utc=started, completed_utc=utc(),
        scope='Endpoint SHA of explicitly named start files, historical NPY actually read, frozen protocol/sources, and supplied archive/vendor copies; not every file or continuous monitoring.',
        protected_start_file_count=len(scopes['protected_old_files']), historical_NPY_files=len(scopes['historical_NPY']),
        frozen_source_files=len(scopes['frozen_sources']), supplied_file_copies=len(scopes['supplied_files']),
        physical_checkpoint_files=45, new_source_feature_NPZ=64,
        git_head_start=start['git_head'], git_head_end=head, git_head_unchanged=head == start['git_head'],
        zip_member_checks=zip_member_checks,
        unique_files_hashed=len(results), unique_bytes_hashed=sum(r.get('size_bytes', 0) for r in results.values()),
        failed_files=[r for r in results.values() if not r['passed']],
        files=[results[path] for path in sorted(set().union(*(paths for name, paths in scopes.items() if name != 'saved_outputs')))],
        no_model_construction=True, no_neural_forward=True, no_probe_fitting=True, no_source_writes=True,
        elapsed_seconds=time.monotonic()-timer)
    write(HERE/'protection_end.json', protected)
    saved = dict(status='passed' if not check.failures and saved_good else 'failed',
        completed_utc=utc(),scope='Independent arithmetic on saved scalar CSVs and completion receipts only; no new probe/activation metric or inferential procedure.',
        statistical_units='Four new bundles, historical unit separately; one task pair, not 21 independent length replications.',
        expected_units=[u['unit'] for u in config['units']], per_unit=per_unit,
        arithmetic_tolerance=TOLERANCE, checks=check.total, maximum_arithmetic_difference=check.maximum_arithmetic_difference,
        failures=check.failures, saved_hash_files=len(scopes['saved_outputs']),
        saved_output_hash_checks=[results[path] for path in sorted(scopes['saved_outputs'])],
        historical_reproduction_sha256=sha(HERE/'historical_reproduction.json'),
        all_required_lengths_preserved=True, per_example_then_equal_length_weights=True,
        no_new_scientific_metrics=True, no_probe_fitting=True, no_forward=True)
    write(HERE/'saved_results_audit.json', saved)
    print(json.dumps(dict(protection_status=protected['status'],saved_results_status=saved['status'],
                         checks=check.total,arithmetic_failures=len(check.failures),
                         maximum_arithmetic_difference=check.maximum_arithmetic_difference,
                         unique_files_hashed=len(results),units=per_unit)),flush=True)
    if protected['status'] != 'passed' or saved['status'] != 'passed':
        raise SystemExit('Audit failed; inspect receipts without changing scientific outputs/tolerance.')


if __name__ == '__main__':
    main()
