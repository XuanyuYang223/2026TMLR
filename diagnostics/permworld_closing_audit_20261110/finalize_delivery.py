"""Seal the three completed closing checks and new writing files; no science run."""
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PACKAGE = ROOT / 'experiments/permworld_closing_audit_20261110'
OUT = Path('/mnt/yangx-projects-f/20261110/permworld_closing_audit')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def main():
    frozen = read(HERE / 'freeze.json')
    protocol = read(HERE / 'protocol.json')
    assert sha(HERE / 'protocol.json') == frozen['protocol_sha256']
    assert sha(HERE / 'freeze_protocol.py') == frozen['freeze_helper_sha256']
    for path, expected in frozen['spec_hashes'].items():
        assert sha(ROOT / path) == expected, path
    af = read(HERE / 'alignment/freeze.json')
    assert sha(PACKAGE / 'alignment.py') == af['source_sha256']
    assert sha(HERE / 'alignment/spec.json') == af['spec_sha256']
    assert sha(HERE / 'alignment/source_plan.json') == af['source_plan_sha256']
    ef = read(HERE / 'ed/code_freeze.json')
    assert sha(HERE / 'ed/analyze.py') == ef['code_sha256']
    assert ef['global_protocol_sha256'] == frozen['protocol_sha256']
    gf = read(HERE / 'gradient/run_freeze.json')
    for path, expected in gf['sources'].items():
        assert sha(path) == expected, path
    assert read(HERE / 'alignment/completion.json')['status'] == 'completed'
    assert read(HERE / 'ed/completion.json')['status'] == 'completed'
    assert read(HERE / 'gradient/summary_completion.json')['status'] == 'completed'
    receipts = {}
    for name in ['alignment/final_audit.json', 'ed/independent_verification.json',
                 'gradient/independent_verification.json', 'gradient/saved_gradient_audit.json']:
        path = HERE / name
        data = read(path)
        assert data['status'] == 'passed', name
        receipts[name] = dict(path=str(path), sha256=sha(path), status=data['status'])
    unit_receipts = []
    for unit in ['archdev001', 'unit02', 'unit03', 'unit04', 'unit05']:
        path = HERE / 'gradient/results' / unit / 'completion.json'
        data = read(path)
        assert data['status'] == 'completed' and data['pairs'] == 20_160
        assert data['finite_difference_all_passed'] and data['finite_difference_rows'] == 48
        assert read(path.with_name('input_bindings.json'))['status'] == 'passed'
        assert all(task['parameter_sha256_before'] == task['parameter_sha256_after'] for task in data['task_reports'])
        unit_receipts.append(dict(unit=unit, path=str(path), sha256=sha(path), pairs=data['pairs'],
                                  finite_difference_rows=data['finite_difference_rows'],
                                  elapsed_seconds=data['elapsed_seconds']))
    assert sum(row['pairs'] for row in unit_receipts) == 100_800
    writer = read(HERE / 'writing/sources.json')
    assert sha(HERE / 'build_writing.py') == writer['code_sha256']
    for path, expected in writer['source_sha256'].items():
        assert sha(path) == expected, path

    start = read(HERE / 'protection_start.json')
    assert sha(HERE / 'protection_start.json') == frozen['protection_start_sha256']
    protection = []
    for path, expected in start['files'].items():
        actual = sha(path)
        protection.append(dict(path=path, expected_sha256=expected, end_sha256=actual, unchanged=actual == expected))
    head = git('rev-parse', 'HEAD')
    tracked = git('status', '--porcelain=v1', '--untracked-files=no')
    assert head == protocol['git_head'] and not tracked
    assert all(row['unchanged'] for row in protection)
    write(HERE / 'protection_end.json', dict(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
        scope=start['scope'], files=protection, named_files=len(protection), git_head=head,
        tracked_status=tracked, stage_consumed_payload_audit_receipts=receipts,
        note='Each stage independently verifies its exact consumed cache/checkpoint/probe/map inputs at its recorded endpoints. This root check rehashes only the explicit metadata/source/manuscript scope, not every disk file or continuous history.'))
    versions = {}
    for package in ['torch', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pytest']:
        try: versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError: versions[package] = 'external vendor/dependency path, or unavailable metadata'
    write(HERE / 'environment.json', dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), package_versions=versions,
        production_dtype='FP32 gradient graph and unchanged cached FP32 endpoints; FP64 scalar arithmetic',
        finite_difference_dtype='FP64 shadow of exact saved weights, derivative-wiring validation only',
        numeric_flags='Threads2, TF32off, all model modules eval; CuDNNoff only for differentiable replay/backward',
        dependencies_installed=False, external_pythonpath=[
            str(ROOT / 'experiments/permworld_peaks_probe_20261109/vendor'),
            '/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies']))

    # Fresh endpoint hashing of all new delivery payloads, deduplicating aliases.
    paths = [p for directory in [HERE, PACKAGE, OUT] for p in directory.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    paths += [ROOT / 'tests/test_permworld_closing_gradient.py']
    excluded = {HERE / 'completion.json', HERE / 'delivery_manifest.json'}
    physical_hashes = {}; inventory = []
    for path in sorted(set(paths) - excluded):
        physical = str(path.resolve())
        if physical not in physical_hashes:
            before = path.stat(); digest = sha(path); after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
            physical_hashes[physical] = dict(sha256=digest, size_bytes=after.st_size)
        inventory.append(dict(path=str(path), physical_path=physical,
                              symlink_target=os.readlink(path) if path.is_symlink() else None,
                              **physical_hashes[physical]))
    manifest = dict(status='completed', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='New closing-audit package, diagnostic, test and external payloads only; root manifest/completion self-excluded, __pycache__ excluded.',
        protocol_sha256=frozen['protocol_sha256'], git_head=head, files=inventory,
        unique_physical_files=len(physical_hashes), unique_physical_bytes=sum(v['size_bytes'] for v in physical_hashes.values()),
        input_and_validation_audit_receipts=receipts,
        root_protection_receipt_sha256=sha(HERE / 'protection_end.json'))
    write(HERE / 'delivery_manifest.json', manifest)
    completion = dict(status='completed', completed_utc=datetime.now(timezone.utc).isoformat(),
        checks_completed=['PV trained-source/model-target specificity', 'Fixed1Dedit early-path geometry and gradients', 'E/D control-specific encoding recheck'],
        formal_units=['unit02', 'unit03', 'unit04', 'unit05'], historical_reference='archdev001',
        protocol_sha256=frozen['protocol_sha256'], manifest_sha256=sha(HERE / 'delivery_manifest.json'),
        exploratory_post_results=True, retrospectively_preregistered=False,
        formal_gradient_pairs=80_640, historical_gradient_pairs=20_160,
        finite_difference_cases=120, finite_difference_directional_checks=240,
        gradient_unit_receipts=unit_receipts, missing_materials=[],
        new_control_maps=10, new_probe_fits=0, new_base_model_training=0,
        new_worlds=0, new_directions_or_doses=0, no_DAS=True, no_cross_model_exchange=True,
        original_manuscript_overwritten=False, no_DMLR_direction_change=True,
        no_upload_or_publication=True, next_experiment_started=False,
        git_head_unchanged=True, tracked_files_unchanged=True, stopped=True,
        writing_files=['writing/claim_evidence_scope.csv', 'writing/claim_evidence_scope.md',
                       'writing/tmlr_structure.md', 'writing/results_draft.md', 'writing/results_draft.tex'])
    write(HERE / 'completion.json', completion)
    print(json.dumps(dict(status=completion['status'], named_old_protected_files=len(protection),
                          new_physical_files=len(physical_hashes),
                          new_payload_bytes=manifest['unique_physical_bytes'],
                          formal_gradient_pairs=completion['formal_gradient_pairs']), indent=2))


if __name__ == '__main__':
    main()
