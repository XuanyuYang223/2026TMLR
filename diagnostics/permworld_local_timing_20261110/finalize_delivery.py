"""Inventory only this completed timing follow-up, without running experiments."""
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
OUT = Path('/mnt/yangx-projects-f/20261110/permworld_local_timing')
PACKAGE = ROOT / 'experiments/permworld_local_timing_20261110'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    frozen = read(HERE / 'freeze.json')
    protocol = read(HERE / 'protocol.json')
    assert sha(HERE / 'protocol.json') == frozen['protocol_sha256']
    for path, expected in frozen['sources'].items():
        assert sha(ROOT / path) == expected, path
    summary = read(HERE / 'summary/completion.json')
    headlines = read(HERE / 'headlines/completion.json')
    for source, receipt_name in [('summarize_results.py', 'summary_freeze.json'),
                                 ('build_headline_tables.py', 'headline_freeze.json')]:
        assert sha(HERE / source) == read(HERE / receipt_name)['source_sha256']
    for name in ['input_audit.json', 'saved_results_audit.json', 'protection_end.json', 'delivery_audit.json']:
        assert read(HERE / name)['status'] == 'passed', name
    assert summary['protocol_sha256'] == headlines['protocol_sha256'] == frozen['protocol_sha256']
    assert summary['raw_rows'] == 2_923_200 and summary['pairs'] == 100_800
    assert headlines['main_cells'] == headlines['auxiliary_cells'] == headlines['positive_control_cells'] == 8
    audit = read(HERE / 'delivery_audit.json')
    fresh = {str(Path(row['path']).resolve()): row for row in audit['new_production_files']}
    assert fresh and all(row['passed'] for row in fresh.values())

    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(['git', 'status', '--porcelain=v1', '--untracked-files=no'], cwd=ROOT, text=True)
    assert head == protocol['git_head'] and not tracked
    try:
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                       '--format=csv,noheader'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    versions = {}
    for package in ['numpy', 'scipy', 'scikit-learn', 'torch', 'matplotlib', 'threadpoolctl', 'pytest']:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = 'provided by external vendor/dependency path or unavailable as metadata'
    environment = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
        package_versions=versions, nvidia_smi=gpu, git_head=head,
        production_flags_from_frozen_protocol=protocol['runtime'],
        production_parameter_dtype='float32', scoring_dtype='float64',
        external_python_paths=[str(ROOT / 'experiments/permworld_peaks_probe_20261109/vendor'),
                               '/mnt/yangx-projects-f/20261109/permworld_peaks_probe/dependencies'],
        note='Existing local environment, no dependency installation. Production flags are from the frozen run, not inferred from packaging-process defaults.')
    write(HERE / 'environment.json', environment)

    roots = [HERE, PACKAGE, OUT]
    files = [p for root in roots for p in root.rglob('*')
             if p.is_file() and '__pycache__' not in p.parts]
    files += [ROOT / 'tests/test_permworld_local_timing_replay.py']
    excluded = {HERE / 'delivery_manifest.json', HERE / 'completion.json'}
    inventory = []
    for path in sorted(set(files) - excluded):
        physical = str(path.resolve())
        if physical in fresh:
            record = fresh[physical]
            before = path.stat()
            assert before.st_size == record['size_bytes'], path
            digest = record['sha256']
            evidence = 'fresh_delivery_audit_hash; size rechecked during packaging'
        else:
            before = path.stat()
            digest = sha(path)
            after = path.stat()
            assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), path
            evidence = 'fresh packaging hash'
        inventory.append(dict(path=str(path), physical_path=physical, size_bytes=before.st_size,
                              sha256=digest, evidence=evidence,
                              symlink_target=os.readlink(path) if path.is_symlink() else None))
    for required in ['README.md', 'reproduce.sh', 'checkpoint_seed_manifest.csv', 'delivery_audit.json',
                     'headlines/timing_payload_comparison.png', 'headlines/timing_payload_comparison.pdf']:
        assert (HERE / required).is_file(), required
    manifest = dict(status='completed', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='New timing diagnostic/package/test and its production output only. Original files remain inputs referenced by separate named protection audits.',
        excluded_generated_self_referential_files=['delivery_manifest.json', 'completion.json'],
        excluded_runtime_files=['__pycache__'], protocol_sha256=frozen['protocol_sha256'],
        git_head=head, unchanged_tracked_status=tracked,
        new_production_hash_receipt_sha256=sha(HERE / 'delivery_audit.json'),
        protected_input_endpoint_receipt_sha256=sha(HERE / 'protection_end.json'),
        inventory=inventory)
    write(HERE / 'delivery_manifest.json', manifest)
    completion = dict(status='completed', created_utc=datetime.now(timezone.utc).isoformat(),
        formal_units=['unit02', 'unit03', 'unit04', 'unit05'], historical_reference='archdev001',
        formal_directed_cases=80_640, historical_directed_cases=20_160,
        saved_rows_per_case=29, total_production_rows=2_923_200,
        protocol_sha256=frozen['protocol_sha256'], delivery_manifest_sha256=sha(HERE / 'delivery_manifest.json'),
        reader_report='README.md', reproduction='reproduce.sh', checkpoint_seed_manifest='checkpoint_seed_manifest.csv',
        no_fits=True, no_new_base_training=True, no_DAS=True, no_cross_model=True,
        no_new_task_generalization=True, no_publication=True,
        original_git_head=head, tracked_files_unchanged=True,
        protection_scope='Named SHA-bound inputs at recorded endpoints, not a whole-disk or continuous-monitoring claim.',
        timing_result='Same1D current-token upper replay did not change any formal modal count; wholeL1 replacement transported the requested change.',
        stopped_after_authorized_within_model_experiments=True)
    write(HERE / 'completion.json', completion)
    print(json.dumps(dict(status=completion['status'], inventory_files=len(inventory),
                          production_rows=completion['total_production_rows'],
                          protocol_sha256=frozen['protocol_sha256']), indent=2))


if __name__ == '__main__':
    main()
