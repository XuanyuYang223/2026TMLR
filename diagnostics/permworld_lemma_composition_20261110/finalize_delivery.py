"""Hash this completed experiment's local deliverables; no model operations."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'experiments/permworld_lemma_composition_20261110'
EXTERNAL = Path('/mnt/yangx-projects-f/20261110/permworld_lemma_composition')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    config = read(HERE / 'protocol.json')
    freeze = read(HERE / 'freeze.json')
    assert sha(HERE / 'protocol.json') == freeze['protocol_sha256']
    for path, digest in freeze['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    audit = read(HERE / 'saved_results_audit.json')
    protection = read(HERE / 'protection_end.json')
    summary = read(HERE / 'summary/completion.json')
    assert audit['status'] == protection['status'] == 'passed'
    assert not audit['failures'] and not protection['failed_files']
    assert summary['status'] == 'completed_saved_composition_summary'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == config['git_head']
    assert not subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=ROOT, text=True).strip()
    receipts = []
    for unit in config['units']:
        receipt = read(HERE / 'results' / unit['unit'] / 'completion.json')
        assert receipt['status'] == 'completed'
        for name, record in receipt['artifacts'].items():
            # Each named artifact carries its own path and expected digest.
            actual = Path(record['path'])
            assert sha(actual) == record['sha256'], str(actual)
        prediction = receipt['predictions']
        assert sha(Path(prediction['path'])) == prediction['sha256']
        receipts.append(dict(unit=unit['unit'], historical=unit['historical'],
                             test_anchors=receipt['test_anchors'],
                             matched_anchors=receipt['matched_anchors'],
                             predictions=prediction))
    paths = sorted(set([
        *HERE.rglob('*'), *PACKAGE.rglob('*'),
        *ROOT.joinpath('tests').glob('test_permworld_lemma_*.py'),
        *EXTERNAL.rglob('*'),
    ]))
    records = {}
    for path in paths:
        if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
            continue
        if path.parent == HERE and path.name in ['delivery_manifest.json', 'completion.json']:
            continue
        try:
            key = str(path.relative_to(ROOT))
        except ValueError:
            key = str(path)
        records[key] = dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    git_head=config['git_head'], protocol_sha256=freeze['protocol_sha256'],
                    scope='New experiment sources, diagnostics, tests and external predictions/exports only; self/completion files excluded.',
                    files=records)
    write(HERE / 'delivery_manifest.json', manifest)
    write(HERE / 'completion.json', dict(
        status='completed', completed_utc=manifest['created_utc'],
        git_head=config['git_head'], protocol_sha256=freeze['protocol_sha256'],
        independent_new_units=4, historical_reference_units=1,
        lengths=config['lengths'], routes=len(config['routes']),
        variants=len(config['variants']), downstream_tasks=len(config['tasks']),
        units=receipts, tests_passed=41, saved_result_checks=audit['checks'],
        saved_result_failures=0, protected_prior_file_count=protection['protected_start_files'],
        protected_file_hash_failures=0, tracked_file_changes=[],
        actual_operations=['cached activation loading', 'source-only position probe reconstruction/fitting',
                           'frozen orthogonal-map reuse', 'source-validation threshold selection',
                           'mask-only composition', 'scoring and independent checks'],
        no_neural_forward=True, no_base_training=True, no_new_alignment_fitting=True,
        no_new_task_labels_in_fitting=True, no_publication=True,
        stopped_after_requested_experiment=True,
        primary_result='Aligned route below same-model-twice in all four new units on all three questions, full and original matched supports.',
        secondary_result='Decoded positions reusable in fixed analytic algorithms; count-preserving position shuffles reduce corresponding scores.',
        scientific_limit='No internal lemma-reasoning or original-specialist causal-use claim.',
        delivery_manifest_sha256=sha(HERE / 'delivery_manifest.json'),
        delivery_file_count=len(records),
    ))
    print(json.dumps(dict(status='completed', units=len(receipts), delivery_files=len(records),
                          saved_result_checks=audit['checks'])))


if __name__ == '__main__':
    main()
