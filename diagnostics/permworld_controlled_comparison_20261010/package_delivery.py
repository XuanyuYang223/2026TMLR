"""Hash finished deliverables; external array hashes reference verified index."""
from common import *
from datetime import datetime, timezone


def main():
    required = ['reader_report.html', 'comparison_manifest.csv', 'per_length_results.csv',
        'contrast_summary.csv', 'source_learning.csv', 'cross_arch_same_task.csv',
        'validation_audit.json', 'four_part_main.png', 'scope_status.csv', 'completion.json',
        'REPRODUCE.txt', 'reader_annotation_audit.json', 'metric_review/final_reader_audit.json']
    assert all((DIAG / name).is_file() for name in required)
    completion = js(DIAG / 'completion.json')
    assert completion['status'] == 'completed_and_stopped' and completion['stopped']
    assert js(DIAG / 'validation_audit.json')['status'] == 'passed'
    annotation = js(DIAG / 'reader_annotation_audit.json')
    assert sha(DIAG / 'reader_report.html') == annotation['annotated_report_sha256']
    export = js(DIAG / 'report_export_audit.json')
    assert export['input_sha256'][str(DIAG / 'completion.json')] == sha(DIAG / 'completion.json')
    assert export['input_sha256'][str(DIAG / 'validation_audit.json')] == sha(DIAG / 'validation_audit.json')
    assert sha(DIAG / 'reader_report_frozen_writer.html') == annotation['original_writer_report_sha256']
    files = {}
    for path in sorted(DIAG.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or path.name == 'artifact_manifest.json':
            continue
        files[str(path.relative_to(DIAG))] = dict(bytes=path.stat().st_size, sha256=sha(path))
    index_path = FROOT / 'layers/index.json'
    index = js(index_path)
    external = {str(index_path): dict(bytes=index_path.stat().st_size, sha256=sha(index_path))}
    for path in [FROOT / 'analysis.log', FROOT / 'extraction.log']:
        if path.exists():
            external[str(path)] = dict(bytes=path.stat().st_size, sha256=sha(path))
    json_write(DIAG / 'artifact_manifest.json', dict(status='passed',
        generated_utc=datetime.now(timezone.utc).isoformat(),
        files=files, required=required, external_files=external,
        large_cache_record_hashes=[{k:v for k,v in r.items() if k.endswith('_path') or k.endswith('_sha256')}
            for r in index['records']],
        large_array_actual_hashes_verified_by='validation_audit.json and extraction_review/index_reader_audit.json',
        source_code_frozen_sha256=js(DIAG / 'freeze.json')['source_sha256'],
        no_github_upload_performed=True, new_training=False, F17_changed=False,
        all_scientific_calculations_finished=True, stopped=True))
    print(dict(delivery_files=len(files), status='packaged_and_stopped'))


if __name__ == '__main__':
    main()
