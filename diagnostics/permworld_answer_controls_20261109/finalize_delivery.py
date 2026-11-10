"""Finalize this completed diagnostic without changing previous research files."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    c=read(HERE/'protocol.json')
    for name in ['reproduction.json','independent_verification.json','learning_review.json','protection_end.json','activation_payload_end.json']:
        assert read(HERE/name)['status']=='passed',name
    assert read(HERE/'cka_completion.json')['rows']==86400
    assert read(HERE/'summary/summary_completion.json')['status']=='completed_saved_score_summary'
    assert read(HERE/'last_stage.json')['completed']
    protection=read(HERE/'protection_end.json')
    assert protection['protected_files_passed']==protection['protected_file_count']==1449
    report=HERE/'READER_REPORT.md'
    text=report.read_text().split('\n## Endpoint protection audit')[0]
    text+='\n## Endpoint protection audit\n\n'
    text+='All **1,449 explicitly named protected old files** match their start SHA-256, and Git HEAD is unchanged. All **960 activation payload files covering 640 logical test-feature records** match their recorded SHA-256. This is an endpoint audit of the named scope, not continuous monitoring or a claim that every repository file was checked. The prior manuscript, reports, protocols, and source artifacts were not edited by this diagnostic. See [protection_end.json](protection_end.json) and [activation_payload_end.json](activation_payload_end.json). No upload or publication was performed.\n'
    report.write_text(text)
    subprocess.run([sys.executable,str(HERE/'build_reader_html.py')],check=True,cwd=ROOT)
    external=Path(c['output'])
    per_length=sorted((external/'scores').glob('*/n*.csv'))
    aggregate=sorted((external/'scores').glob('*/aggregate.csv'))
    assert len(per_length)==840 and len(aggregate)==40
    external_tables={str(p):dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in per_length+aggregate}
    (HERE/'external_score_table_manifest.json').write_text(json.dumps(dict(per_length_files=840,aggregate_files=40,files=external_tables),indent=2)+'\n')
    excluded={'delivery_manifest.json','completion.json'}
    files=[p for p in HERE.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts and not p.name.endswith('.tmp')]
    files+=list((ROOT/'experiments/permworld_answer_controls_20261109').glob('*.py'))
    files+=list((ROOT/'tests').glob('test_permworld_answer_*.py'))
    manifest={str(p.relative_to(ROOT)):dict(sha256=sha(p),size_bytes=p.stat().st_size) for p in sorted(set(files))}
    (HERE/'delivery_manifest.json').write_text(json.dumps(dict(files=manifest,scope='Current diagnostic, helper scripts, result tables, figures, package sources and targeted tests; large source inputs have separate audited manifests.'),indent=2)+'\n')
    complete=dict(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
        units=5,new_units=4,trained_specialists=320,actual_saved_step0=10,missing_eligible_checkpoints=0,
        anchors_per_unit=10752,anchors_per_length=512,lengths=list(range(10,31)),
        aggregate_score_rows=86400,per_length_score_rows=1209600,per_length_tables=840,
        undefined_aggregate_scores=0,learning_summaries=320,learning_length_rows=6720,
        primary_task_pair_blocks=8,bootstrap_replicates=10000,bootstrap_seed=20261109,
        targeted_tests_passed=18,independent_numeric_checks=36,
        protocol_sha256=sha(HERE/'protocol.json'),delivery_manifest_sha256=sha(HERE/'delivery_manifest.json'),
        external_table_manifest_sha256=sha(HERE/'external_score_table_manifest.json'),
        no_new_training=True,no_probe_training=True,no_new_units=True,no_upload=True,
        stopped_after_current_diagnostic=True,prior_scientific_results_unmodified_within_named_audit=True,
        endpoint_audit_not_continuous_or_repository_wide=True)
    (HERE/'completion.json').write_text(json.dumps(complete,indent=2)+'\n')
    print(json.dumps(complete))


if __name__=='__main__':main()
