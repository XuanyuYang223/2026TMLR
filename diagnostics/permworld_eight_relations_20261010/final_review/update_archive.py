"""Append a logical archive only after the complete fixed-scope provenance audit.

No model, array or scientific metric is read. No files are deleted or moved.
The original archive bytes/rows and cleanup candidates remain intact. Large
scientific file identities come from the already completed independent audit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import tempfile

WORKSPACE = Path('/home/yangx/ICML')
DIAG = WORKSPACE / 'diagnostics/permworld_eight_relations_20261010'
REVIEW = DIAG / 'final_review'
GRAPH_DIAG = WORKSPACE / 'diagnostics/graphworld_pilot_20261010'
PERM_ROOT = Path('/mnt/yangx-projects-f/20261010/permworld_eight_relations')
GRAPH_ROOT = Path('/mnt/yangx-projects-f/20261010/graphworld_pilot')
ROOTS = [WORKSPACE/'experiments/permworld_eight_relations_20261010',
         WORKSPACE/'experiments/graphworld_pilot_20261010', DIAG, GRAPH_DIAG,
         PERM_ROOT, GRAPH_ROOT]
FREEZE = REVIEW / 'archive_update_freeze.json'
BASE = REVIEW / 'archive_base_snapshot.csv'
ARCHIVE = WORKSPACE / 'archive_index.csv'
CLEANUP = WORKSPACE / 'cleanup_candidates.csv'
CAT1, CAT2 = '1_current_PermWorld_graph_research', '2_legacy_dependencies_of_current_research'
OLD_GROUPS = {'reused_weights', 'initial_weights', 'saved_start_RNG',
              'original_cache', 'perm_cache_reused_physical'}
STATE_NAMES = {'queue_state.json', 'orchestration_state.json', 'scope_bridge_state.json'}
SELF_METADATA = ['scope_artifact_manifest.json','current_entry_manifest.csv',
                 'archive_update_audit.json','archive_update_pending.json']


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_rows(data):
    reader = csv.DictReader(io.StringIO(data.decode('utf-8')))
    return list(reader), reader.fieldnames


def csv_bytes(rows, fields, header=True):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode('utf-8')


def under(path, roots):
    return any(path.is_relative_to(root.resolve()) for root in roots)


def phase_names(orchestration):
    return {p[0] for p in orchestration['perm_stages'] + orchestration['graph_stages']} | {
        'perm_result_readout', 'graph_result_readout'}


def ready(base=DIAG, review=REVIEW):
    """Small metadata only. No directory walking is permitted before readiness."""
    needed = [review/'final_audit.json', review/'physical_file_manifest.json',
              base/'phase_timing.json', base/'scope_bridge_state.json',
              base/'orchestration_state.json']
    missing = [str(p) for p in needed if not p.is_file()]
    problems = []
    if not missing:
        audit = read(needed[0]); ledger = read(needed[1]); phases = read(needed[2])['phases']
        if audit.get('status') != 'passed' or not audit.get('all_fixed_stages_complete'):
            problems.append('Final provenance audit has not passed the entire fixed scope.')
        if ledger.get('status') != 'passed':
            problems.append('Physical identity ledger is not passed.')
        expected = phase_names(read(base/'orchestration_freeze.json'))
        if (len(phases) != len(expected) or {p['phase'] for p in phases} != expected
                or any(p.get('exit_code') != 0 or not p.get('finish_utc') for p in phases)):
            problems.append('The complete fixed fourteen phases have not finalized successfully.')
        if read(needed[3]).get('status') != 'scientific_scope_completed_report_review_pending':
            problems.append('Fixed scope bridge has not finalized.')
        if read(needed[4]).get('status') != 'all_fixed_scientific_stages_completed':
            problems.append('Fixed scientific orchestration has not finalized.')
    return dict(status='ready' if not missing and not problems else 'pending',
                missing=missing, problems=problems, metadata_only=True,
                directory_walk=False, models_or_arrays_opened=False)


def freeze():
    require(not FREEZE.exists(), 'Archive updater is already frozen; do not silently rebind.')
    original = ARCHIVE.read_bytes(); rows, _ = csv_rows(original)
    require(len(rows) == 699, 'Expected the original 699-row logical archive.')
    require(original.endswith(b'\n'), 'Original archive must end in a newline for byte-preserving append.')
    cleanup, _ = csv_rows(CLEANUP.read_bytes())
    require(len(cleanup) == 13, 'Expected exactly thirteen original cleanup candidates.')
    require(all(r['deleted'] == r['moved'] == 'False' for r in cleanup), 'Cleanup contains an executed action.')
    fixture_evidence=read(REVIEW/'archive_update_fixture.json')
    require(fixture_evidence['status']=='passed' and fixture_evidence['updater_sha256']==sha(__file__),
            'The current source must pass its fixture before freezing.')
    BASE.write_bytes(original)
    bindings = {str(p):sha(p) for p in [REVIEW/'audit_freeze.json', DIAG/'protocol.json',
                GRAPH_DIAG/'protocol.json', DIAG/'orchestration_freeze.json',
                DIAG/'scope_bridge_freeze.json',
                DIAG/'inventory/archive_dependency_edges.json']}
    value = dict(registered_utc=datetime.now(timezone.utc).isoformat(), source_path=str(Path(__file__).resolve()),
                 source_sha256=sha(__file__), baseline_archive_sha256=sha(BASE),
                 baseline_rows=len(rows), cleanup_sha256=sha(CLEANUP), cleanup_rows=len(cleanup),
                 bindings=bindings, scope_roots=[str(p) for p in ROOTS],
                 policy=dict(original_rows_and_raw_byte_prefix_preserved=True,
                             current_research_body_untouched=True, cleanup_unchanged=True,
                             deleted_files=0, moved_files=0, scientific_metrics=0,
                             no_large_file_rehash=True, only_after_all_phases_and_final_audit=True,
                             mutable_annotations_never_exempt_audited_scientific_files=True))
    write(FREEZE, value)
    return value


def verify():
    frozen = read(FREEZE)
    require(sha(__file__) == frozen['source_sha256'], 'Archive updater source changed after freeze.')
    require(sha(BASE) == frozen['baseline_archive_sha256'], 'Original archive snapshot changed.')
    require(sha(CLEANUP) == frozen['cleanup_sha256'], 'Cleanup candidates changed; updater refuses to overwrite them.')
    for path, expected in frozen['bindings'].items():
        require(sha(path) == expected, 'Frozen archive input changed: ' + path)
    original = BASE.read_bytes(); current = ARCHIVE.read_bytes()
    require(current.startswith(original), 'Original archive bytes/rows were edited; refusing to replace them.')
    if current != original:
        prior = read(REVIEW/'archive_update_audit.json')
        require(prior['status'] == 'passed' and sha(ARCHIVE) == prior['updated_archive_sha256'],
                'Archive has an unrecognized post-baseline append.')
    return frozen, original


def dependency_paths(ledger, perm_index, roots):
    """Evidence outranks a new directory name, especially for old cache aliases."""
    deps = {}
    for rec in ledger['records']:
        p = Path(rec['path']).resolve(); groups = set(rec['groups'])
        # Protection is preservation evidence, not a current scientific use.
        if groups == {'protected_old'}:
            continue
        if groups & OLD_GROUPS or not under(p, roots):
            deps[str(p)] = 'Completed provenance ledger roles: ' + ', '.join(sorted(groups))
    for rec in perm_index['records']:
        if rec.get('reused_original'):
            for desc in rec['arrays'].values():
                deps[str(Path(desc['path']).resolve())] = 'Exact original source-cache reuse/bitwise conversion; preserve dependency even inside new F root.'
        if rec.get('initial_hidden_alias_source_arrays'):
            for desc in rec['initial_hidden_alias_source_arrays'].values():
                deps[str(Path(desc['path']).resolve())] = 'Saved initial hidden alias target; referenced old initialization geometry, not an independent new cache.'
    return deps


def classify(paths, ledger, deps, roots, closed_log_paths=()):
    references = {str(Path(r['path']).resolve()):r for r in ledger['records']}
    result = []
    for name in sorted(paths):
        p = Path(name); ref = references.get(name)
        old = name in deps
        require(old or under(p, roots), 'Unbound old project cannot be reclassified: '+name)
        category = CAT2 if old else CAT1
        observational = (not old and ref is None and under(p, roots)
                         and (p.suffix=='.log' or p.name in STATE_NAMES))
        reason = deps.get(name, 'Existing new authorized scope artifact; no directory-based removal or historical-project reclassification.')
        info = p.stat()
        if ref is not None and 'bytes' in ref:
            require(info.st_size==ref['bytes'], 'Audited artifact size changed after final provenance audit: '+name)
        closed_log_sha=None
        if observational and name in closed_log_paths and p.suffix=='.log' and info.st_size<=8*1024*1024:
            # Fixed source/phase completion makes these bounded small logs closed.
            # This supplementary observation is not a new scientific identity.
            closed_log_sha=sha(p); after=p.stat()
            require((info.st_size,info.st_mtime_ns,info.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
                    'Closed log changed during supplementary hash: '+name)
        result.append(dict(path=name, category=category, dependency_evidence=reason,
                    provenance_groups=ref['groups'] if ref else [],
                    audited_sha256=ref['sha256'] if ref else None,
                    provenance_SHA_verified=ref is not None,
                    bytes=info.st_size, observation_mtime_ns=info.st_mtime_ns,
                    mutable_observational_only=False,
                    observational_log_or_state=observational,
                    closed_small_log_observation_sha256=closed_log_sha,
                    immutability_note=('New log/state observation only; no mutation exemption or scientific identity.' if observational else
                                       'Preserve artifact; no mutation authorization. Stat-only entries do not claim SHA verification.'),
                    existing_original_path=True, delete=False, move=False))
    return result


def append_rows(original, records):
    rows, fields = csv_rows(original)
    known = {(r['path'],r['category']) for r in rows}
    added = []
    for r in records:
        if (r['path'], r['category']) in known:
            continue
        known.add((r['path'],r['category']))
        added.append(dict(path=r['path'], category=r['category'],
                 use='New artifacts of the current fixed experiment' if r['category']==CAT1 else 'Exact legacy dependency/alias target required for current reproduction',
                 dependency_evidence=r['dependency_evidence'], retain_reason=r['dependency_evidence'],
                 rebuildable='Retain provenance and protocols; actual weights, RNG states, and unique data must not be assumed reconstructible',
                 suggested_action='Keep original path; logical archival only', exists='True',
                 protected_action='Do not delete, move, or modify earlier documents/published artifacts'))
    return original + csv_bytes(added, fields, header=False), added


def run():
    _, original = verify()
    gate = ready()
    if gate['status'] != 'ready':
        write(REVIEW/'archive_update_pending.json', gate)
        print(json.dumps(gate)); return 2
    audit = read(REVIEW/'final_audit.json'); ledger = read(REVIEW/'physical_file_manifest.json')
    audit_freeze = read(REVIEW/'audit_freeze.json')
    require(audit['auditor_sha256'] == next(iter(audit_freeze['source_sha256'].values())), 'Final auditor identity differs.')
    require(audit['audit_freeze_sha256'] == sha(REVIEW/'audit_freeze.json'), 'Final provenance freeze differs.')
    require(len(ledger['records']) == audit['physical_storage']['audited_references']['unique_hashed_physical_files'], 'Ledger/count differs from final audit.')
    phase = next(r for r in ledger['records'] if r['path']==str(DIAG/'phase_timing.json'))
    require(sha(DIAG/'phase_timing.json') == phase['sha256'], 'Phase ledger changed after final provenance audit.')
    perm_index = read(PERM_ROOT/'evaluation/index.json')
    require(perm_index['status']=='completed' and len(perm_index['records'])==256, 'Perm index is not complete.')
    deps = dependency_paths(ledger, perm_index, ROOTS)
    paths = set(deps)
    for root in ROOTS:
        require(root.is_dir(), 'Missing finalized scope directory: '+str(root))
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in {'.pyc','.pyo'}:
                paths.add(str(p.resolve()))
    paths.update(str(Path(r['path']).resolve()) for r in ledger['records']
                 if under(Path(r['path']).resolve(),ROOTS) or str(Path(r['path']).resolve()) in deps)
    # Output destinations are known, but future files are never invented as complete.
    paths.difference_update(str(REVIEW/n) for n in SELF_METADATA)
    # Root final completion is written afterwards and binds this manifest; a
    # self-referential completion SHA must never be invented here.
    paths.discard(str(DIAG/'completion.json'))
    closed_logs={str(Path(p['log_path']).resolve()) for p in audit['timing']['instrumented_phase_timings']}
    closed_logs.update(str((PERM_ROOT/'logs'/(j['run_id']+'.log')).resolve()) for j in read(DIAG/'protocol.json')['training_manifest'])
    bridge=read(DIAG/'scope_bridge_state.json')
    closed_logs.add(str(Path(bridge['graph_driver']['log_path']).resolve()))
    records = classify(paths, ledger, deps, ROOTS, closed_logs)
    # Re-entrant logical updates preserve earlier append-only rows as well.
    updated, added = append_rows(ARCHIVE.read_bytes(), records)
    manifest = dict(status='passed', original_paths_preserved=True, records=records,
                    scope_roots=[str(p) for p in ROOTS],
                    provenance_ledger_path=str(REVIEW/'physical_file_manifest.json'),
                    provenance_ledger_sha256=sha(REVIEW/'physical_file_manifest.json'),
                    final_audit_sha256=sha(REVIEW/'final_audit.json'), unique_resolved_paths=True,
                    sha_verification_scope='SHA identities inherited from the passed final audit; this updater observes file stat and never rehashes scientific files.',
                    self_output_destinations_not_inferred_as_preexisting_complete_artifacts=SELF_METADATA+[str(DIAG/'completion.json')],
                    root_final_completion_reverse_binds_manifest=True,
                    large_files_rehashed=False, deleted_files=0, moved_files=0,
                    mutable_policy='No mutable exemptions after finalized phases. New unbound log/state observations are metadata only; bounded closed training/pipeline logs may receive supplementary SHA without becoming scientific identity.',
                    independent_old_projects_and_original_classification_rows_unchanged=True)
    write(REVIEW/'scope_artifact_manifest.json', manifest)
    fields=['path','category','dependency_evidence','provenance_SHA_verified','audited_sha256','bytes','mutable_observational_only','delete','move']
    auxiliary=[{k:r[k] for k in fields} for r in records]
    (REVIEW/'current_entry_manifest.csv').write_bytes(csv_bytes(auxiliary,fields))
    temporary = WORKSPACE/'archive_index.csv.logical-update.tmp'
    temporary.write_bytes(updated); temporary.replace(ARCHIVE)
    require(ARCHIVE.read_bytes().startswith(original), 'Original byte prefix not preserved.')
    old_rows,_=csv_rows(original); new_rows,_=csv_rows(updated)
    require(new_rows[:len(old_rows)]==old_rows, 'Original archive row content/order changed.')
    require(sha(CLEANUP)==read(FREEZE)['cleanup_sha256'], 'Cleanup changed during update.')
    write(REVIEW/'archive_update_audit.json', dict(status='passed', registered_utc=datetime.now(timezone.utc).isoformat(),
          updater_sha256=sha(__file__), updater_freeze_sha256=sha(FREEZE), original_rows=len(old_rows),
          added_rows=len(added), updated_rows=len(new_rows), original_byte_prefix_preserved=True,
          original_rows_order_and_protection_reasons_preserved=True, updated_archive_sha256=sha(ARCHIVE),
          category_counts=dict(Counter(r['category'] for r in new_rows)), cleanup_rows=13,
          cleanup_sha256=sha(CLEANUP), cleanup_unchanged=True, CURRENT_RESEARCH_body_written=False,
          final_audit_sha256=sha(REVIEW/'final_audit.json'), scope_manifest_sha256=sha(REVIEW/'scope_artifact_manifest.json'),
          append_role_rows_only_no_existing_rewrites=True, duplicate_original_rows_preserved=True, deleted_files=0,moved_files=0,
          training=False,GPU_started=False,models_opened=False,scientific_metrics_recomputed=False,
          small_closed_log_hash_scans=sum(r['closed_small_log_observation_sha256'] is not None for r in records),
          large_artifact_hash_scans=0))
    print(json.dumps(dict(status='passed', original_rows=len(old_rows), added_rows=len(added), updated_rows=len(new_rows))))
    return 0


def fixture():
    checks = {}
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); new=root/'new'; old=root/'old'; new.mkdir(); old.mkdir()
        alias=new/'alias.npy'; alias.write_bytes(b'old cache bytes')
        init=new/'copied_step0.pt'; init.write_bytes(b'old initial weights')
        weight=new/'new_checkpoint.pt'; weight.write_bytes(b'new weights')
        log=new/'run.log'; log.write_text('mutable observation')
        (new/'logs').mkdir(); closedlog=new/'logs/run.log'; closedlog.write_text('completed training log')
        boundlog=new/'protected.log'; boundlog.write_text('bound log')
        dependency=old/'common.py'; dependency.write_text('old imported source')
        independent=old/'f17.txt'; independent.write_text('independent publication')
        ledger=dict(records=[dict(path=str(alias),groups=['perm_cache_new_physical'],sha256='a'),
                             dict(path=str(init),groups=['initial_weights'],sha256='b'),
                             dict(path=str(weight),groups=['new_formal_weights'],sha256='c'),
                             dict(path=str(boundlog),groups=['phase_timing'],sha256='d'),
                             dict(path=str(dependency),groups=['frozen_source'],sha256='e'),
                             dict(path=str(independent),groups=['protected_old'],sha256='f')])
        index=dict(records=[dict(reused_original=True,arrays={'prefix_final':dict(path=str(alias))})])
        deps=dependency_paths(ledger,index,[new]); records=classify({str(p) for p in [alias,init,weight,log,closedlog,boundlog,dependency]},ledger,deps,[new],{str(closedlog)})
        by={r['path']:r for r in records}
        checks['new_root_old_alias_remains_dependency2']=by[str(alias)]['category']==CAT2
        checks['copied_actual_step0_remains_dependency2']=by[str(init)]['category']==CAT2
        checks['old_import_dependency2']=by[str(dependency)]['category']==CAT2
        checks['protection_alone_does_not_turn_F17_into_current_dependency']=str(independent) not in deps
        checks['new_weight_current1_immutable']=by[str(weight)]['category']==CAT1 and not by[str(weight)]['mutable_observational_only']
        checks['unbound_new_log_observation_never_mutable_exemption']=by[str(log)]['observational_log_or_state'] and not any(r['mutable_observational_only'] for r in records)
        checks['closed_small_log_SHA_is_supplementary_only']=by[str(closedlog)]['closed_small_log_observation_sha256']==sha(closedlog) and not by[str(closedlog)]['provenance_SHA_verified']
        wrong=dict(records=[dict(path=str(weight),groups=['new_formal_weights'],sha256='c',bytes=999)])
        try:classify({str(weight)},wrong,{},[new])
        except ValueError:checks['postaudit_file_size_change_refuses_without_large_rehash']=True
        else:checks['postaudit_file_size_change_refuses_without_large_rehash']=False
        fields=['path','category','use','dependency_evidence','retain_reason','rebuildable','suggested_action','exists','protected_action']
        orig=[{f:'' for f in fields} for _ in range(3)]
        orig[0].update(path=str(independent),category='3_independent_legacy_research',retain_reason='keep F17 byte by byte')
        orig[1].update(path=str(new),category=CAT1,retain_reason='original duplicate')
        orig[2]=dict(orig[1])
        original=csv_bytes(orig,fields); updated,added=append_rows(original,records)
        after,_=csv_rows(updated)
        checks['original_raw_prefix_and_duplicate_rows_preserved']=updated.startswith(original) and after[:3]==orig
        checks['independent_project3_protection_preserved']=after[0]==orig[0]
        twice,_=append_rows(updated,records)
        checks['existing_role_entries_not_duplicated']=twice==updated
        checks['incomplete_gate_has_no_directory_walk']=ready(root,root)['status']=='pending'
        base=root/'diagnostic'; review=base/'final_review'; review.mkdir(parents=True)
        orchestration=dict(perm_stages=[[f'p{i}',[]] for i in range(5)],graph_stages=[[f'g{i}',[]] for i in range(7)])
        write(base/'orchestration_freeze.json',orchestration)
        write(review/'final_audit.json',dict(status='passed',all_fixed_stages_complete=True))
        write(review/'physical_file_manifest.json',dict(status='passed',records=[]))
        phases=[dict(phase=p,exit_code=0,finish_utc='2026-10-10T00:00:00+00:00') for p in sorted(phase_names(orchestration))]
        write(base/'phase_timing.json',dict(phases=phases))
        write(base/'scope_bridge_state.json',dict(status='scientific_scope_completed_report_review_pending'))
        write(base/'orchestration_state.json',dict(status='all_fixed_scientific_stages_completed'))
        checks['complete_fourteen_phases_and_final_audit_accept_gate']=ready(base,review)['status']=='ready'
        write(base/'phase_timing.json',dict(phases=phases[:-1]))
        checks['missing_readout_phase_refuses_gate']=ready(base,review)['status']=='pending'
        write(base/'phase_timing.json',dict(phases=phases))
        write(review/'final_audit.json',dict(status='pending',all_fixed_stages_complete=False))
        checks['unpassed_final_audit_refuses_gate']=ready(base,review)['status']=='pending'
        checks['no_deletions_or_moves']=all(not r['delete'] and not r['move'] for r in records) and independent.exists()
    require(all(checks.values()), 'Archive fixture failed: '+str(checks))
    result=dict(status='passed', updater_sha256=sha(__file__), checks=checks,
                real_archive_modified=False, real_cleanup_modified=False,
                real_models_or_arrays_opened=False, large_artifact_hash_scans=0,GPU_started=False,training=False)
    write(REVIEW/'archive_update_fixture.json',result); print(json.dumps(result)); return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    for mode in ('freeze','fixture','pending','run'):
        group.add_argument('--'+mode,action='store_true')
    args=parser.parse_args()
    if args.fixture:return fixture()
    if args.freeze:print(json.dumps(freeze()));return 0
    if args.pending:
        verify(); result=ready(); write(REVIEW/'archive_update_pending.json',result); print(json.dumps(result));return 0
    return run()


if __name__=='__main__':
    raise SystemExit(main())
