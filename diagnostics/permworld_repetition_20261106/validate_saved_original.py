"""Validate the new analysis on fixed slices of original saved activations.

This validation constructs no model and does not import Torch, run a forward
pass, sample inputs, fit a probe, or train. Existing activation arrays are
adapted without changing their values, then passed through the unchanged new
analysis pipeline. The four cases and tolerance are fixed in this source.
"""
from __future__ import annotations

import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
from experiments.permworld_repetition_20261106 import analysis

OUTPUT = Path('/mnt/yangx-projects-f/20261106/permworld_repetition/original_metric_verification')
TOLERANCE = 1e-8
CASES = (
    ('transformer', 'one_line', 'descent_inverse', 10),
    ('transformer', 'canonical_cycle', 'fixed_complement', 20),
    ('gru', 'one_line', 'decomposition_complement', 30),
    ('gru', 'canonical_cycle', 'exceedance_inverse', 10),
)
FIELDS = ('prefix_final', 'output_prob', 'prediction')


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def numbers_equal(expected, observed):
    a, b = analysis.number(expected), analysis.number(observed)
    if a is None or b is None:
        return (a is None and b is None), None, a is None and b is None
    error = abs(a-b)
    return error <= TOLERANCE, error, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    old_config_path = WORKSPACE/'diagnostics/permworld_eight_relations_20261010/protocol.json'
    new_config_path = Path(__file__).with_name('protocol.json')
    a_path = WORKSPACE/'diagnostics/permworld_eight_relations_20261010/per_length_results.csv'
    b_path = WORKSPACE/'diagnostics/representation_pairing_review_20261011/pair_per_length.csv'
    old, new = read(old_config_path), read(new_config_path)
    index_path = Path(old['evaluation_index_path'])
    index = read(index_path)
    data_path = Path(old['source_data_path'])
    assert index['status'] == 'completed'
    assert index['source_data_sha256'] == old['source_data_sha256']
    assert index['protocol_sha256'] == sha(old_config_path)
    records = {(r['architecture'], r['encoding'], r['task'], r['condition'], r['split']):r
               for r in index['records']}
    assert len(records) == 256
    tasks = old['label_task_names']
    assert len(tasks) == 16 and tasks == new['tasks']
    source_paths = {old_config_path, new_config_path, a_path, b_path, index_path,
                    data_path, Path(analysis.__file__), analysis.METRIC_PATH,
                    Path(__file__).with_name('freeze.json')}
    expected_array_hashes = {}
    for arch, enc, relation_id, n in CASES:
        for task in tasks:
            for phase in analysis.PHASES:
                for split in ('validation', 'test'):
                    record = records[arch, enc, task, phase, split]
                    assert record['source_data_sha256'] == old['source_data_sha256']
                    assert record['protocol_sha256'] == sha(old_config_path)
                    for field in FIELDS:
                        meta = record['arrays'][field]
                        path = Path(meta['path'])
                        source_paths.add(path)
                        if path in expected_array_hashes:
                            assert expected_array_hashes[path] == meta['sha256']
                        expected_array_hashes[path] = meta['sha256']
    start_hashes = {str(path):sha(path) for path in sorted(source_paths)}
    assert start_hashes[str(data_path)] == old['source_data_sha256']
    for path, expected in expected_array_hashes.items():
        assert start_hashes[str(path)] == expected, ('Original cache bytes changed', str(path))
    freeze = read(Path(__file__).with_name('freeze.json'))
    for relative, expected in freeze['source_sha256'].items():
        if (WORKSPACE/relative).resolve() == Path(analysis.__file__).resolve():
            assert sha(analysis.__file__) == expected
    write(out/'protection_start.json', dict(utc=utc(), files=start_hashes,
          scope='Only the named source data, indices, tables, protocols, metric and activation files; no checkpoint or global-file claim.'))
    protocol = dict(utc=utc(), exploratory_engineering_validation=True,
                    cases=[dict(architecture=a, encoding=e, relation=r, length=n) for a,e,r,n in CASES],
                    tolerance=TOLERANCE, metrics=list(analysis.METRICS),
                    controls=list(analysis.CONTROLS), branches=['hidden','output_prob'],
                    rows='Original full 256 validation and 512 test anchors at each fixed length; no answer-based subsampling.',
                    training_labels='All 4096 training anchors of the same fixed length, for the unchanged length-mode diagnostic only.',
                    index_shift='Local row IDs remove the constant fixed-length offset; stable ID ordering and duplicate identities are preserved.',
                    no_model_construction=True, no_forward=True, no_training=True,
                    analysis_source_sha256=sha(analysis.__file__),script_sha256=sha(__file__),
                    git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKSPACE,text=True).strip(),
                    comparison_fields=list(analysis.COMPONENTS + analysis.EFFECTS))
    write(out/'validation_protocol.json', protocol)
    protocol_hash = sha(out/'validation_protocol.json')
    a_original, b_original = rows(a_path), rows(b_path)
    all_checks, case_results, adapters = [], [], []
    with np.load(data_path, allow_pickle=False) as archive:
        full_data = {k:archive[k].copy() for k in archive.files
                     if k.endswith(('_permutations','_lengths','_labels'))}
    for arch, enc, relation_id, n in CASES:
        case_name = f'{arch}_{enc}_{relation_id}_n{n}'
        root = out/case_name
        root.mkdir(parents=True,exist_ok=True)
        selectors = {split:np.flatnonzero(full_data[split+'_lengths'] == n)
                     for split in ('train','validation','test')}
        assert [len(selectors[s]) for s in ('train','validation','test')] == [4096,256,512]
        sliced = {key:value[selectors[key.split('_',1)[0]]] for key,value in full_data.items()}
        np.savez(root/'data.npz',**sliced)
        relation = next(r for r in old['relations'] if r['pair_id']==relation_id)
        config = copy.deepcopy(new)
        config.update(architectures=[arch],encodings=[enc],relations=[relation],
                      lengths=[n],length_weights={str(n):1.0},
                      validation_only_adapter=True,validation_protocol_sha256=protocol_hash)
        write(root/'adapter_config.json',config)
        feature_root = root/'features'
        feature_root.mkdir(exist_ok=True)
        for task in tasks:
            for phase in analysis.PHASES:
                for split in ('validation','test'):
                    record = records[arch,enc,task,phase,split]
                    arrays = {}
                    for field in FIELDS:
                        meta = record['arrays'][field]
                        original = np.load(meta['path'],mmap_mode='r',allow_pickle=False)
                        assert list(original.shape)==meta['shape'] and str(original.dtype)==meta['dtype']
                        arrays[field] = original[selectors[split]].copy()
                    destination=feature_root/f'{arch}_{enc}_{task}_{phase}_{split}.npz'
                    np.savez(destination,**arrays)
                    adapters.append(dict(case=case_name,task=task,phase=phase,split=split,
                        source_checkpoint_sha256=record['source_checkpoint_sha256'],
                        sources={field:record['arrays'][field] for field in FIELDS},
                        selected_original_rows=selectors[split].tolist(),path=str(destination),sha256=sha(destination)))
        completion = analysis.analyze_unit(config,root,root/'analysis')
        assert completion['status']=='completed'
        new_a = rows(root/'analysis/comparison_a_per_length.csv')
        old_a = [r for r in a_original if r['architecture']==arch and r['encoding']==enc
                 and r['relation']==relation_id and int(r['length'])==n
                 and r['layer'] in ('final_norm','output') and r['metric'] in analysis.METRICS]
        a_keys=('control','branch','metric')
        a_index={tuple(r[k] for k in a_keys):r for r in old_a}
        assert len(new_a)==len(old_a)==16
        case_checks=[]
        for row in new_a:
            source=a_index[tuple(row[k] for k in a_keys)]
            for field in analysis.COMPONENTS + analysis.EFFECTS + ('fit_samples','test_samples'):
                passed,error,both_undefined=numbers_equal(source[field],row[field])
                case_checks.append(dict(case=case_name,comparison='A',control=row['control'],
                    branch=row['branch'],metric=row['metric'],right_task=relation['right'],field=field,
                    expected=analysis.number(source[field]),observed=analysis.number(row[field]),
                    absolute_difference=error,both_undefined=both_undefined,passed=passed))
        new_b = rows(root/'analysis/comparison_b_per_length.csv')
        old_b = [r for r in b_original if r['domain']=='permworld' and r['architecture']==arch
                 and r['encoding']==enc and r['relation']==relation_id and int(r['length'])==n]
        b_keys=('control','right_task')
        b_index={tuple(r[k] for k in b_keys):r for r in old_b}
        assert len(new_b)==len(old_b)==30
        for row in new_b:
            source=b_index[tuple(row[k] for k in b_keys)]
            for field in ('fixed_transform_score_initial','fixed_transform_score_trained',
                          'anchor_count','selected_count','excluded_count','groups','residual_df'):
                passed,error,both_undefined=numbers_equal(source[field],row[field])
                case_checks.append(dict(case=case_name,comparison='B',control=row['control'],
                    branch='hidden',metric='linear_cka',right_task=row['right_task'],field=field,
                    expected=analysis.number(source[field]),observed=analysis.number(row[field]),
                    absolute_difference=error,both_undefined=both_undefined,passed=passed))
            for field in ('category','designated_right_task','transform'):
                case_checks.append(dict(case=case_name,comparison='B',control=row['control'],
                    branch='hidden',metric='linear_cka',right_task=row['right_task'],field=field,
                    expected=source[field],observed=row[field],absolute_difference=None,
                    both_undefined=False,passed=source[field]==row[field]))
        all_checks.extend(case_checks)
        failures=[r for r in case_checks if not r['passed']]
        result=dict(case=case_name,architecture=arch,encoding=enc,relation=relation_id,length=n,
                    a_rows=len(new_a),b_rows=len(new_b),checks=len(case_checks),
                    both_undefined=sum(r['both_undefined'] for r in case_checks),
                    failed=len(failures),maximum_absolute_difference=max(
                        (r['absolute_difference'] for r in case_checks if r['absolute_difference'] is not None),default=None),
                    raw_comparison_b_initial_zero_reference=completion['raw_initial_zero_reference_verified'],
                    analysis_seconds=completion['analysis_seconds'])
        case_results.append(result)
        print(json.dumps(result),flush=True)
        analysis.write_csv(out/'checks.csv',all_checks)
        if failures:
            write(out/'failure.json',dict(status='failed',case=case_name,tolerance=TOLERANCE,failures=failures))
            raise AssertionError('Original saved-activation comparison failed: '+case_name)
    end_hashes={str(path):sha(path) for path in sorted(source_paths)}
    differences={path:dict(start=digest,end=end_hashes.get(path)) for path,digest in start_hashes.items()
                 if end_hashes.get(path)!=digest}
    write(out/'protection_end.json',dict(utc=utc(),files=end_hashes,differences=differences,
          all_named_files_unchanged=not differences,protected_files=len(start_hashes)))
    write(out/'adapter_manifest.json',dict(records=adapters,source_index_sha256=sha(index_path),
          original_data_sha256=sha(data_path)))
    summary=dict(status='passed' if not differences else 'failed',utc=utc(),
        script_sha256=sha(__file__),validation_protocol_sha256=protocol_hash,
        no_model_construction=True,no_forward=True,no_training=True,cpu_threads=2,
        cases=case_results,cases_completed=len(case_results),
        total_a_rows=sum(r['a_rows'] for r in case_results),total_b_rows=sum(r['b_rows'] for r in case_results),
        checks=len(all_checks),failed_checks=sum(not r['passed'] for r in all_checks),
        both_undefined_checks=sum(r['both_undefined'] for r in all_checks),
        maximum_absolute_difference=max(r['absolute_difference'] for r in all_checks if r['absolute_difference'] is not None),
        tolerance=TOLERANCE,protected_files=len(start_hashes),protection_differences=differences,
        elapsed_seconds=time.monotonic()-started,
        scope='Four prespecified relation/configuration/encoding/length slices; all four final-state metrics, both branches/controls/phases, all 15 right candidates; not every original condition or a forward-pass audit.')
    write(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    assert not differences


if __name__=='__main__':
    main()
