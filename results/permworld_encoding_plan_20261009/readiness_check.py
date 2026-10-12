"""Delivery checks; all synthetic analysis files live in disposable directories.

Run from the workspace with PYTHONDONTWRITEBYTECODE=1. This script neither
creates production data/checkpoints nor calls any training routine.
"""
import ast
import copy
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.permworld_encoding.core import (
    ACTIONS, ENCODINGS, RELATIONS, TASKS, encode, fingerprint, json_write, sha,
    verify_frozen,
)
from experiments.permworld_encoding.data import build_history, contains
from experiments.permworld_encoding.evaluate import components, summarize, write_csv
import numpy as np
from scipy.stats import t as student_t


def synthetic_statistics_check(config):
    """Independently check Gamma decomposition, simultaneous CIs and null labels."""
    with tempfile.TemporaryDirectory(prefix='encoding_analysis_readiness_') as temp:
        root = Path(temp)
        cfg = copy.deepcopy(config)
        cfg['output'] = str(root / 'analysis_fixture')
        cfg['lengths'] = [10, 11]
        cfg['units'] = [{'unit': f'fixture{i}'} for i in range(3)]
        cp = root / 'protocol.json'
        json_write(cp, cfg)
        json_write(root / 'freeze_manifest.json', {
            'config_sha256': sha(cp), 'code_sha256': {},
        })
        for i, unit in enumerate(cfg['units']):
            out = Path(cfg['output']) / unit['unit']
            out.mkdir(parents=True)
            rows, baselines, accuracy = [], [], []
            for relation in RELATIONS:
                correct = ACTIONS.index(relation['input_transform'])
                wrong = 1 if correct == 2 else 2
                for enc in ENCODINGS:
                    for control in ['raw_within_length', 'answer_strata']:
                        for n in cfg['lengths']:
                            # The independently known paired effects are .1, .2, .3.
                            cc = .1 * (i + 1) if enc == 'canonical_cycle' else 0.
                            v = {'initial': [0., 0., 0.], 'trained': [0., 0., 0.]}
                            v['initial'][correct] = .3
                            v['initial'][wrong] = .4
                            v['initial'][0] = .2
                            v['trained'][correct] = .4 + cc
                            v['trained'][wrong] = .3
                            v['trained'][0] = .2
                            fields = components(v, correct, wrong)
                            base = dict(unit=unit['unit'], relation=relation['pair_id'],
                                        encoding=enc, control=control, length=n)
                            for metric in ['ordinary', 'debiased', 'input_adjusted']:
                                rows.append(dict(**base, metric=metric, **fields))
                            baselines.append(dict(**base, baseline='output_prob', **fields))
                            for label in ['true_answer_onehot', 'true_answer_scalar']:
                                undefined = components({'initial': [None]*3,
                                                        'trained': [None]*3}, correct, wrong)
                                baselines.append(dict(**base, baseline=label, **undefined))
                    for condition in ['initial', 'trained']:
                        for task in TASKS:
                            for action in ACTIONS:
                                for n in cfg['lengths']:
                                    accuracy.append(dict(unit=unit['unit'], encoding=enc,
                                        condition=condition, task=task, action=action, length=n,
                                        full_vocab_accuracy=.125, numeric_conditional_accuracy=.25))
            write_csv(out / 'evaluation_length.csv', rows)
            write_csv(out / 'baseline_length.csv', baselines)
            # Accuracy is per task, rather than per relation; the fixture loop
            # above repeats it, so retain the unique natural evaluation keys.
            accuracy=list({(r['encoding'],r['condition'],r['task'],r['action'],r['length']):r
                           for r in accuracy}.values())
            assert len(accuracy)==2*2*16*3*2
            write_csv(out / 'task_accuracy_length.csv', accuracy)
            json_write(out / 'evaluation_complete.json', {'protocol_sha256': sha(cp)})
        summarize(cp)
        with (Path(cfg['output'])/'primary_statistics.csv').open() as f:
            primary = [r for r in csv.DictReader(f) if r['primary_test']=='True']
        assert len(primary)==2
        mean, sd, se = .2, .1, .1 / np.sqrt(3)
        critical = student_t.ppf(.9875, 2)
        p = 2 * student_t.sf(mean / se, 2)
        for row in primary:
            for field, expected in [('mean', mean), ('sd', sd),
                ('ci_low', mean-critical*se), ('ci_high', mean+critical*se),
                ('p_two_sided', p), ('bonferroni_p', min(1, 2*p))]:
                assert abs(float(row[field])-expected)<1e-12, (field, row)
        with (Path(cfg['output'])/'baseline_encoding_interactions.csv').open() as f:
            gold = [r for r in csv.DictReader(f) if r['baseline'].startswith('true_answer')]
        assert gold and all(r['gamma_delta']=='' for r in gold)
        # An undefined length must invalidate, rather than silently reweight, an endpoint.
        unitpath = Path(cfg['output'])/'fixture0'/'evaluation_length.csv'
        with unitpath.open() as f:
            incomplete = list(csv.DictReader(f))
        for row in incomplete:
            if (row['relation']=='descent_inverse' and row['encoding']=='canonical_cycle'
                and row['control']=='answer_strata' and row['metric']=='ordinary'
                and row['length']=='10'):
                row['delta']=''
        write_csv(unitpath, incomplete)
        summarize(cp)
        with (Path(cfg['output'])/'primary_statistics.csv').open() as f:
            incomplete_result = next(r for r in csv.DictReader(f)
                if r['relation']=='descent_inverse' and r['contrast']=='gamma_delta')
        assert incomplete_result['mean']=='' and incomplete_result['available_units']=='2'
    return {'known_encoding_effects_and_t_intervals': True,
            'undefined_gold_preserved': True, 'undefined_length_not_reweighted': True,
            'fixtures_deleted': True}


def synthetic_history_check():
    """Check old JSON and both archive layouts without scanning production data."""
    with tempfile.TemporaryDirectory(prefix='encoding_history_readiness_') as temp:
        root=Path(temp)
        p=(2,1,3,4,5,6,7,8,9,10)
        q=(3,2,1,4,5,6,7,8,9,10)
        s=(4,2,3,1,5,6,7,8,9,10)
        shard=root/'fixture.jsonl.gz'
        with gzip.open(shard,'wt') as f:
            f.write(json.dumps({'primary':list(p)})+'\n')
        parent=root/'manifest.json'
        json_write(parent,{'shards':[{'filename':shard.name,'sha256':sha(shard)}]})
        archive1=root/'objects.npz'
        np.savez(archive1,permutations=np.array([q],np.uint8),lengths=np.array([10]))
        archive2=root/'inputs.npz'
        np.savez(archive2,test_input=np.array([[encode(s,'one_line')]],np.uint16),
                 test_lengths=np.array([10]))
        cfg={'historical_exclusions':{'parent_manifest':str(parent),
            'parent_manifest_sha256':sha(parent),
            'archives':[{'path':str(a),'sha256':sha(a)} for a in [archive1,archive2]]}}
        indexpath,meta=build_history(cfg,root)
        index=np.load(indexpath)
        assert contains(index,[fingerprint(v) for v in [p,q,s]]).all()
        assert meta['unique_inputs']==3
        cached,cachedmeta=build_history(cfg,root)
        assert cached==indexpath and cachedmeta==meta
    return {'historical_json_and_both_archive_layouts': True,
            'historical_index_cache_identity': True, 'fixtures_deleted': True}


def main():
    plan=Path(__file__).resolve().parent
    config=verify_frozen(plan/'protocol.json')
    freeze=json.loads((plan/'freeze_manifest.json').read_text())
    assert sha(plan/'vocabulary.json')==freeze['vocabulary_sha256']
    assert sha(plan/'training_manifest.csv')==freeze['training_manifest_sha256']
    with (plan/'training_manifest.csv').open() as f:
        jobs=list(csv.DictReader(f))
    assert len(jobs)==len(config['units'])*32==768
    assert len({j['run_id'] for j in jobs})==len(jobs)
    for u in config['units']:
        chosen=[j for j in jobs if j['unit']==u['unit']]
        assert len(chosen)==32
        assert {(j['encoding'],j['task']) for j in chosen}=={
            (e,t) for e in ENCODINGS for t in TASKS}
    for field in ['model_seed','data_seed','batch_seed']:
        assert len({u[field] for u in config['units']})==len(config['units'])
    for path in (ROOT/'experiments/permworld_encoding').glob('*.py'):
        ast.parse(path.read_text())
    for module in ['data','train','evaluate']:
        subprocess.run([sys.executable,'-m',f'experiments.permworld_encoding.{module}',
                        '--help'],cwd=ROOT,check=True,capture_output=True,text=True)
    statistics=synthetic_statistics_check(config)
    history=synthetic_history_check()
    checks=[]
    for name,base in [
        ('diagnostics/permworld_cka_20261009/protected_before.json',ROOT),
        ('diagnostics/permworld_cka_20261009/deliverable_manifest.json',
            ROOT/'diagnostics/permworld_cka_20261009'),
        ('diagnostics/permworld_encoding_plan_20261009/manifest.json',
            ROOT/'diagnostics/permworld_encoding_plan_20261009')]:
        files=json.loads((ROOT/name).read_text())['files']
        for path,digest in files.items():
            assert sha(base/path)==digest, f'Protected artifact changed: {base/path}'
        checks.append({'inventory':name,'matched_files':len(files),'all_unchanged':True})
    tracked=subprocess.run(['git','status','--porcelain','--untracked-files=no'],
                           cwd=ROOT,check=True,capture_output=True,text=True).stdout
    assert tracked==''
    assert not (ROOT/config['output']).exists()
    report={'status':'passed','protocol_sha256':sha(plan/'protocol.json'),
        'frozen_source_and_vocabulary_identity':True,'manifest_models':len(jobs),
        'manifest_units':len(config['units']),'manifest_complete':True,
        'code_syntax_and_three_cli_entrypoints':True,
        'synthetic_statistics':statistics,'synthetic_history':history,
        'protected_artifacts':checks,'tracked_worktree_unchanged':True,
        'formal_output_directory_absent':True,'formal_training_started':False}
    json_write(plan/'readiness_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
