"""Audit saved E/D CKA and juxtapose saved GRU position-readout margins.

No CKA kernel, activation, checkpoint, model, scaler, probe or mapping is fit.
The original fixed aggregates/undefined policy are preserved.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'diagnostics/permworld_answer_controls_20261109'
PROBE = ROOT / 'diagnostics/permworld_exceedance_alignment_20261110'
UNITS = ('archdev001', 'unit02', 'unit03', 'unit04', 'unit05')
NEW = UNITS[1:]
ARCHES = ('gru', 'transformer')
ENCS = ('one_line', 'canonical_cycle')
VERSIONS = ('pooled_raw', 'within_length_debiased', 'within_length_dual_answer_debiased')
PHASES = ('initial', 'trained')
LENGTHS = tuple(range(10, 31))
CONTROLS = ('means_only', 'identity_map', 'wrong_transform_map',
            *(f'shuffle_{seed}' for seed in range(1026, 1034)))
TOL = 1e-12


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def csvread(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def number(value):
    if value is None or str(value).lower() in ('', 'none', 'nan', 'null'):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def fixed_mean(values, expected):
    vals = [number(value) for value in values]
    return math.fsum(vals) / expected if len(vals) == expected and all(v is not None for v in vals) else None


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def csvwrite(path, data):
    fields = list(dict.fromkeys(key for row in data for key in row)) or ['status']
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(data)


def stats(values):
    vals = [number(value) for value in values]
    finite = [value for value in vals if value is not None]
    return dict(registered_units=len(values), finite_units=len(finite), undefined_units=len(vals)-len(finite),
                mean=math.fsum(finite)/len(finite) if finite else None,
                sample_sd=statistics.stdev(finite) if len(finite)>1 else None,
                minimum=min(finite) if finite else None, maximum=max(finite) if finite else None,
                positive_units=sum(v>0 for v in finite),negative_units=sum(v<0 for v in finite),
                zero_units=sum(v==0 for v in finite),raw_unit_values=json.dumps(dict(zip(NEW, vals))))


class Checks:
    def __init__(self): self.n=0; self.max_error=0.
    def same(self, actual, expected, name):
        actual, expected = number(actual), number(expected); self.n+=1
        if actual is None or expected is None:
            assert actual is None and expected is None, (name,actual,expected)
            return
        error=abs(actual-expected);self.max_error=max(self.max_error,error)
        assert error<=TOL, (name,actual,expected,error,TOL)


def check_sources(spec):
    missing=[]; verified=[]
    for item in spec['sources']:
        path=Path(item['path'])
        if item['status']=='missing':
            missing.append(item);continue
        assert path.is_file() and sha(path)==item['sha256'], path
        verified.append(dict(path=str(path),sha256=item['sha256'],size_bytes=path.stat().st_size))
    return verified,missing


def calculate_cka(config, checks):
    original=[r for r in csvread(OLD/'summary/transform_advantage_by_unit.csv') if r['relation']=='exceedance_inverse']
    phases=[r for r in csvread(OLD/'summary/transform_advantage_by_phase.csv') if r['relation']=='exceedance_inverse']
    index={(r['unit'],r['architecture'],r['encoding'],r['cka_version']):r for r in original}
    phase_index={(r['unit'],r['architecture'],r['encoding'],r['cka_version'],r['phase']):r for r in phases}
    raw=[r for r in csvread(OLD/'cka_results.csv') if r['source_task']=='exceedances' and r['candidate_task']=='deficiencies']
    raw_index={(r['unit'],r['architecture'],r['encoding'],r['phase'],r['cka_version'],r['transform']):r for r in raw}
    assert len(original)==60 and len(phases)==120 and len(raw)==360
    components=[]; actual={}; phase_rows=[]
    for unit in UNITS:
        for arch in ARCHES:
            for enc in ENCS:
                for phase in PHASES:
                    folder=Path(config['output'])/'scores'/f'{unit}_{arch}_{enc}_{phase}'
                    selected=[]
                    for n in LENGTHS:
                        source=folder/f'n{n}.csv'
                        with source.open(newline='') as handle:
                            cell=[r for r in csv.DictReader(handle) if r['source_task']=='exceedances' and r['candidate_task']=='deficiencies']
                        assert len(cell)==6, (source,len(cell))
                        for row in cell:
                            assert int(row['length'])==n and int(row['samples'])==512
                            selected.append(row);components.append(dict(row,source_path=str(source)))
                    for version in VERSIONS:
                        scores={}
                        for transform in ['identity','complement','inverse']:
                            key=(unit,arch,enc,phase,version,transform)
                            if version=='pooled_raw': score=number(raw_index[key]['cka'])
                            else:
                                cell=[r for r in selected if r['cka_version']==version and r['transform']==transform]
                                assert len(cell)==21
                                score=fixed_mean([r['cka'] for r in cell],21)
                            checks.same(score,raw_index[key]['cka'],str(key)+'/saved_aggregate')
                            scores[transform]=score
                        complete=all(v is not None for v in scores.values())
                        maximum=max(scores['identity'],scores['complement']) if complete else None
                        advantage=scores['inverse']-maximum if complete else None
                        actual[unit,arch,enc,version,phase]=advantage
                        known=phase_index[unit,arch,enc,version,phase]
                        checks.same(scores['inverse'],known['designated_cka'],'designated')
                        checks.same(maximum,known['strongest_eligible_alternative_cka'],'strongest')
                        checks.same(advantage,known['transform_advantage'],'phase_advantage')
                        phase_rows.append(dict(unit=unit,historical_reference=unit=='archdev001',architecture=arch,encoding=enc,cka_version=version,phase=phase,
                            inverse_cka=scores['inverse'],identity_cka=scores['identity'],complement_cka=scores['complement'],
                            strongest_eligible_alternative_cka=maximum,transform_advantage=advantage,
                            strongest_eligible_alternative=';'.join(t for t in ('identity','complement') if scores[t]==maximum) if complete else '',
                            required_lengths=21,anchors_per_length=512,all_anchors=10752,
                            aggregation='Pooled ordinary' if version=='pooled_raw' else 'Fixed1/21 mean; no undefined reweight',
                            undefined_reason='' if complete else 'undefined designated or eligible alternative'))
    per_unit=[]
    for unit in UNITS:
        for arch in ARCHES:
            for enc in ENCS:
                for version in VERSIONS:
                    key=(unit,arch,enc,version);row=dict(index[key]);initial=actual[*key,'initial'];trained=actual[*key,'trained']
                    increment=trained-initial if initial is not None and trained is not None else None
                    for field,value in [('initial_transform_advantage',initial),('trained_transform_advantage',trained),('advantage_training_increment',increment)]:
                        checks.same(value,row[field],field);row[field]=value
                    row['historical_reference']=unit=='archdev001';per_unit.append(row)
    csvwrite(HERE/'cka_per_length_components.csv',components)
    csvwrite(HERE/'cka_source_aggregates.csv',raw)
    csvwrite(HERE/'cka_by_phase.csv',phase_rows)
    csvwrite(HERE/'cka_per_unit.csv',per_unit)
    summaries=[]
    for arch in ARCHES:
        for enc in ENCS:
            for version in VERSIONS:
                for effect in ['initial_transform_advantage','trained_transform_advantage','advantage_training_increment']:
                    cell={r['unit']:r for r in per_unit if r['architecture']==arch and r['encoding']==enc and r['cka_version']==version}
                    summaries.append(dict(architecture=arch,encoding=enc,cka_version=version,effect=effect,
                                          **stats([cell[u][effect] for u in NEW])))
    csvwrite(HERE/'cka_four_unit_summary.csv',summaries)
    contrasts=[];contrast_summary=[]
    for arch in ARCHES:
        for version in VERSIONS:
            for unit in UNITS:
                row=dict(unit=unit,historical_reference=unit=='archdev001',architecture=arch,cka_version=version)
                for effect in ['initial_transform_advantage','trained_transform_advantage','advantage_training_increment']:
                    source={r['encoding']:r for r in per_unit if r['unit']==unit and r['architecture']==arch and r['cka_version']==version}
                    one,cycle=number(source['one_line'][effect]),number(source['canonical_cycle'][effect])
                    row[effect+'_cycle_minus_one_line']=cycle-one if one is not None and cycle is not None else None
                contrasts.append(row)
            for effect in ['initial_transform_advantage','trained_transform_advantage','advantage_training_increment']:
                cell={r['unit']:r for r in contrasts if r['architecture']==arch and r['cka_version']==version}
                contrast_summary.append(dict(architecture=arch,cka_version=version,effect=effect+'_cycle_minus_one_line',
                                             **stats([cell[u][effect+'_cycle_minus_one_line'] for u in NEW])))
    csvwrite(HERE/'encoding_contrasts_by_unit.csv',contrasts)
    csvwrite(HERE/'encoding_contrasts_summary.csv',contrast_summary)
    return summaries,contrast_summary


def calculate_probe(checks):
    by_length=[r for r in csvread(PROBE/'summary/all_pair_by_length.csv')
               if r['pair']=='exceedances_deficiencies' and r['phase']=='trained']
    known={(r['unit'],r['encoding'],r['support']):r for r in csvread(PROBE/'summary/per_unit_comparison.csv')
           if r['pair']=='exceedances_deficiencies'}
    grouped=defaultdict(list)
    for row in by_length:
        grouped[row['unit'],row['encoding'],row['support'],row['method']].append(row)
    output=[]
    for unit in UNITS:
        for enc in ENCS:
            for support in ['eligible','matched']:
                base=dict(unit=unit,historical_reference=unit=='archdev001',encoding=enc,support=support)
                for method in ['direct','correct_map',*CONTROLS,'within_source','within_target']:
                    cell=grouped[unit,enc,support,method]
                    assert sorted(int(r['length']) for r in cell)==list(LENGTHS)
                    base[method+'_auc']=fixed_mean([r['auc'] for r in cell],21)
                    base[method+'_finite_anchors']=sum(int(r['finite']) for r in cell)
                    base[method+'_anchors']=sum(int(r['n']) for r in cell)
                    base[method+'_undefined_lengths']=';'.join(r['length'] for r in cell if number(r['auc']) is None)
                    checks.same(base[method+'_auc'],known[unit,enc,support][method+'_auc'],'probe_'+method)
                control=[base[m+'_auc'] for m in CONTROLS]
                maximum=max(control) if all(v is not None for v in control) else None
                base['best_common_control_auc']=maximum
                base['margin_vs_best_common']=base['correct_map_auc']-maximum if maximum is not None and base['correct_map_auc'] is not None else None
                base['best_common_control_names']=';'.join(m for m in CONTROLS if base[m+'_auc']==maximum) if maximum is not None else ''
                checks.same(base['best_common_control_auc'],known[unit,enc,support]['best_common_control_auc'],'probe_maximum')
                checks.same(base['margin_vs_best_common'],known[unit,enc,support]['margin_vs_best_common'],'probe_margin')
                output.append(base)
    summaries=[]
    for enc in ENCS:
        for support in ['eligible','matched']:
            for field in ['direct_auc','correct_map_auc','best_common_control_auc','margin_vs_best_common','within_source_auc','within_target_auc']:
                cell={r['unit']:r for r in output if r['encoding']==enc and r['support']==support}
                summaries.append(dict(encoding=enc,support=support,effect=field,**stats([cell[u][field] for u in NEW])))
    csvwrite(HERE/'probe_per_unit.csv',output)
    csvwrite(HERE/'probe_four_unit_summary.csv',summaries)
    csvwrite(HERE/'probe_per_length_source.csv',by_length)
    return summaries


def report(cka,contrast,probe):
    def ck(arch,enc,version,effect):return next(r for r in cka if (r['architecture'],r['encoding'],r['cka_version'],r['effect'])==(arch,enc,version,effect))
    def pr(enc,effect):return next(r for r in probe if (r['encoding'],r['support'],r['effect'])==(enc,'eligible',effect))
    reference=[]
    refs={('gru','one_line'):.01951662,('gru','canonical_cycle'):.09678745,('transformer','one_line'):.10890765,('transformer','canonical_cycle'):.04960034}
    for (arch,enc),value in refs.items():
        actual=ck(arch,enc,VERSIONS[2],'trained_transform_advantage')['mean']
        reference.append(dict(metric='dual_answer_CKA_trained_advantage',architecture=arch,encoding=enc,reference=value,actual=actual,difference=actual-value,rounding_tolerance=5e-9,within_reference_rounding=abs(actual-value)<=5e-9))
    for enc,value in [('one_line',.010082),('canonical_cycle',.000150)]:
        actual=pr(enc,'margin_vs_best_common')['mean']
        reference.append(dict(metric='position_AUROC_common_control_margin',architecture='gru',encoding=enc,reference=value,actual=actual,difference=actual-value,rounding_tolerance=5e-7,within_reference_rounding=abs(actual-value)<=5e-7))
    csvwrite(HERE/'reference_checks.csv',reference)
    lines=['# E/D encoding audit: separate geometric and readout comparisons','',
        'Completed saved-table arithmetic audit. Four new units (unit02–unit05) define each summary; archdev001 remains a separate historical row. No fresh CKA, activation forward, probe, mapping or base model was fitted. This is a post-results audit, not a confirmation experiment.','',
        '## Three CKA versions and their initialization references','',
        'The trained transform advantage is inverse CKA minus max(identity CKA, complement CKA). This is not the paper’s old one-wrong-transform Bt, and its trained-minus-initial difference is not the old Delta. The maximum is taken after each transform’s registered length aggregation.','',
        '|Architecture|Encoding|Version|Initial advantage, mean ± sample SD|Trained advantage, mean ± sample SD|Trained range|',
        '|---|---|---|---|---|---|']
    for arch in ARCHES:
        for enc in ENCS:
            for version in VERSIONS:
                initial=ck(arch,enc,version,'initial_transform_advantage');trained=ck(arch,enc,version,'trained_transform_advantage')
                lines.append(f"|{arch}|{enc}|{version}|{initial['mean']:.8f} ± {initial['sample_sd']:.8f}|{trained['mean']:.8f} ± {trained['sample_sd']:.8f}|[{trained['minimum']:.8f}, {trained['maximum']:.8f}]|")
    lines += ['','Pooled_raw is ordinary CKA mixing the original lengths. Within_length_debiased computes U-centered CKA at each length; within_length_dual_answer_debiased adds the specified intercept and additive one-hot regressors for both true answers, with shared same-evaluation-sample residualization. Per-length versions use all original 512 anchors per n, and each n=10–30 has fixed1/21 weight. Undefined required components propagate rather than being reweighted. Moving from pooled to within-length changes both length handling and estimator.','',
        '## Paired encoding contrasts','',
        '|Architecture|Version|Trained cycle − one-line, mean ± sample SD|Range|Positive units /4|',
        '|---|---|---|---|---|']
    for row in contrast:
        if row['effect']=='trained_transform_advantage_cycle_minus_one_line':
            lines.append(f"|{row['architecture']}|{row['cka_version']}|{row['mean']:.8f} ± {row['sample_sd']:.8f}|[{row['minimum']:.8f}, {row['maximum']:.8f}]|{row['positive_units']}|")
    lines += ['','## GRU readout migration is a different measurement','',
        '|Encoding|Correct mapping AUROC, mean ± sample SD|Margin over best common mapping control, mean ± sample SD|Margin range|',
        '|---|---|---|---|']
    for enc in ENCS:
        correct=pr(enc,'correct_map_auc');margin=pr(enc,'margin_vs_best_common')
        lines.append(f"|{enc}|{correct['mean']:.8f} ± {correct['sample_sd']:.8f}|{margin['mean']:.8f} ± {margin['sample_sd']:.8f}|[{margin['minimum']:.8f}, {margin['maximum']:.8f}]|")
    lines += ['','These are the existing final-state, full-eligible-support probe results. The common comparison takes max over means-only, identity, the wrong complement mapping and each of eight shuffled mappings, after equal-length averaging. Direct transfer and model-internal probes remain separate. Matched support is a labeled sensitivity; its results and undefined/finite denominators are retained in the CSVs.','',
        'After the specified additive answer control, GRU CKA favors cycle over one-line, whereas the E/D position-probe margin favors one-line. Transformer CKA has the opposite encoding ordering to GRU. CKA selectivity and an aligned probe’s readout transfer are different questions with different nuisance controls; their numerical values must not be subtracted or treated as one effect. The original cross-task-pair encoding double-dissociation prediction failed and remains failed. This audit does not rehabilitate it.','',
        'The residual removes the prescribed additive answer-mean terms; it does not erase all answer information. Answer-dependent covariance, joint-answer interactions, confidence and prediction errors can remain. Same-evaluation residual fitting also couples row geometry; U-centered HSIC does not turn this into an unbiased conditional-independence test. Shared initialization means initial identity states are identical before task-specific training; candidate-specific residual designs can still affect other initial-transform scores.','',
        'Both encoding and model-family rankings here are observational across the tested tasks, trained configurations, datasets and saved initialization scheme. They do not alone identify an encoding-caused mathematical mechanism. Four stochastic units quantify within-setting variation; lengths, positions and derived comparisons are not additional independent units.','',
        '## Files and scope','',
        '* cka_per_unit.csv and cka_by_phase.csv: all five raw unit results, historical flagged.','* cka_four_unit_summary.csv: each architecture/encoding/version, initial/trained/increment; raw four-unit values, mean, sample SD, minimum and maximum.','* encoding_contrasts_by_unit.csv and encoding_contrasts_summary.csv: paired cycle−one-line contrasts including initialization and the training increment.','* cka_per_length_components.csv: all5,040 existing E/D length components, sample counts and residual-rank/DF fields preserved. No new per-length CKA was calculated.','* probe_per_unit.csv, probe_four_unit_summary.csv and probe_per_length_source.csv: the saved E/D probe comparison with support and undefined/finite counts.','* reference_checks.csv: reference numbers compared at their displayed rounding precision; references were not computational targets.','* spec.json and completion.json: named source SHA bindings, current code hash, global protocol binding and validation scope.','',
        'No paper or old result was overwritten. The primary deliverable is bounded evidence for writing, not a new mechanism claim or a new experiment.']
    (HERE/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha256',required=True)
    args=parser.parse_args()
    global_protocol=HERE.parent/'protocol.json'
    assert global_protocol.is_file() and sha(global_protocol)==args.protocol_sha256
    code_freeze=load(HERE/'code_freeze.json')
    assert code_freeze['code_sha256']==sha(__file__)
    assert code_freeze['spec_sha256']==sha(HERE/'spec.json')
    assert code_freeze['global_protocol_sha256']==args.protocol_sha256
    spec=load(HERE/'spec.json'); start_sources,missing=check_sources(spec)
    assert not missing, ('Frozen sources missing',missing)
    checks=Checks();config=load(OLD/'protocol.json')
    cka,contrast=calculate_cka(config,checks);probe=calculate_probe(checks)
    report(cka,contrast,probe)
    end_sources,end_missing=check_sources(spec)
    assert start_sources==end_sources and not end_missing
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head==spec['git_head']
    outputs={str(p):sha(p) for p in HERE.glob('*') if p.is_file() and p.name!='completion.json'}
    result=dict(status='completed',completed_utc=datetime.now(timezone.utc).isoformat(),
        global_protocol_sha256=args.protocol_sha256,spec_sha256=sha(HERE/'spec.json'),code_sha256=sha(__file__),
        source_files=len(start_sources),source_bytes=sum(r['size_bytes'] for r in start_sources),sources_unchanged=True,
        missing_sources=missing,arithmetic_checks=checks.n,maximum_arithmetic_error=checks.max_error,tolerance=TOL,
        cka_unit_rows=60,cka_phase_rows=120,cka_length_components=5040,formal_units=list(NEW),historical_reference=UNITS[0],
        no_new_CKA=True,no_model_forward=True,no_training=True,no_probe_fit=True,no_map_fit=True,
        results_exploratory_posthoc=True,git_head=head,outputs=outputs)
    write(HERE/'completion.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='outputs'},indent=2))


if __name__=='__main__':main()
