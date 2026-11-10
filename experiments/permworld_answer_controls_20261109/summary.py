"""Saved-score summaries for the exploratory dual-answer-control diagnostic.

No checkpoint, optimizer, forward pass, or CKA calculation is performed here.
The inferential bootstrap block is one of the eight prescribed task pairs;
initializations, encodings and architectures remain coupled inside that block.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np


VERSIONS = ("pooled_raw", "within_length_debiased", "within_length_dual_answer_debiased")
VERSION_LABELS = {
    "pooled_raw": "Pooled ordinary CKA",
    "within_length_debiased": "Within-length U-CKA",
    "within_length_dual_answer_debiased": "Within-length dual-answer residual U-CKA",
}
PHASES = ("initial", "trained")
SCORE_KEYS = ("unit", "phase", "architecture", "encoding", "source_task", "candidate_task", "transform", "cka_version")
CELL_KEYS = ("relation", "architecture", "encoding", "cka_version")
BOOTSTRAP_SEED = 20261109
BOOTSTRAP_REPLICATES = 10000


def number(value):
    if value is None or value in ("", "nan", "NaN", "None", "undefined"):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row)) or ["status"]
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(safe(rows))
    temp.replace(path)


def unit_ids(config, rows):
    declared = [u["unit"] if isinstance(u, dict) else str(u) for u in config.get("units", [])]
    observed = sorted({str(r["unit"]) for r in rows})
    # Do not silently omit a missing registered unit.
    return list(dict.fromkeys(declared + observed))


def fixed_mean(values, expected):
    if len(values) != expected:
        return None
    values = [number(x) for x in values]
    return math.fsum(values) / expected if expected and all(v is not None for v in values) else None


def descriptives(values, expected=None):
    vals = [number(v) for v in values]
    finite = [v for v in vals if v is not None]
    n = len(finite)
    return dict(expected_units=len(vals) if expected is None else expected,
                finite_units=n, undefined_units=(len(vals) if expected is None else expected)-n,
                mean=math.fsum(finite)/n if n else None,
                median=statistics.median(finite) if n else None,
                sample_sd=statistics.stdev(finite) if n > 1 else None,
                minimum=min(finite) if n else None, maximum=max(finite) if n else None,
                positive_units=sum(v > 0 for v in finite), negative_units=sum(v < 0 for v in finite),
                zero_units=sum(v == 0 for v in finite),
                positive_in_all_registered_units=bool(len(vals) == (expected or len(vals)) and n == len(vals) and n >= 2 and all(v > 0 for v in finite)),
                same_nonzero_direction_all_registered_units=bool(n >= 2 and n == (expected or len(vals)) and (all(v > 0 for v in finite) or all(v < 0 for v in finite))))


def aggregate_scores(config, rows):
    """Accept supplied aggregates or fixed complete-length rows, never pool n."""
    lengths = list(config.get("lengths", range(10, 31)))
    weights=config.get('length_weights')
    if weights is not None:
        values=list(weights.values()) if isinstance(weights,dict) else list(weights)
        if len(values)!=len(lengths) or not np.allclose(values,1/len(lengths),rtol=0,atol=1e-15):
            raise ValueError('This diagnostic requires the fixed equal-length weights')
    groups = defaultdict(list)
    for row in rows:
        if row["cka_version"] not in VERSIONS:
            raise ValueError("Unregistered CKA version: " + row["cka_version"])
        groups[tuple(str(row[k]) for k in SCORE_KEYS)].append(dict(row))
    output = []
    for key, group in groups.items():
        row = dict(zip(SCORE_KEYS, key))
        aggregate = [g for g in group if g.get("length") in (None, "", "pooled", "aggregate", "all")]
        if aggregate:
            if len(aggregate) != 1 or len(group) != 1:
                raise ValueError("Duplicate or mixed aggregate/per-length CKA rows")
            row.update(cka=number(aggregate[0].get("cka")), aggregation="supplied_pooled" if row["cka_version"] == "pooled_raw" else "supplied_fixed_length_aggregate")
            row["undefined_reason"] = aggregate[0].get("undefined_reason", "")
            for name in ("samples", "sample_count", "minimum_samples_per_length", "residual_df", "nuisance_rank"):
                if name in aggregate[0]:row[name] = aggregate[0][name]
        else:
            if row["cka_version"] == "pooled_raw":
                raise ValueError("Pooled ordinary CKA cannot be reconstructed by averaging per-length scores")
            by_length = {int(g["length"]): g for g in group}
            if len(by_length) != len(group):
                raise ValueError("Duplicate fixed-length CKA component")
            values = [number(by_length[n].get("cka")) if n in by_length else None for n in lengths]
            row.update(cka=fixed_mean(values, len(lengths)), aggregation=f"fixed_equal_1/{len(lengths)}",
                       required_lengths=len(lengths), supplied_lengths=len(by_length),
                       undefined_lengths=";".join(str(n) for n, v in zip(lengths, values) if v is None),
                       undefined_reason="" if all(v is not None for v in values) else "missing_or_undefined_required_length")
        output.append(row)
    return output


def rank_candidates(config, scores):
    tasks = list(config["tasks"])
    groups = defaultdict(list)
    keys = tuple(k for k in SCORE_KEYS if k != "candidate_task")
    for row in scores:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in groups.items():
        identity = dict(zip(keys, key)); expected = set(tasks)-{identity["source_task"]}
        actual = [r["candidate_task"] for r in group]
        if len(set(actual)) != len(actual):
            raise ValueError("Duplicate candidate in a ranking inventory")
        full = set(actual) == expected and all(number(r["cka"]) is not None for r in group)
        for original in group:
            score = number(original["cka"])
            others = [number(r["cka"]) for r in group if r["candidate_task"] != original["candidate_task"]]
            row = dict(original, candidate_count=len(expected), observed_candidate_count=len(actual),
                       finite_candidate_count=sum(number(r["cka"]) is not None for r in group), ranking_complete=full)
            row["rank"] = 1+sum(x > score for x in others) if full else None
            row["ties_with_candidate"] = sum(x == score for x in others) if full else None
            row["unique_first"] = (row["rank"] == 1 and row["ties_with_candidate"] == 0) if full else None
            row["ranking_undefined_reason"] = "" if full else "missing_or_undefined_candidate_in_fixed_inventory"
            output.append(row)
    return output


def exact_alternatives(config, relation):
    rid = relation["pair_id"]
    known = set(config.get("alternative_exact_transforms", {}).get(rid, []))
    known.update(relation.get("also_exact_transforms", []))
    known.update(relation.get("additional_exact_transforms", []))
    known.discard(relation["input_transform"])
    unknown = known-set(config.get("actions", ["identity", "complement", "inverse"]))
    if unknown:raise ValueError("Unknown exact alternative transforms: " + str(unknown))
    return sorted(known)


def operation_class(encoding, transform):
    if transform == "identity":return "simple"
    if (encoding, transform) in (("one_line", "complement"), ("canonical_cycle", "inverse")):
        return "simple"
    if (encoding, transform) in (("canonical_cycle", "complement"), ("one_line", "inverse")):
        return "scrambling"
    return "unclassified"


def transformation_advantages(config, scores, rankings):
    index = {tuple(row[k] for k in SCORE_KEYS): row for row in scores}
    if len(index) != len(scores):raise ValueError("Duplicate aggregate CKA identity")
    ranking_index = {tuple(row[k] for k in SCORE_KEYS): row for row in rankings}
    units = unit_ids(config, scores)
    actions = list(config.get("actions", ["identity", "complement", "inverse"]))
    rows, paired = [], []
    for relation in config["relations"]:
        excluded = exact_alternatives(config, relation)
        designated = relation["input_transform"]
        controls = [t for t in actions if t != designated and t not in excluded]
        for arch in config["architectures"]:
            for enc in config["encodings"]:
                for version in VERSIONS:
                    for unit in units:
                        base = dict(unit=unit, relation=relation["pair_id"], architecture=arch, encoding=enc,
                                    source_task=relation["left"], candidate_task=relation["right"], cka_version=version,
                                    designated_transform=designated, operation_class=operation_class(enc, designated),
                                    eligible_alternative_transforms=";".join(controls),
                                    excluded_also_exact_transforms=";".join(excluded))
                        phases = {}
                        for phase in PHASES:
                            def get(t):
                                key = (unit, phase, arch, enc, relation["left"], relation["right"], t, version)
                                return index.get(key, {}), ranking_index.get(key, {})
                            original, rank = get(designated); correct = number(original.get("cka"))
                            candidates = [(t, number(get(t)[0].get("cka"))) for t in controls]
                            complete = bool(candidates and correct is not None and all(v is not None for _, v in candidates))
                            maximum = max(v for _, v in candidates) if complete else None
                            record = dict(base, phase=phase, designated_cka=correct,
                                strongest_eligible_alternative_cka=maximum,
                                strongest_eligible_alternative=";".join(t for t, v in candidates if complete and v == maximum),
                                transform_advantage=correct-maximum if complete else None,
                                designated_candidate_rank=rank.get("rank"), designated_candidate_ties=rank.get("ties_with_candidate"),
                                designated_unique_first=rank.get("unique_first"),
                                undefined_reason="" if complete else "no_eligible_alternative" if not controls else "missing_or_undefined_designated_or_alternative")
                            rows.append(record);phases[phase]=record
                        initial, trained = phases["initial"], phases["trained"]
                        ia, ta = initial["transform_advantage"], trained["transform_advantage"]
                        paired.append(dict(base, initial_transform_advantage=ia, trained_transform_advantage=ta,
                            advantage_training_increment=ta-ia if ia is not None and ta is not None else None,
                            initial_designated_cka=initial["designated_cka"], trained_designated_cka=trained["designated_cka"],
                            initial_strongest_alternative_cka=initial["strongest_eligible_alternative_cka"],
                            trained_strongest_alternative_cka=trained["strongest_eligible_alternative_cka"],
                            trained_designated_rank=trained["designated_candidate_rank"],
                            trained_designated_unique_first=trained["designated_unique_first"]))
    return rows, paired


def cohorts(config, units):
    historical = str(config.get("historical_unit", "archdev001"))
    new = [u for u in units if u != historical]
    return (("new_units", new), ("all_units", list(units)))


def cell_summaries(config, paired):
    units = unit_ids(config, paired); output=[]
    by_cell = defaultdict(dict)
    for row in paired:by_cell[tuple(row[k] for k in CELL_KEYS)][row["unit"]]=row
    for cohort, selected in cohorts(config, units):
        for key, values in by_cell.items():
            for effect in ("initial_transform_advantage", "trained_transform_advantage", "advantage_training_increment"):
                row = dict(zip(CELL_KEYS, key));sample=[values.get(u, {}).get(effect) for u in selected]
                row.update(cohort=cohort, effect=effect, **descriptives(sample, len(selected)), unit_ids=";".join(selected))
                row["per_unit_values"]=";".join(u+":"+("undefined" if number(values.get(u,{}).get(effect)) is None else repr(number(values[u][effect]))) for u in selected)
                output.append(row)
    return output


def _grid_estimate(config, paired, units, version, effect, exclude_pair=None, exclude_arch=None, exclude_encoding=None):
    relations=[r["pair_id"] for r in config["relations"] if r["pair_id"]!=exclude_pair]
    arches=[a for a in config["architectures"] if a!=exclude_arch]
    encs=[e for e in config["encodings"] if e!=exclude_encoding]
    by_key={(r["relation"],r["architecture"],r["encoding"],r["unit"]):r for r in paired if r["cka_version"]==version}
    pair_means=[];expected=0;finite=0;all_values=[]
    for relation in relations:
        values=[]
        for arch in arches:
            for enc in encs:
                for unit in units:
                    v=number(by_key.get((relation,arch,enc,unit),{}).get(effect));values.append(v);expected+=1
                    if v is not None:finite+=1;all_values.append(v)
        pair_means.append(fixed_mean(values,len(arches)*len(encs)*len(units)))
    return dict(mean=fixed_mean(pair_means,len(relations)),finite_cell_unit_mean=math.fsum(all_values)/len(all_values) if all_values else None,
                expected_cell_unit_values=expected,finite_cell_unit_values=finite,
                task_pair_count=len(relations),finite_task_pair_count=sum(v is not None for v in pair_means),
                pair_ids=relations,pair_means=pair_means)


def bootstrap_pairs(pair_values, bootstrap_indices):
    values=[number(v) for v in pair_values]
    if not values or any(v is None for v in values):
        return dict(mean=None,ci_low=None,ci_high=None,undefined_reason="incomplete_prespecified_task_pair_support")
    values=np.asarray(values,np.float64)
    if bootstrap_indices.shape[1]!=len(values):raise ValueError("Bootstrap pair inventory differs")
    means=values[bootstrap_indices].mean(1)
    low,high=np.quantile(means,[.025,.975])
    return dict(mean=float(values.mean()),ci_low=float(low),ci_high=float(high),undefined_reason="")


def bootstrap_and_leaveouts(config, paired):
    units=unit_ids(config,paired); n_pairs=len(config["relations"])
    generator=np.random.default_rng(BOOTSTRAP_SEED)
    draws=generator.integers(0,n_pairs,size=(BOOTSTRAP_REPLICATES,n_pairs))
    intervals=[];leaveouts=[];reductions=[]
    for cohort,selected in cohorts(config,units):
        for version in VERSIONS:
            for effect in ("trained_transform_advantage","advantage_training_increment"):
                full=_grid_estimate(config,paired,selected,version,effect)
                interval=bootstrap_pairs(full["pair_means"],draws)
                intervals.append(dict(cohort=cohort,cka_version=version,effect=effect,
                    **{k:v for k,v in full.items() if k not in ("pair_means","mean")},**interval,
                    pair_mean_values=';'.join(p+':'+('undefined' if v is None else repr(v)) for p,v in zip(full['pair_ids'],full['pair_means'])),
                    bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=BOOTSTRAP_REPLICATES,
                    bootstrap_unit="task_pair",seed_units=len(selected),
                    interval_scope="Exploratory task-pair bootstrap conditional on eight prescribed pairs and two transform families; not independent conditions or universal relations"))
                choices=[("task_pair",p["pair_id"],dict(exclude_pair=p["pair_id"])) for p in config["relations"]]
                choices += [("architecture",a,dict(exclude_arch=a)) for a in config["architectures"]]
                choices += [("encoding",e,dict(exclude_encoding=e)) for e in config["encodings"]]
                for kind,label,kwargs in choices:
                    value=_grid_estimate(config,paired,selected,version,effect,**kwargs)
                    change=value["mean"]-full["mean"] if value["mean"] is not None and full["mean"] is not None else None
                    leaveouts.append(dict(cohort=cohort,cka_version=version,effect=effect,exclusion_type=kind,excluded=label,
                        overall_mean=value["mean"],full_grid_mean=full["mean"],change_from_full=change,
                        absolute_change=abs(change) if change is not None else None,
                        expected_cell_unit_values=value["expected_cell_unit_values"],finite_cell_unit_values=value["finite_cell_unit_values"],
                        task_pair_count=value["task_pair_count"]))
        for earlier,later in zip(VERSIONS[:-1],VERSIONS[1:]):
            for effect in ("trained_transform_advantage","advantage_training_increment"):
                a=_grid_estimate(config,paired,selected,earlier,effect);b=_grid_estimate(config,paired,selected,later,effect)
                diffs=[y-x if x is not None and y is not None else None for x,y in zip(a["pair_means"],b["pair_means"])]
                reductions.append(dict(cohort=cohort,earlier_version=earlier,later_version=later,effect=effect,
                    comparison="later_minus_earlier",**bootstrap_pairs(diffs,draws),
                    interpretation="Pooled-to-within changes both length handling and estimator; within-to-residual adds the specified additive answer control"))
    for cohort,selected in cohorts(config,units):
        for version in VERSIONS:
            for effect in ("trained_transform_advantage","advantage_training_increment"):
                for kind in ("task_pair","architecture","encoding"):
                    group=[r for r in leaveouts if r["cohort"]==cohort and r["cka_version"]==version and r["effect"]==effect and r["exclusion_type"]==kind]
                    finite=[r["absolute_change"] for r in group if r["absolute_change"] is not None]
                    maximum=max(finite) if finite else None
                    for row in group:row["largest_absolute_influence_in_type"] = maximum is not None and row["absolute_change"]==maximum
    return intervals,leaveouts,reductions,draws


def simplicity(config,paired,draws):
    """Fixed coding hypothesis; pairing encodings within each task pair."""
    units=unit_ids(config,paired);rows=[];summary=[]
    lookup={(r["unit"],r["relation"],r["architecture"],r["encoding"],r["cka_version"]):r for r in paired}
    for relation in config["relations"]:
        transform=relation["input_transform"]
        simple=[e for e in config["encodings"] if operation_class(e,transform)=="simple"]
        complex_=[e for e in config["encodings"] if operation_class(e,transform)=="scrambling"]
        if len(simple)!=1 or len(complex_)!=1:raise ValueError("Coding hypothesis requires the fixed paired two-encoding design")
        for unit in units:
            for arch in config["architectures"]:
                for version in VERSIONS:
                    a=lookup.get((unit,relation["pair_id"],arch,simple[0],version),{});b=lookup.get((unit,relation["pair_id"],arch,complex_[0],version),{})
                    for effect in ("trained_transform_advantage","advantage_training_increment"):
                        x,y=number(a.get(effect)),number(b.get(effect))
                        rows.append(dict(unit=unit,relation=relation["pair_id"],architecture=arch,cka_version=version,effect=effect,
                            simple_encoding=simple[0],scrambling_encoding=complex_[0],simple_value=x,scrambling_value=y,
                            simple_minus_scrambling=x-y if x is not None and y is not None else None))
    for cohort,selected in cohorts(config,units):
        for version in VERSIONS:
            for effect in ("trained_transform_advantage","advantage_training_increment"):
                pair_values=[]
                for relation in config["relations"]:
                    values=[r["simple_minus_scrambling"] for r in rows if r["unit"] in selected and r["relation"]==relation["pair_id"] and r["cka_version"]==version and r["effect"]==effect]
                    pair_values.append(fixed_mean(values,len(selected)*len(config["architectures"])))
                summary.append(dict(cohort=cohort,cka_version=version,effect=effect,**bootstrap_pairs(pair_values,draws),
                    paired_contrast="simple minus scrambling within task pair/architecture/unit",task_pair_count=len(pair_values),
                    hypothesis="one_line/complement and canonical_cycle/inverse simple; other two nonidentity combinations scrambling; fixed before new CKA"))
    catalog=[dict(encoding=e,transform=t,operation_class=operation_class(e,t),
                  mathematical_scope="Exact token relabeling / per-cycle orientation reversal; no claim of matched network computation cost")
             for e in config["encodings"] for t in config.get("actions",["identity","complement","inverse"])]
    return rows,summary,catalog


def _learning_field(row,names):
    for field in names:
        if field in row:return number(row[field])
    return None


def learning_associations(config,paired,learning_rows):
    """Descriptive correlations only; neither cells nor shared tasks are iid."""
    if not learning_rows:return [],[]
    units=unit_ids(config,paired)
    # The learning module may supply length rows; retain a separate explicit
    # identity-metric summary, not a selected favorable transform or length.
    groups=defaultdict(list)
    for r in learning_rows:
        if r.get("phase","trained")!='trained' or r.get('split','test')!='test':continue
        action=r.get('transform',r.get('action','identity'))
        groups[(str(r['unit']),r['architecture'],r['encoding'],r.get('task',r.get('source_task')),action)].append(r)
    measures={
        'accuracy_gap':('accuracy_minus_length_mode','accuracy_gap','answer_accuracy_minus_mode','accuracy_difference','accuracy_gap_vs_mode'),
        'full_nll_gap':('full_nll_minus_length_prior_nll',),
        'conditional_nll_gap':('conditional_numeric_nll_minus_length_prior_nll',),
        'spearman':('spearman','spearman_rho','expected_answer_spearman','spearman_expected_answer','prediction_truth_spearman'),
    }
    lookup={}
    for key,rows in groups.items():
        aggregates=[r for r in rows if r.get('length') in (None,'','aggregate','all','mean','fixed_equal_mean')]
        summary={}
        for measure,names in measures.items():
            if len(aggregates)==1:summary[measure]=_learning_field(aggregates[0],names)
            else:
                by_n={int(r['length']):r for r in rows if str(r.get('length','')).isdigit()}
                summary[measure]=fixed_mean([_learning_field(by_n[n],names) if n in by_n else None for n in config['lengths']],len(config['lengths']))
        lookup[key]=summary
    detailed=[]
    for r in paired:
        if r['cka_version']!='within_length_dual_answer_debiased':continue
        source=lookup.get((r['unit'],r['architecture'],r['encoding'],r['source_task'],'identity'),{})
        partner=lookup.get((r['unit'],r['architecture'],r['encoding'],r['candidate_task'],r['designated_transform']))
        partner_action=r['designated_transform']
        if partner is None:
            partner=lookup.get((r['unit'],r['architecture'],r['encoding'],r['candidate_task'],'identity'),{});partner_action='identity'
        row={k:r[k] for k in ('unit','relation','architecture','encoding','source_task','candidate_task')}
        row.update(transform_advantage=r['trained_transform_advantage'],advantage_training_increment=r['advantage_training_increment'],partner_learning_transform=partner_action)
        for measure in measures:
            x,y=source.get(measure),partner.get(measure)
            row['source_'+measure]=x;row['partner_'+measure]=y
            row['pair_mean_'+measure]=(x+y)/2 if x is not None and y is not None else None
        detailed.append(row)
    from scipy.stats import spearmanr
    associations=[]
    for cohort,selected in cohorts(config,units):
        by_cell=defaultdict(list)
        for r in detailed:
            if r['unit'] in selected:by_cell[(r['relation'],r['architecture'],r['encoding'])].append(r)
        for effect in ('transform_advantage','advantage_training_increment'):
            for measure in ('pair_mean_accuracy_gap','pair_mean_full_nll_gap','pair_mean_conditional_nll_gap','pair_mean_spearman'):
                xy=[]
                for group in by_cell.values():
                    x=fixed_mean([r.get(measure) for r in group],len(selected));y=fixed_mean([r.get(effect) for r in group],len(selected))
                    if x is not None and y is not None:xy.append((x,y))
                valid=len(xy)>1 and len({x for x,y in xy})>1 and len({y for x,y in xy})>1
                rho=float(spearmanr(np.asarray(xy)[:,0],np.asarray(xy)[:,1]).statistic) if valid else None
                associations.append(dict(cohort=cohort,effect=effect,learning_measure=measure,spearman=rho,finite_cells=len(xy),
                    unit_aggregation='Equal unit mean within each fixed condition',
                    scope='Descriptive only: cells share tasks/transformations/models; no iid-cell p value',undefined_reason='' if valid else 'insufficient_or_constant_finite_cell_values'))
    return detailed,associations


def plots(config,paired,leaveouts,rankings,outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    outdir=Path(outdir);relations=list(config['relations']);units=unit_ids(config,paired)
    fig,axes=plt.subplots(len(config['architectures'])*len(config['encodings']),1,figsize=(13,12),sharex=True,constrained_layout=True,squeeze=False)
    colors=dict(zip(VERSIONS,('C0','C1','C2')))
    for ax,(arch,enc) in zip(axes.flat,((a,e) for a in config['architectures'] for e in config['encodings'])):
        for vi,version in enumerate(VERSIONS):
            offset=(vi-1)*.22
            for ri,relation in enumerate(relations):
                records=[r for r in paired if r['architecture']==arch and r['encoding']==enc and r['relation']==relation['pair_id'] and r['cka_version']==version]
                values=[number(r['trained_transform_advantage']) for r in records]
                finite=[v for v in values if v is not None]
                if not finite:continue
                mean=statistics.mean(finite)
                ax.errorbar(ri+offset,mean,yerr=[[mean-min(finite)],[max(finite)-mean]],fmt='o',color=colors[version],capsize=3,label=VERSION_LABELS[version] if ri==0 else None)
                for ui,(record,value) in enumerate(zip(records,values)):
                    if value is not None:ax.scatter(ri+offset+(ui-(len(values)-1)/2)*.022,value,s=16,color=colors[version],alpha=.65,
                                                   marker='x' if record['unit']==config.get('historical_unit','archdev001') else '.')
        ax.axhline(0,color='.5',linewidth=.8);ax.set_title(arch+' / '+enc);ax.set_ylabel('New transform advantage');ax.grid(axis='y',alpha=.2)
    axes.flat[0].legend(fontsize=7,ncol=3)
    axes.flat[-1].set_xticks(range(len(relations)),[r['pair_id'] for r in relations],rotation=25,ha='right')
    fig.suptitle('Three CKA versions; x = historical unit, dots = new units; bars = unit min–max (not bootstrap CIs)',fontsize=10)
    fig.savefig(outdir/'three_version_advantage.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
    for ci,cohort in enumerate(('new_units','all_units')):
        for ei,(effect,title) in enumerate(zip(('trained_transform_advantage','advantage_training_increment'),('Trained transform advantage','Trained minus initial advantage'))):
            ax=axes[ci,ei]
            chosen=[r for r in leaveouts if r['cohort']==cohort and r['cka_version']==VERSIONS[-1] and r['effect']==effect]
            for i,r in enumerate(chosen):
                value=number(r['overall_mean'])
                if value is not None:ax.scatter(value,i,color={'task_pair':'C0','architecture':'C1','encoding':'C2'}[r['exclusion_type']],s=25)
            full=number(chosen[0]['full_grid_mean']) if chosen else None
            if full is not None:ax.axvline(full,color='k',linestyle='--',linewidth=.8,label='Full grid')
            ax.axvline(0,color='.6',linewidth=.6);ax.set_yticks(range(len(chosen)),[r['exclusion_type']+': '+r['excluded'] for r in chosen],fontsize=7);ax.set_title(cohort+' / '+title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Residual U-CKA: prespecified leave-one-out sensitivity; dependent cells retained',fontsize=10)
    fig.savefig(outdir/'leave_one_out.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(10,10),sharex=True,sharey=True,constrained_layout=True)
    actions=list(config.get('actions',['identity','complement','inverse']))
    for vi,version in enumerate(VERSIONS):
        for pi,phase in enumerate(PHASES):
            ax=axes[vi,pi]
            for ui,unit in enumerate(units):
                vals=[]
                for t in actions:
                    matching=[r for r in rankings if r['unit']==unit and r['phase']==phase and r['architecture']=='gru' and r['encoding']=='canonical_cycle' and r['source_task']=='peaks' and r['candidate_task']=='valleys' and r['transform']==t and r['cka_version']==version]
                    vals.append(number(matching[0]['rank']) if len(matching)==1 else None)
                ax.plot(range(len(actions)),[np.nan if v is None else v for v in vals],marker='o',markersize=4,label=unit)
            ax.set_title(VERSION_LABELS[version]+'\n'+phase,fontsize=8);ax.set_xticks(range(len(actions)),actions,rotation=15);ax.set_ylim(len(config['tasks'])-.5,.5);ax.set_ylabel('Candidate rank');ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=7)
    fig.suptitle('GRU / canonical cycle: valleys among peaks candidates (rank 1 can be tied; see CSV)',fontsize=10)
    fig.savefig(outdir/'peaks_valleys_transform_ranks.png',dpi=170);plt.close(fig)


def peaks_summaries(config,rankings,learning_rows):
    units=unit_ids(config,rankings);result=[]
    for cohort,selected in cohorts(config,units):
        for phase in PHASES:
            for version in VERSIONS:
                for transform in config.get('actions',['identity','complement','inverse']):
                    rows=[r for r in rankings if r['unit'] in selected and r['phase']==phase and r['cka_version']==version and r['transform']==transform]
                    finite=[r for r in rows if number(r.get('rank')) is not None]
                    result.append(dict(cohort=cohort,phase=phase,cka_version=version,transform=transform,
                        registered_units=len(selected),finite_units=len(finite),
                        rank_minimum=min(number(r['rank']) for r in finite) if finite else None,
                        rank_maximum=max(number(r['rank']) for r in finite) if finite else None,
                        rank_one_units=sum(number(r['rank'])==1 for r in finite),
                        uniquely_first_units=sum(r.get('unique_first') is True for r in finite),
                        maximum_ties=max(int(r['ties_with_candidate']) for r in finite) if finite else None,
                        per_unit_rank_and_ties=';'.join(r['unit']+':'+str(r.get('rank'))+'(ties='+str(r.get('ties_with_candidate'))+')' for r in rows)))
    learning=[r for r in learning_rows if r.get('architecture')=='gru' and r.get('encoding')=='canonical_cycle'
              and r.get('task',r.get('source_task')) in ('peaks','valleys') and r.get('phase','trained')=='trained'
              and r.get('split','test')=='test']
    return result,learning


def report(config,summary_rows,intervals,leaveouts,simplicity_rows,associations,peaks,peak_summary,peak_learning,outdir):
    def fmt(value):return 'undefined' if number(value) is None else f'{float(value):.6f}'
    lines=['# Exploratory length-and-answer-control diagnostic','',
        'This report summarizes saved scores only. It does not establish a mathematical mechanism or functional benefit. The new transform advantage is designated-transform CKA minus the maximum over every eligible alternative transform, including identity; mathematically also-correct alternatives are excluded by the frozen formula catalog. This is not the old one-wrong-transform Bt, and its trained-minus-initial change is not the old Delta.','',
        'The pooled ordinary-CKA comparator is newly requested, rather than the frozen paper primary analysis (which was already within length and joint-answer stratified). Moving from pooled ordinary CKA to within-length U-CKA changes both length handling and estimator. Moving from within-length U-CKA to dual-answer residual U-CKA adds the specified additive one-hot regression.','',
        '## Prespecified task-pair bootstrap','',
        '| Cohort | Version | Effect | Mean | 95% percentile interval | Pairs |','|---|---|---|---:|---:|---:|']
    for r in intervals:
        lines.append(f"| {r['cohort']} | {r['cka_version']} | {r['effect']} | {fmt(r['mean'])} | {fmt(r['ci_low'])} to {fmt(r['ci_high'])} | {r['task_pair_count']} |")
    lines+=['',f'Bootstrap: {BOOTSTRAP_REPLICATES:,} draws of the {len(config["relations"])} prescribed task-pair blocks, seed {BOOTSTRAP_SEED}. All architectures, encodings and stochastic units stay inside their selected task-pair block. This small, exploratory interval is conditional on the finite task catalog and its two transform families, not a population guarantee over mathematical relationships. Unit repetitions do not increase the number of task pairs. Undefined required lengths, alternatives or grid components are not filled with zero or silently reweighted.','',
        '## Cells and sensitivity','',
        '| Cohort | Version | Cells positive in all registered units | Fixed cells |','|---|---|---:|---:|']
    for cohort,_ in cohorts(config,unit_ids(config,[])):
        for version in VERSIONS:
            selected=[r for r in summary_rows if r['cohort']==cohort and r['cka_version']==version and r['effect']=='trained_transform_advantage']
            lines.append(f"| {cohort} | {version} | {sum(r['positive_in_all_registered_units'] for r in selected)} | {len(selected)} |")
    lines+=['','Full per-unit scores, ranks, initial references, contrast changes, cell means/ranges and every exclusion remain in CSV. The largest absolute deletions are marked within task-pair, architecture and encoding exclusion types; this identifies influence rather than selecting a favorable retained subset.','',
        '| Cohort | Residual effect | Exclusion type | Largest absolute influence | Retained mean | Change from full |',
        '|---|---|---|---|---:|---:|']
    for row in leaveouts:
        if row['cka_version']==VERSIONS[-1] and row['largest_absolute_influence_in_type']:
            lines.append('| '+' | '.join(str(row[k]) for k in ('cohort','effect','exclusion_type','excluded'))+
                         f" | {fmt(row['overall_mean'])} | {fmt(row['change_from_full'])} |")
    lines += ['',
        '## Coding-simplicity hypothesis','',
        '| Cohort | Version | Effect | Simple minus scrambling | Pair-bootstrap interval |','|---|---|---|---:|---:|']
    for r in simplicity_rows:
        lines.append(f"| {r['cohort']} | {r['cka_version']} | {r['effect']} | {fmt(r['mean'])} | {fmt(r['ci_low'])} to {fmt(r['ci_high'])} |")
    lines+=['', 'The simple/scrambling labels were specified from the input encoding, not chosen from CKA: complement in one-line is token relabeling; inverse in canonical cycles reverses each cycle orientation about its fixed minimum. Complement in canonical cycles and inverse in one-line are the prespecified scrambling cases. Each simplicity contrast pairs the two encodings within a task pair, architecture and stochastic unit before the task-pair bootstrap; it is not an unpaired comparison between the six complement and two inverse pairs. This is a coding-description hypothesis, not proof of matched computational difficulty or an independently identified cause.','',
        '## Learning and peaks/valleys','',
        'The source-learning CSVs must distinguish full-vocabulary NLL from numeric-conditional NLL. A model is not declared at the length baseline solely from exact-match accuracy; prior-relative NLL and expectation–truth rank correlation are retained continuously. No arbitrary near-baseline tolerance was selected after observing results. Cell-level learning associations are descriptive because cells share tasks and models.','',
        'For GRU / canonical cycles, `peaks_valleys_rankings.csv` retains valleys\' rank and exact-tie count at identity, complement and inverse, for every version and available initial/trained unit. Rank one with fourteen exact ties at shared initialization is not unique task identification. Comparisons across transformations distinguish task identity from transformation-specific selectivity.','',
        '| Cohort | Phase | CKA version | Transform | Finite units | Rank range | Uniquely first | Maximum ties |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for row in peak_summary:
        lines.append('| '+' | '.join(str(row[k]) if row[k] is not None else 'undefined' for k in ('cohort','phase','cka_version','transform','finite_units'))+
                     f" | {fmt(row['rank_minimum'])}–{fmt(row['rank_maximum'])} | {row['uniquely_first_units']}/{row['registered_units']} | {row['maximum_ties']} |")
    lines += ['', 'Raw shared-initialization candidate ties need not survive candidate-dependent answer regression; the initial residual ranks are measured rather than assumed tied.', '',
              '| Unit | Task | Accuracy minus length mode | Full NLL minus length-prior NLL | Conditional-numeric NLL minus prior | Expected-answer Spearman |',
              '|---|---|---:|---:|---:|---:|']
    groups=defaultdict(list)
    for row in peak_learning:
        if row.get('transform',row.get('action','identity'))!='identity':continue
        groups[(row['unit'],row.get('task',row.get('source_task')))].append(row)
    for (unit,task),group in sorted(groups.items()):
        aggregates=[r for r in group if r.get('length') in (None,'','aggregate','all','mean','fixed_equal_mean')]
        fields=('accuracy_minus_length_mode','full_nll_minus_length_prior_nll','conditional_numeric_nll_minus_length_prior_nll','expected_answer_spearman')
        if len(aggregates)==1:values=[number(aggregates[0].get(f)) for f in fields]
        else:
            by_n={int(r['length']):r for r in group if str(r.get('length','')).isdigit()}
            values=[fixed_mean([by_n.get(n,{}).get(f) for n in config['lengths']],len(config['lengths'])) for f in fields]
        lines.append('| '+str(unit)+' | '+str(task)+' | '+' | '.join(fmt(v) for v in values)+' |')
    lines += ['', 'Learning entries above are identity-input, fixed-length summaries. Lower model-minus-prior NLL is better; numeric-conditional NLL is a separate quantity, never a replacement for unavailable full-vocabulary NLL. Undefined Spearman can result from a constant answer at a required length and does not mean zero ability. No near-baseline equivalence margin was registered, so these values are not converted into a binary claim that a specialist uses no input information.']
    lines += ['',
        '## Interpretation limits','',
        'No practical disappearance/equivalence bound was specified. A confidence interval crossing zero is not evidence that the effect is absent. A residual advantage, if present, supports structure beyond the prescribed additive label regressors; it does not remove joint-answer interactions, confidence, errors, input geometry or every nonlinear output feature. Using unbiased HSIC components does not by itself turn in-sample fitted residuals into an unbiased conditional-independence test. Uniform positivity, magnitude, initialization-relative changes and deletion sensitivity must be read together before advancing to probes. No probe or new-model training is launched here.','']
    (Path(outdir)/'report.md').write_text('\n'.join(lines))


def summarize(config, rows, learning_rows, outdir):
    """Public summary API; rows may be an iterable of dicts or a CSV path."""
    if isinstance(config,(str,Path)):config=json.loads(Path(config).read_text())
    rows=read_csv(rows) if isinstance(rows,(str,Path)) else list(rows)
    config=dict(config,units=unit_ids(config,rows))
    learning_rows=read_csv(learning_rows) if isinstance(learning_rows,(str,Path)) else list(learning_rows or [])
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    scores=aggregate_scores(config,rows);rankings=rank_candidates(config,scores)
    contrasts,paired=transformation_advantages(config,scores,rankings)
    cells=cell_summaries(config,paired);intervals,leaveouts,reductions,draws=bootstrap_and_leaveouts(config,paired)
    simple,sim_summary,catalog=simplicity(config,paired,draws)
    joined,associations=learning_associations(config,paired,learning_rows)
    peaks=[r for r in rankings if r['architecture']=='gru' and r['encoding']=='canonical_cycle' and r['source_task']=='peaks' and r['candidate_task']=='valleys']
    peak_summary,peak_learning=peaks_summaries(config,peaks,learning_rows)
    for name,data in [('cka_scores_aggregated.csv',scores),('candidate_ranks.csv',rankings),('transform_advantage_by_phase.csv',contrasts),
                      ('transform_advantage_by_unit.csv',paired),('cell_summary.csv',cells),('task_pair_bootstrap.csv',intervals),
                      ('leave_one_out.csv',leaveouts),('version_changes.csv',reductions),('simplicity_by_unit.csv',simple),
                      ('simplicity_summary.csv',sim_summary),('encoding_operation_classification.csv',catalog),
                      ('learning_geometry_join.csv',joined),('learning_associations.csv',associations),('peaks_valleys_rankings.csv',peaks),
                      ('peaks_valleys_summary.csv',peak_summary),('peaks_valleys_learning.csv',peak_learning)]:
        write_csv(outdir/name,data)
    np.save(outdir/'task_pair_bootstrap_indices.npy',draws,allow_pickle=False)
    if config.get('summary_make_plots',True):plots(config,paired,leaveouts,rankings,outdir)
    report(config,cells,intervals,leaveouts,sim_summary,associations,peaks,peak_summary,peak_learning,outdir)
    complete=dict(status='completed_saved_score_summary',score_rows=len(scores),rank_rows=len(rankings),
                  primary_task_pairs=len(config['relations']),stochastic_units=unit_ids(config,scores),
                  bootstrap_seed=BOOTSTRAP_SEED,bootstrap_replicates=BOOTSTRAP_REPLICATES,
                  no_checkpoint_forward=True,no_training=True,original_results_modified=False,
                  new_advantage_distinct_from_original_Bt=True,practical_equivalence_not_tested=True)
    write_json(outdir/'summary_completion.json',complete);return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--scores',required=True)
    parser.add_argument('--learning');parser.add_argument('--output-dir',required=True)
    a=parser.parse_args();print(json.dumps(summarize(a.config,a.scores,a.learning,a.output_dir),indent=2))


if __name__=='__main__':main()
