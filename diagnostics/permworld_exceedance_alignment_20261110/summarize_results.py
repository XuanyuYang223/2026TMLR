"""Frozen saved-score summary for the two position-alignment task pairs.

No probe fitting, model forward, activation adaptation, or test-score tuning.
The full-support result is primary. Each registered length has fixed 1/21
weight; an undefined length makes the scalar undefined instead of reweighting.
The common control maximum is computed AFTER equal-length aggregation.
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
import tempfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UNITS = ('archdev001', 'unit02', 'unit03', 'unit04', 'unit05')
NEW_UNITS = UNITS[1:]
ENCODINGS = ('canonical_cycle', 'one_line')
PHASES = ('initial', 'trained')
PHASES_BY_PAIR = dict(peaks_valleys=PHASES, exceedances_deficiencies=('trained',))
SUPPORTS = ('eligible', 'matched')
LENGTHS = tuple(range(10, 31))
SHUFFLE_SEEDS = tuple(range(1026, 1034))
PAIRS = ('peaks_valleys', 'exceedances_deficiencies')
CORRECT = dict(peaks_valleys='complement', exceedances_deficiencies='inverse')
WRONG = dict(peaks_valleys='inverse', exceedances_deficiencies='complement')
SOURCE_TARGET = dict(peaks_valleys=('peaks', 'valleys'),
                     exceedances_deficiencies=('exceedances', 'deficiencies'))
CONTROL_METHODS = ('means_only', 'identity_map', 'wrong_transform_map',
                   *(f'shuffle_{s}' for s in SHUFFLE_SEEDS))
REQUIRED_METHODS = ('direct', 'correct_map', *CONTROL_METHODS,
                    'within_source', 'within_target')
EXTRA_METHODS = ('oracle_count_position_prior', 'source_position_prior',
                 'target_position_prior_reindexed', 'oracle_count_source_position_prior',
                 'oracle_count_target_position_prior_reindexed', 'aligned_step0_target',
                 'aligned_fixed_points_target')
TOLERANCE = 1e-12


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row)) or ['status']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def maybe_float(value):
    if value is None or str(value).strip().lower() in ('', 'none', 'nan', 'null'):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def mean(values):
    return math.fsum(values) / len(values)


def canonical_method(pair, method):
    correct = CORRECT[pair]
    wrong = WRONG[pair]
    aliases = {
        'within_peaks': 'within_source', 'within_valleys': 'within_target',
        'within_exceedances': 'within_source', 'within_deficiencies': 'within_target',
        'within_target_reindexed': 'within_target',
        f'fit_{correct}__test_{correct}': 'correct_map',
        f'fit_identity__test_{correct}': 'identity_map',
        f'fit_{wrong}__test_{correct}': 'wrong_transform_map',
        **{f'fit_shuffle_{seed}__test_{correct}': f'shuffle_{seed}'
           for seed in SHUFFLE_SEEDS},
    }
    return aliases.get(method, method)


def normalize_detail(rows, pair, default_unit=None):
    """Accept vendor by_length.csv without mutating the original tables."""
    output = []
    for original in rows:
        raw = dict(original)
        method = canonical_method(pair, raw['method'])
        # Native-input diagnostics stay in the copied raw table; the common
        # comparison only includes scores on one shared correct-transform input.
        if method not in (*REQUIRED_METHODS, *EXTRA_METHODS):
            continue
        unit = raw.get('unit') or default_unit
        support = dict(all='eligible', full='eligible').get(raw['support'], raw['support'])
        n = int(raw.get('n', raw.get('anchors', 0)))
        finite = int(raw.get('finite', raw.get('defined_n', 0)))
        auc = maybe_float(raw.get('auc'))
        if n < 0 or finite < 0 or finite > n:
            raise ValueError(('Invalid sample counts', pair, unit, raw))
        if auc is not None and not -TOLERANCE <= auc <= 1 + TOLERANCE:
            raise ValueError(('AUROC outside [0,1]', pair, unit, raw))
        if (finite == 0) != (auc is None):
            raise ValueError(('Undefined score/count mismatch', pair, unit, raw))
        if (unit not in UNITS or raw['encoding'] not in ENCODINGS or
                raw['phase'] not in PHASES_BY_PAIR[pair] or support not in SUPPORTS or
                int(raw['length']) not in LENGTHS):
            raise ValueError(('Unregistered condition', pair, unit, raw))
        output.append(dict(pair=pair, unit=unit, encoding=raw['encoding'],
                           phase=raw['phase'], support=support, length=int(raw['length']),
                           method=method, original_method=raw['method'], n=n,
                           finite=finite, auc=auc))
    return output


def sample_auc_rows(auc, lengths, eligible, matched, *, unit, encoding, phase, method):
    """Optional independent aggregation path from saved per-sample AUROCs.

    AUROC itself is a within-sample position ranking. Single-class samples are
    undefined and excluded from that length's mean, with exact counts retained.
    No test labels or counts are used to fit or choose any readout.
    """
    auc, lengths = np.asarray(auc, dtype=np.float64), np.asarray(lengths)
    eligible, matched = np.asarray(eligible, dtype=bool), np.asarray(matched, dtype=bool)
    if auc.ndim != 1 or any(x.shape != auc.shape for x in (lengths, eligible, matched)):
        raise ValueError('Per-sample AUROC/support shapes differ')
    if np.any(matched & ~eligible):
        raise ValueError('Matched subset is not contained in eligible support')
    if np.any(np.isinf(auc)) or np.any((auc[np.isfinite(auc)] < 0) | (auc[np.isfinite(auc)] > 1)):
        raise ValueError('Invalid saved per-sample AUROC')
    rows = []
    for support, subset in [('eligible', eligible), ('matched', matched)]:
        for n in LENGTHS:
            keep = subset & (lengths == n)
            values = auc[keep]
            finite = values[np.isfinite(values)]
            rows.append(dict(unit=unit, encoding=encoding, phase=phase,
                             support=support, length=n, method=method, n=int(keep.sum()),
                             finite=len(finite), auc=mean(finite.tolist()) if len(finite) else None))
    return rows


def aggregate_detail(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = tuple(row[k] for k in ('pair', 'unit', 'encoding', 'phase', 'support', 'method'))
        if row['length'] in groups[key]:
            raise ValueError(('Duplicate length', key, row['length']))
        groups[key][row['length']] = row
    expected = {(pair, unit, enc, phase, support, method)
                for pair in PAIRS for unit in UNITS for enc in ENCODINGS
                for phase in PHASES_BY_PAIR[pair] for support in SUPPORTS for method in REQUIRED_METHODS}
    for pair in PAIRS:
        observed_extras = {r['method'] for r in rows if r['pair'] == pair and r['method'] in EXTRA_METHODS}
        expected |= {(pair, unit, enc, phase, support, method)
                     for unit in UNITS for enc in ENCODINGS for phase in PHASES_BY_PAIR[pair]
                     for support in SUPPORTS for method in observed_extras}
    extra = set(groups) - expected
    if extra:
        raise ValueError(('Unexpected score groups', sorted(extra)))
    output, coverage = [], []
    for key in sorted(expected):
        selected = groups.get(key, {})
        missing = [n for n in LENGTHS if n not in selected]
        undefined = [n for n in LENGTHS if n in selected and selected[n]['auc'] is None]
        values = [selected[n]['auc'] for n in LENGTHS if n in selected and selected[n]['auc'] is not None]
        row = dict(zip(('pair', 'unit', 'encoding', 'phase', 'support', 'method'), key))
        row.update(auc=mean(values) if not missing and not undefined else None,
                   registered_lengths=len(LENGTHS), present_lengths=len(selected),
                   finite_lengths=len(values), missing_lengths=';'.join(map(str, missing)),
                   undefined_lengths=';'.join(map(str, undefined)),
                   anchors=sum(r['n'] for r in selected.values()),
                   finite=sum(r['finite'] for r in selected.values()),
                   aggregation='Fixed 1/21 per registered length; no undefined-length reweighting')
        output.append(row)
        coverage.append({k: v for k, v in row.items() if k != 'auc'})
    # Shared-label main methods must retain exactly the same scoring support.
    for pair in PAIRS:
        for unit in UNITS:
            for enc in ENCODINGS:
                for phase in PHASES_BY_PAIR[pair]:
                    for support in SUPPORTS:
                        for n in LENGTHS:
                            counts = set()
                            for method in ('direct', 'correct_map', *CONTROL_METHODS):
                                r = groups.get((pair, unit, enc, phase, support, method), {}).get(n)
                                if r is not None:
                                    counts.add((r['n'], r['finite']))
                            if len(counts) > 1:
                                raise ValueError(('Shared-test support differs', pair, unit, enc, phase, support, n, counts))
    return output, coverage


def difference(a, b):
    return None if a is None or b is None else a - b


def comparisons(scores):
    ix = {tuple(r[k] for k in ('pair', 'unit', 'encoding', 'phase', 'support', 'method')): r for r in scores}
    rows = []
    for pair in PAIRS:
        for unit in UNITS:
            for encoding in ENCODINGS:
                for support in SUPPORTS:
                    get = lambda p, m: ix.get((pair, unit, encoding, p, support, m), {}).get('auc')
                    row = dict(pair=pair, unit=unit, encoding=encoding, support=support,
                               historical_reference=unit == UNITS[0],
                               correct_transform=CORRECT[pair], wrong_transform=WRONG[pair],
                               initial_source_reference_available='initial' in PHASES_BY_PAIR[pair],
                               step0_reference_scope='Initial source+target and refitted initial-source probe' if pair == 'peaks_valleys' else 'Frozen trained source/probe with initial target, separately named aligned_step0_target')
                    for phase in PHASES:
                        prefix = '' if phase == 'trained' else 'initial_'
                        for method in (*REQUIRED_METHODS, *EXTRA_METHODS):
                            row[prefix + method + '_auc'] = get(phase, method)
                        controls = {m: get(phase, m) for m in CONTROL_METHODS}
                        defined = all(v is not None for v in controls.values())
                        best = max(controls.values()) if defined else None
                        row[prefix + 'best_common_control_auc'] = best
                        row[prefix + 'best_common_control_names'] = ';'.join(k for k, v in controls.items() if v == best) if defined else ''
                        row[prefix + 'margin_vs_best_common'] = difference(get(phase, 'correct_map'), best)
                        row[prefix + 'gain_vs_direct'] = difference(get(phase, 'correct_map'), get(phase, 'direct'))
                        row[prefix + 'finite_common_controls'] = sum(v is not None for v in controls.values())
                    row['correct_map_change_from_initial'] = difference(row['correct_map_auc'], row['initial_correct_map_auc'])
                    row['common_margin_change_from_initial'] = difference(row['margin_vs_best_common'], row['initial_margin_vs_best_common'])
                    row['beats_direct_and_common'] = (row['gain_vs_direct'] > 0 and row['margin_vs_best_common'] > 0) if row['gain_vs_direct'] is not None and row['margin_vs_best_common'] is not None else None
                    row['control_definition'] = 'Maximum of means-only, identity, pair-specific wrong transform and EACH eight shuffle map, AFTER equal-length averaging; direct, own probes and step0 separate'
                    rows.append(row)
    return rows


def cross_pair(rows):
    ix = {(r['pair'], r['unit'], r['encoding'], r['support']): r for r in rows}
    output = []
    for unit in UNITS:
        for support in SUPPORTS:
            pv_ol = ix['peaks_valleys', unit, 'one_line', support]['margin_vs_best_common']
            pv_cy = ix['peaks_valleys', unit, 'canonical_cycle', support]['margin_vs_best_common']
            ed_ol = ix['exceedances_deficiencies', unit, 'one_line', support]['margin_vs_best_common']
            ed_cy = ix['exceedances_deficiencies', unit, 'canonical_cycle', support]['margin_vs_best_common']
            pv_contrast = difference(pv_ol, pv_cy)
            ed_contrast = difference(ed_cy, ed_ol)
            row = dict(unit=unit, support=support, historical_reference=unit == UNITS[0],
                       pv_one_line_margin=pv_ol, pv_cycle_margin=pv_cy,
                       ed_one_line_margin=ed_ol, ed_cycle_margin=ed_cy,
                       pv_margin_one_line_minus_cycle=pv_contrast,
                       ed_margin_cycle_minus_one_line=ed_contrast,
                       both_predicted_encoding_contrasts_positive=(pv_contrast > 0 and ed_contrast > 0) if pv_contrast is not None and ed_contrast is not None else None)
            output.append(row)
    return output


def summarize_numeric(rows, grouping, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in grouping)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        ix = {r['unit']: r for r in group}
        for cohort, units in [('new_units', NEW_UNITS), ('historical_reference', UNITS[:1])]:
            for metric in metrics:
                chosen = [(u, ix.get(u, {}).get(metric)) for u in units]
                values = [v for _, v in chosen if v is not None]
                complete = len(values) == len(units)
                row = dict(zip(grouping, key))
                row.update(cohort=cohort, metric=metric, registered_units=len(units), finite_units=len(values),
                           mean=mean(values) if complete else None,
                           sample_sd=statistics.stdev(values) if complete and len(values) > 1 else None,
                           minimum=min(values) if complete else None, maximum=max(values) if complete else None,
                           positive_units=sum(v > 0 for v in values), negative_units=sum(v < 0 for v in values),
                           zero_units=sum(v == 0 for v in values),
                           per_unit_values=';'.join(u + ':' + str(v) for u, v in chosen),
                           undefined_reason='' if complete else 'Missing/undefined unit; no cohort reweighting')
                output.append(row)
    return output


def compare_existing_summary(normalized, original, pair, unit):
    """Validate existing scalar means; no new score choice or tolerance."""
    groups = defaultdict(list)
    for row in normalized:
        groups[tuple(row[k] for k in ('encoding', 'phase', 'support', 'method'))].append(row)
    maximum, checked = 0., 0
    for row in original:
        method = canonical_method(pair, row['method'])
        if method not in (*REQUIRED_METHODS, *EXTRA_METHODS):
            continue
        key = (row['encoding'], row['phase'], row['support'], method)
        selected = groups[key]
        if len(selected) != len(LENGTHS) or any(r['auc'] is None for r in selected):
            continue
        observed = maybe_float(row['auc'])
        expected = mean([r['auc'] for r in selected])
        if observed is None or abs(observed - expected) > TOLERANCE:
            raise ValueError(('Saved summary disagrees', pair, unit, key, observed, expected))
        maximum = max(maximum, abs(observed - expected))
        checked += 1
    return dict(pair=pair, unit=unit, original_aggregates_checked=checked,
                maximum_absolute_error=maximum, tolerance=TOLERANCE)


def load_unit_tables(pv_root, ed_root):
    rows, raw, audits, sources = [], [], [], []
    for pair, root in [('peaks_valleys', pv_root), ('exceedances_deficiencies', ed_root)]:
        for unit in UNITS:
            detail_path = Path(root) / 'results' / unit / 'by_length.csv'
            if not detail_path.exists():
                # Missing units remain missing, never filled with zero.
                audits.append(dict(pair=pair, unit=unit, status='missing_detail', path=str(detail_path)))
                continue
            detail = read_csv(detail_path)
            sources.append(dict(path=str(detail_path.resolve()), sha256=sha(detail_path)))
            normalized = normalize_detail(detail, pair, unit)
            rows.extend(normalized)
            raw.extend(dict(r, pair=pair, unit=unit) for r in detail)
            summary_path = detail_path.with_name('summary.csv')
            if summary_path.exists():
                sources.append(dict(path=str(summary_path.resolve()), sha256=sha(summary_path)))
                audits.append(compare_existing_summary(normalized, read_csv(summary_path), pair, unit))
    return rows, raw, audits, sources


def load_auxiliary(pv_learning_root, ed_root):
    """Copy count-learning and coverage records with original scopes intact."""
    learning, coverage, sources = [], [], []
    for pair in PAIRS:
        for unit in UNITS:
            if pair == 'peaks_valleys':
                path = Path(pv_learning_root) / 'results' / unit / 'learning_by_length.csv'
                coverage_path = path.with_name('coverage.csv')
            else:
                path = Path(ed_root) / 'results' / unit / 'model_diagnostics_by_length.csv'
                coverage_path = path.with_name('coverage.csv')
            if path.exists():
                sources.append(dict(path=str(path.resolve()), sha256=sha(path)))
                for row in read_csv(path):
                    # Preserve every count metric and any extra scope columns.
                    learning.append(dict(row, pair=pair, unit=unit))
            if coverage_path.exists():
                sources.append(dict(path=str(coverage_path.resolve()), sha256=sha(coverage_path)))
                coverage.extend(dict(r, pair=pair, unit=unit) for r in read_csv(coverage_path))
    return learning, coverage, sources


def aggregate_learning(rows):
    groups = defaultdict(dict)
    for row in rows:
        task = row.get('task') or row.get('model_task')
        if task is None:
            raise ValueError(('Count diagnostic lacks task', row))
        if row.get('phase', 'trained') != 'trained':
            continue
        key = (row['pair'], row['unit'], row['encoding'], task,
               row.get('phase', 'trained'), row.get('support', 'eligible'),
               row.get('transform', 'identity'))
        n = int(row['length'])
        if n in groups[key]:
            raise ValueError(('Duplicate count-diagnostic length/scope', key, n))
        groups[key][n] = row
    result = []
    for key, lengths in sorted(groups.items()):
        row = dict(zip(('pair', 'unit', 'encoding', 'task', 'phase', 'support', 'transform'), key))
        missing = [n for n in LENGTHS if n not in lengths]
        for metric in ('accuracy', 'baseline', 'nll', 'baseline_nll', 'spearman', 'nll_difference', 'spearman_correlation'):
            values = [maybe_float(lengths[n].get(metric)) for n in LENGTHS if n in lengths]
            row[metric] = mean(values) if not missing and all(v is not None for v in values) else None
        row['accuracy_minus_length_baseline'] = difference(row['accuracy'], row['baseline'])
        row['nll_minus_length_baseline'] = difference(row['nll'], row['baseline_nll'])
        row['registered_lengths'] = len(LENGTHS)
        row['present_lengths'] = len(lengths)
        row['missing_lengths'] = ';'.join(map(str, missing))
        row['anchors'] = sum(int(r.get('n', 0)) for r in lengths.values())
        row['aggregation'] = 'Fixed 1/21 mean of per-length metrics; exact accuracy alone is not task mastery'
        result.append(row)
    return result


def plots(rows, contrasts, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for support in SUPPORTS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
        for ax, pair in zip(axes, PAIRS):
            for enc, offset in zip(ENCODINGS, (-.08, .08)):
                chosen = [next(r for r in rows if r['pair'] == pair and r['unit'] == u and r['encoding'] == enc and r['support'] == support) for u in UNITS]
                ax.plot(np.arange(len(UNITS)) + offset,
                        [r['margin_vs_best_common'] if r['margin_vs_best_common'] is not None else np.nan for r in chosen],
                        marker='o', label=enc)
            ax.axhline(0, color='.5', ls='--')
            ax.set_xticks(range(len(UNITS)), ['history', *NEW_UNITS], rotation=15)
            ax.set_title(pair.replace('_', ' → ', 1))
            ax.set_ylabel('AUROC: correct map − best common control')
            ax.grid(alpha=.2)
        axes[1].legend(fontsize=8)
        fig.suptitle(f'{support}: every control aggregated over all 21 lengths before maximization')
        fig.savefig(Path(out) / f'common_control_margins_{support}.png', dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    for support, marker in [('eligible', 'o'), ('matched', 'x')]:
        for field, offset in [('pv_margin_one_line_minus_cycle', -.08), ('ed_margin_cycle_minus_one_line', .08)]:
            chosen = [next(r for r in contrasts if r['unit'] == u and r['support'] == support) for u in UNITS]
            ax.plot(np.arange(len(UNITS)) + offset, [r[field] if r[field] is not None else np.nan for r in chosen], marker=marker, label=f'{support}: {field}')
    ax.axhline(0, color='.5', ls='--')
    ax.set_xticks(range(len(UNITS)), ['history', *NEW_UNITS], rotation=15)
    ax.set_ylabel('Difference in common-control margin')
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    fig.savefig(Path(out) / 'cross_pair_encoding_contrasts.png', dpi=160)
    plt.close(fig)


def fmt(value):
    return 'undefined' if value is None else f'{value:.6f}'


def report(rows, contrasts, summary, cross_summary, out):
    lines = ['# Existing-cache position-alignment comparison', '',
        'This is an exploratory cross-pair comparison using saved activations. The historical unit is an engineering reference; the four new units are reported individually and summarized with sample SD and range. This script fits no probe or map and performs no neural forward.', '',
        'Full eligible support is primary; count-matched support is secondary. Within each length, AUROC is averaged over defined per-sample position rankings. Every length n=10–30 then has fixed weight 1/21. Undefined lengths and units remain undefined. The common maximum includes means-only, identity, the pair-specific incorrect-transform map, and each of eight shuffled-pair maps separately. Maximization occurs after length aggregation. Direct transfer, model-internal readouts, step-0 and count-prior diagnostics remain separate.', '',
        '## Primary per-unit results', '',
        '|Pair|Unit|Encoding|Direct|Correct map|Best common|Margin|Own source|Own target|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['support'] != 'eligible':
            continue
        lines.append('|' + '|'.join([r['pair'], r['unit'], r['encoding'], *(fmt(r[k]) for k in ['direct_auc', 'correct_map_auc', 'best_common_control_auc', 'margin_vs_best_common', 'within_source_auc', 'within_target_auc'])]) + '|')
    lines += ['', '## Step-0 and position-prior diagnostics', '',
        'The old peaks→valleys initial phase refits a probe with an initial source and target. The new exceedances→deficiencies step-0 target control instead preserves the trained source and probe. These are different references, not interchangeable initialization effects. Prior and fixed-point controls are excluded from the common maximum and kept individually.', '',
        '|Pair|Unit|Encoding|PV initial-source map|ED initial target|ED fixed-point target|Source prior|Reindexed target prior|',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['support'] == 'eligible':
            lines.append('|' + '|'.join([r['pair'], r['unit'], r['encoding'], *(fmt(r[k]) for k in ['initial_correct_map_auc', 'aligned_step0_target_auc', 'aligned_fixed_points_target_auc', 'source_position_prior_auc', 'target_position_prior_reindexed_auc'])]) + '|')
    lines += ['', '## Paired encoding contrasts', '',
        'For peaks→valleys the contrast is one-line minus cycle; for exceedances→deficiencies it is cycle minus one-line. Both contrasts compare correct-minus-best-common margins, not raw aligned AUROC. Their signs are reported together within each unit; task pairs, encodings, lengths and shuffle seeds are not additional independent units.', '',
        '|Unit|Support|PV one-line−cycle|ED cycle−one-line|Both positive|', '|---|---|---:|---:|---|']
    for r in contrasts:
        lines.append('|' + '|'.join([r['unit'], r['support'], fmt(r['pv_margin_one_line_minus_cycle']), fmt(r['ed_margin_cycle_minus_one_line']), str(r['both_predicted_encoding_contrasts_positive'])]) + '|')
    lines += ['', '## Four-new-unit summaries', '',
        '|Pair|Encoding|Support|Metric|Mean|Sample SD|Min|Max|Positive units|',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['cohort'] == 'new_units' and r['metric'] in ('correct_map_auc', 'gain_vs_direct', 'margin_vs_best_common'):
            lines.append('|' + '|'.join([r['pair'], r['encoding'], r['support'], r['metric'], *(fmt(r[k]) for k in ('mean','sample_sd','minimum','maximum')), str(r['positive_units'])]) + '|')
    lines += ['', '## Interpretation limits', '',
        'Position AUROC measures recoverable and transferable position information under the frozen linear-probe and orthogonal-map protocol. It does not establish the neural counting mechanism, natural computational reuse, task accuracy gains or universal encoding effects. Per-sample true counts are only scoring/stratification information (and explicitly labeled oracle-prior diagnostics), never new fitting targets or test-time threshold selection. Shared checkpoints and inputs induce dependence. Undefined entries, support counts and every individual control are retained in the CSVs.', '']
    (Path(out) / 'report.md').write_text('\n'.join(lines))


def summarize(normalized, out, make_plots=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scores, coverage = aggregate_detail(normalized)
    rows = comparisons(scores)
    contrasts = cross_pair(rows)
    metrics = [k for k, v in rows[0].items() if k.endswith(('_auc', '_direct', '_common', '_initial')) and isinstance(v, (float, int, type(None)))]
    metrics = list(dict.fromkeys([*metrics, 'gain_vs_direct', 'margin_vs_best_common', 'correct_map_change_from_initial', 'common_margin_change_from_initial']))
    cohorts = summarize_numeric(rows, ('pair', 'encoding', 'support'), metrics)
    cross_stats = summarize_numeric(contrasts, ('support',), ('pv_margin_one_line_minus_cycle', 'ed_margin_cycle_minus_one_line'))
    conjunctions = []
    for support in SUPPORTS:
        selected = [r for r in contrasts if r['support'] == support and r['unit'] in NEW_UNITS]
        conjunctions.append(dict(support=support, registered_units=4,
                                 finite_units=sum(r['both_predicted_encoding_contrasts_positive'] is not None for r in selected),
                                 simultaneous_positive_units=sum(r['both_predicted_encoding_contrasts_positive'] is True for r in selected),
                                 per_unit_flags=';'.join(r['unit'] + ':' + str(r['both_predicted_encoding_contrasts_positive']) for r in selected)))
    for name, table in [('all_pair_by_length', normalized), ('per_unit_scores', scores), ('coverage', coverage),
                        ('per_unit_comparison', rows), ('per_unit_cross_pair', contrasts),
                        ('cohort_summary', cohorts), ('cross_pair_summary', cross_stats),
                        ('simultaneous_directions', conjunctions)]:
        write_csv(out / (name + '.csv'), table)
    if make_plots:
        plots(rows, contrasts, out)
    report(rows, contrasts, cohorts, cross_stats, out)
    return dict(status='completed_saved_position_alignment_summary', units=list(UNITS), new_units=list(NEW_UNITS),
                normalized_length_rows=len(normalized), scalar_rows=len(scores), comparison_rows=len(rows),
                cross_pair_rows=len(contrasts), registered_lengths=list(LENGTHS),
                common_controls=list(CONTROL_METHODS), maximum_after_fixed_length_aggregation=True,
                full_primary_matched_secondary=True, no_new_fits=True, no_neural_forward=True,
                undefined_lengths_not_reweighted=True)


def synthetic_checks():
    rows = []
    for pair in PAIRS:
        for u_i, unit in enumerate(UNITS):
            for encoding in ENCODINGS:
                for phase in PHASES_BY_PAIR[pair]:
                    for support in SUPPORTS:
                        for method in REQUIRED_METHODS:
                            for i, n in enumerate(LENGTHS):
                                value = .55
                                if method == 'correct_map':
                                    value = .7 + .001 * u_i
                                    if ((pair == 'peaks_valleys' and encoding == 'one_line') or
                                            (pair == 'exceedances_deficiencies' and encoding == 'canonical_cycle')):
                                        value += .03
                                if method == 'identity_map':
                                    value = .9 if i < 10 else .4
                                if method == 'wrong_transform_map':
                                    value = .4 if i < 10 else .9
                                rows.append(dict(pair=pair, unit=unit, encoding=encoding, phase=phase,
                                                 support=support, method=method, length=n,
                                                 n=20, finite=20, auc=value, original_method=method))
    with tempfile.TemporaryDirectory(prefix='alignment_summary_synthetic_') as temporary:
        result = summarize(rows, temporary, make_plots=False)
        table = read_csv(Path(temporary) / 'per_unit_comparison.csv')
        # max(mean(control)) differs from mean(max(control))=.9.
        expected = (10 * .4 + 11 * .9) / 21
        assert abs(float(table[0]['best_common_control_auc']) - expected) < TOLERANCE
        assert float(table[0]['best_common_control_auc']) < .9
        cross = read_csv(Path(temporary) / 'per_unit_cross_pair.csv')
        assert all(abs(float(r['pv_margin_one_line_minus_cycle']) - .03) < TOLERANCE and
                   abs(float(r['ed_margin_cycle_minus_one_line']) - .03) < TOLERANCE and
                   r['both_predicted_encoding_contrasts_positive'] == 'True' for r in cross)
        cohort = read_csv(Path(temporary) / 'cohort_summary.csv')
        selected = next(r for r in cohort if r['cohort'] == 'new_units' and r['pair'] == PAIRS[0] and
                        r['encoding'] == 'one_line' and r['support'] == 'eligible' and r['metric'] == 'correct_map_auc')
        assert abs(float(selected['mean']) - .7325) < TOLERANCE
        assert abs(float(selected['sample_sd']) - statistics.stdev([.731,.732,.733,.734])) < TOLERANCE
        assert selected['registered_units'] == '4'
    missing = [r for r in rows if not (r['pair'] == PAIRS[0] and r['unit'] == UNITS[0] and
               r['encoding'] == ENCODINGS[0] and r['phase'] == 'trained' and r['support'] == 'eligible' and
               r['method'] == 'correct_map' and r['length'] == 10)]
    scores, _ = aggregate_detail(missing)
    absent = next(r for r in scores if r['pair'] == PAIRS[0] and r['unit'] == UNITS[0] and
                  r['encoding'] == ENCODINGS[0] and r['phase'] == 'trained' and
                  r['support'] == 'eligible' and r['method'] == 'correct_map')
    assert absent['auc'] is None and absent['missing_lengths'] == '10'
    tiny = sample_auc_rows([.2, np.nan, .8], [10,10,11], [1,1,1], [1,0,1],
                           unit='archdev001', encoding='one_line', phase='trained', method='direct')
    assert tiny[0]['n'] == 2 and tiny[0]['finite'] == 1 and tiny[0]['auc'] == .2
    assert tiny[2]['auc'] is None and tiny[2]['finite'] == 0
    # A shared-score label/support mismatch must fail.
    bad = [dict(r) for r in rows]
    bad[0]['finite'] = 19
    try:
        aggregate_detail(bad)
    except ValueError as error:
        assert 'support differs' in str(error)
    else:
        raise AssertionError('Support mismatch was not rejected')
    aliases = [('peaks_valleys', 'fit_complement__test_complement', 'correct_map'),
               ('peaks_valleys', 'fit_inverse__test_complement', 'wrong_transform_map'),
               ('exceedances_deficiencies', 'fit_inverse__test_inverse', 'correct_map'),
               ('exceedances_deficiencies', 'fit_complement__test_inverse', 'wrong_transform_map')]
    assert all(canonical_method(p, original) == wanted for p, original, wanted in aliases)
    ed_reference = dict(pair='exceedances_deficiencies', unit=UNITS[0], encoding='one_line',
                        phase='trained', support='eligible', task='exceedances', n=20,
                        accuracy=.8, baseline=.4)
    diagnostics = [dict(ed_reference, length=n) for n in LENGTHS]
    learned = aggregate_learning(diagnostics)[0]
    assert abs(learned['accuracy_minus_length_baseline'] - .4) < TOLERANCE
    assert aggregate_learning(diagnostics[1:])[0]['accuracy'] is None
    scores, _ = aggregate_detail(rows)
    ed = next(r for r in comparisons(scores) if r['pair'] == PAIRS[1])
    assert ed['initial_correct_map_auc'] is None and not ed['initial_source_reference_available']
    # ED's initial target is explicitly a trained-source control, not a new
    # initial-source reference; it cannot populate PV's initial-phase fields.
    step0 = normalize_detail([dict(unit=UNITS[0], encoding='one_line', phase='trained',
                                  support='eligible', length=10, method='aligned_step0_target',
                                  n=2, finite=2, auc=.5)], PAIRS[1])[0]
    assert step0['phase'] == 'trained' and step0['method'] == 'aligned_step0_target'
    return dict(status='passed', synthetic_only=True, rows=len(rows),
                tests=['fixed length means', 'maximum after aggregation', 'paired encoding contrasts',
                       'sample SD across new units only', 'missing length is undefined',
                       'per-sample finite support', 'shared score-support guard', 'pair-specific method aliases',
                       'count diagnostic fixed-length aggregation', 'initial target is not initial source'],
                summary_rows=result['comparison_rows'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pv-root', type=Path, default=ROOT / 'diagnostics/permworld_orthogonal_alignment_20261110')
    parser.add_argument('--ed-root', type=Path, default=HERE)
    parser.add_argument('--out', type=Path, default=HERE / 'summary')
    parser.add_argument('--pv-learning-root', type=Path, default=ROOT / 'diagnostics/permworld_peaks_probe_20261109')
    parser.add_argument('--protocol', type=Path, default=HERE / 'protocol.json')
    parser.add_argument('--synthetic-checks', action='store_true')
    args = parser.parse_args()
    if args.synthetic_checks:
        result = synthetic_checks()
        result['source_sha256'] = sha(__file__)
        write_json(HERE / 'summary_synthetic_verification.json', result)
        print(json.dumps(result, indent=2))
        return
    # New score access is only permitted after the parent freezes the protocol.
    if not args.protocol.exists():
        raise FileNotFoundError('Freeze the protocol before accessing scores: ' + str(args.protocol))
    normalized, raw, checks, sources = load_unit_tables(args.pv_root, args.ed_root)
    learning, coverage, extra_sources = load_auxiliary(args.pv_learning_root, args.ed_root)
    result = summarize(normalized, args.out)
    write_csv(args.out / 'raw_source_by_length.csv', raw)
    write_csv(args.out / 'saved_summary_arithmetic_checks.csv', checks)
    write_csv(args.out / 'model_diagnostics_by_length.csv', learning)
    write_csv(args.out / 'model_diagnostics_equal_length.csv', aggregate_learning(learning))
    write_csv(args.out / 'source_coverage.csv', coverage)
    result.update(completed_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(args.protocol),
                  script_sha256=sha(__file__), source_tables=sources + extra_sources)
    write_json(args.out / 'completion.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
