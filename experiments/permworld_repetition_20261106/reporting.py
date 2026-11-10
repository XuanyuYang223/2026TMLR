"""Generate a precisely scoped manuscript status from completed analyses only."""
from pathlib import Path
import csv


def update_paper_status(summary_dir, workspace):
    summary_dir=Path(summary_dir);workspace=Path(workspace)
    with (summary_dir/'unit_status.csv').open() as f:statuses=list(csv.DictReader(f))
    with (summary_dir/'unit_counts.csv').open() as f:counts=list(csv.DictReader(f))
    completed=[r['unit'] for r in statuses if r['analysis_completed']=='True']
    section=workspace/'manuscripts/representation_selectivity_20261012/sections/independent_repetition_status.tex'
    lines=[r'\subsection{Independent-unit repetition: current completion}',
           r'\label{sec:independent-repetition-status}',
           f'The repetition manifest targets five complete units. At this reporting checkpoint, {len(completed)} of five units have complete main analyses.']
    if len(completed)==1:
        lines += ['Only the historical exploratory unit is currently complete. Its saved-table arithmetic reproduces the original counts; this is a verification of the original evidence, not a new independent replication.',
                  r'No across-unit standard deviation, robustness claim, or five-unit result is available yet. The four new units remain explicit pending entries in the execution record.']
    else:
        lines += ['The following are unit-level observations. The 32 conditions and their candidate pairs within each unit are dependent measurements, and are not pooled as binomial trials.',
                  r'\begin{center}\small',r'\begin{tabular}{lrrrr}',
                  r'Unit & A: $B_t>0$ & A: $\Delta>0$ & B: above mean & B: unique first\\\hline']
        for unit in completed:
            selected=[r for r in counts if r['unit']==unit and r['control']=='answer_strata' and r['metric']=='linear_cka']
            a=next(r for r in selected if r['comparison']=='A');b=next(r for r in selected if r['comparison']=='B')
            label=unit.replace('_',r'\_')
            lines.append(f"{label} & {a['Bt_positive']}/32 & {a['Delta_positive']}/32 & {b['above_other_mean']}/32 & {b['unique_first']}/32"+r'\\')
        lines += [r'\end{tabular}\end{center}',
                  'The companion results retain unrounded condition effects, unit-level sample standard deviations and ranges, all unit-specific values, undefined cells, and direction consistency. A sign change near zero is interpreted with its magnitude rather than as evidence of a substantive reversal.']
        if len(completed)<5:lines.append('This is an interim report; incomplete units are not treated as negative results or omitted from the manifest.')
    lines += ['These repetitions jointly change sampling, shared architecture-level initialization, batch order, and the registered training random streams. They do not isolate the contribution of each random factor, and they do not test specialists initialized independently within a unit.','']
    temp=section.with_suffix('.tex.tmp');temp.write_text('\n'.join(lines));temp.replace(section)
