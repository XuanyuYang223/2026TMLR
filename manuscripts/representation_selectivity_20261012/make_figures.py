#!/usr/bin/env python3
"""Replot saved CSVs for the working paper; no model or feature-array access."""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PERM = REPO / 'diagnostics/permworld_eight_relations_20261010/final_review/permworld'
PAIR = REPO / 'diagnostics/representation_pairing_review_20261011'
MATRIX = REPO / 'diagnostics/matrixgroupworld_20261011'
RELATIONS = ['descent_inverse', 'peak_complement', 'fixed_complement',
             'exceedance_inverse', 'subsequence_complement', 'run_complement',
             'record_complement', 'decomposition_complement']
LABELS = ['Descents / inverse descents', 'Peaks / valleys', 'Fixed / anti-fixed points',
          'Exceedances / deficiencies', 'LIS / LDS', 'Increasing / decreasing run',
          'Left records: max / min', 'Global descents / components']
CONFIGS = [('transformer', 'one_line'), ('transformer', 'canonical_cycle'),
           ('gru', 'one_line'), ('gru', 'canonical_cycle')]
SHORT = ['TF / line', 'TF / cycle', 'GRU / line', 'GRU / cycle']
plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 9,
                     'mathtext.fontset': 'stix', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.bbox': 'tight'})
INPUTS = {}

def read(path):
    INPUTS[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))

def number(value):
    return float(value) if value not in ('', 'None') else float('nan')

def fmt(value):
    return '--' if not math.isfinite(value) else f'{value:+.3f}'

def save(fig, name):
    fig.savefig(HERE / 'figures' / (name + '.pdf'))
    fig.savefig(HERE / 'figures' / (name + '.png'), dpi=170)
    plt.close(fig)

def write_table(name, columns, rows):
    with (HERE / 'tables' / (name + '.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

def main():
    records = read(PERM / 'main32_final_cka_readout.csv')
    assert len(records) == 32
    keyed = {(x['relation'], x['architecture'], x['encoding']): x for x in records}
    ordered = [keyed[(rel, arch, enc)] for rel in RELATIONS for arch, enc in CONFIGS]
    assert sum(number(x['Bt']) > 0 for x in ordered) == 26
    assert sum(number(x['Delta']) > 0 for x in ordered) == 18
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.8), constrained_layout=True)
    matrices = [np.array([[number(keyed[(rel, a, e)][key]) for a, e in CONFIGS]
                          for rel in RELATIONS]) for key in ['B0', 'Bt', 'Delta']]
    extent = max(np.abs(x).max() for x in matrices)
    for ax, matrix, title in zip(axes, matrices, [r'Initialization $B_0$', r'Trained preference $B_t$', r'Increment $\Delta$']):
        im = ax.imshow(matrix, cmap='RdBu', vmin=-extent, vmax=extent, aspect='auto')
        ax.set_title(title)
        ax.set_xticks(range(4), SHORT, rotation=38, ha='right', fontsize=8)
        compact=['P1: descents / recoils','P2: peaks / valleys','P3: fixed / anti-fixed',
                 'P4: exceed. / defic.','P5: LIS / LDS','P6: increasing / decr. run',
                 'P7: left records max / min','P8: global desc. / comp.']
        ax.set_yticks(range(8), compact if ax is axes[0] else [''] * 8, fontsize=8)
        for i in range(8):
            for j in range(4):
                ax.text(j, i, f'{matrix[i,j]:+.2f}', ha='center', va='center',
                        color='white' if abs(matrix[i,j]) > .57 * extent else 'black', fontsize=8)
    fig.colorbar(im, ax=axes, shrink=.72, label='CKA score difference (common scale)')
    save(fig, 'figure2_perm_preferences')
    export = [{k: x[k] for k in ['relation', 'architecture', 'encoding', 'initial_correct',
                               'initial_wrong', 'trained_correct', 'trained_wrong', 'B0', 'Bt', 'dc', 'dw', 'Delta']}
              for x in ordered]
    write_table('perm_main32', list(export[0]), export)
    lines = []
    for i, x in enumerate(ordered):
        rel = RELATIONS.index(x['relation']) + 1
        config = SHORT[CONFIGS.index((x['architecture'], x['encoding']))]
        values = [fmt(number(x[k])) for k in ['initial_correct', 'initial_wrong', 'trained_correct', 'trained_wrong', 'B0', 'Bt', 'dc', 'dw', 'Delta']]
        lines.append(f'P{rel} & {config} & ' + ' & '.join('$'+v+'$' for v in values) + r' \\')
    (HERE / 'tables/perm_main32.tex').write_text('\n'.join(lines) + '\n')
    (HERE / 'tables/perm_main32_tabular.tex').write_text(
        r'\begin{tabular}{llrrrrrrrrr}'+'\n'+r'\toprule'+'\n'+
        r'& & $S_c^0$ & $S_u^0$ & $S_c^t$ & $S_u^t$ & $B_0$ & $B_t$ & $\delta_c$ & $\delta_u$ & $\Delta$\\'+'\n'+
        r'\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n')

    candidates = read(PAIR / 'pair_summary.csv')
    fixed_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 7.0), sharey=True, constrained_layout=True)
    for ax, control, title in zip(axes, ['raw_within_length', 'answer_strata'], ['Raw, within length', 'Joint-answer residuals']):
        wins, above = 0, 0
        for i, original in enumerate(ordered):
            selected = [x for x in candidates if x['domain'] == 'permworld'
                        and x['relation'] == original['relation'] and x['architecture'] == original['architecture']
                        and x['encoding'] == original['encoding'] and x['control'] == control]
            assert len(selected) == 15
            designated = next(x for x in selected if x['category'] == 'designated')
            others = sorted([x for x in selected if x['category'] == 'not_designated'], key=lambda x:x['right_task'])
            assert len(others) == 14
            values = [number(x['fixed_transform_score_trained']) for x in others]
            score = number(designated['fixed_transform_score_trained'])
            mean = sum(values) / 14
            rank = 1 + sum(x > score for x in values)
            ties = sum(x == score for x in values)
            above += score > mean
            wins += rank == 1 and ties == 0
            ax.scatter(values, i + np.linspace(-.15, .15, 14), s=11, c='#a1a1aa', alpha=.8, zorder=1)
            ax.scatter([mean], [i], marker='|', s=110, c='#252525', zorder=2)
            ax.scatter([score], [i], marker='D', s=28, c='#c56616', edgecolor='white', linewidth=.4, zorder=3)
            fixed_rows.append({'relation': original['relation'], 'architecture':original['architecture'],
                               'encoding':original['encoding'], 'control':control,
                               'designated_score':score, 'other_mean':mean, 'rank':rank,
                               'ties_with_designated':ties, 'candidate_count':15, 'other_count':14})
        assert (above, wins) == ((30,25) if control == 'raw_within_length' else (29,19))
        ax.set_title(title + f'\nAbove mean: {above}/32\nUnique first: {wins}/32',fontsize=9)
        ax.set_xlabel('Trained fixed-transform CKA')
        ax.set_xlim(-.03,1.03)
        ax.set_yticks(range(32), [f'P{RELATIONS.index(x["relation"])+1}  {SHORT[CONFIGS.index((x["architecture"],x["encoding"]))]}' for x in ordered], fontsize=7)
        ax.set_ylim(31.6,-.6)
        for y in np.arange(3.5,31,4): ax.axhline(y, color='#d6d6d6', linewidth=.65)
    axes[0].scatter([], [], c='#a1a1aa', s=14, label='14 non-designated candidates')
    axes[0].scatter([], [], c='#252525', marker='|', s=90, label='Non-designated mean')
    axes[0].scatter([], [], c='#c56616', marker='D', s=28, label='Designated correspondence')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',
               bbox_to_anchor=(.5,-.05),ncol=3,fontsize=7,frameon=False)
    save(fig, 'figure3_fixed_transform')
    write_table('fixed_transform32', list(fixed_rows[0]), fixed_rows)

    layers = read(PERM / 'all_layer_directions.csv')
    fig, axes = plt.subplots(1,2,figsize=(6.9,3.1),sharey=True,constrained_layout=True)
    layer_rows=[]
    for ax, arch in zip(axes, ['transformer','gru']):
        names=['embedding']+([f'block{i}'for i in range(1,5)]if arch=='transformer'else[f'layer{i}'for i in range(1,5)])+['final_norm']
        positive={'Bt':[], 'Delta':[]}
        for layer in names:
            selected=[x for x in layers if x['architecture']==arch and x['layer']==layer and x['control']=='answer_strata' and x['metric']=='linear_cka']
            assert len(selected)==16
            for key in positive:
                vals=[number(x[key])for x in selected]
                finite=sum(math.isfinite(v)for v in vals)
                count=sum(v>0 for v in vals)
                positive[key].append(count if finite else float('nan'))
                layer_rows.append({'architecture':arch,'layer':layer,'contrast':key,'positive':count,
                                   'finite':finite,'declared_conditions':16,'undefined':16-finite})
        x=np.arange(6)
        ax.bar(x-.18,positive['Bt'],width=.34,color='#256b7a',label=r'$B_t>0$')
        ax.bar(x+.18,positive['Delta'],width=.34,color='#c56616',label=r'$\Delta>0$')
        ax.text(0,6,'16/16\nundefined',ha='center',va='center',color='#606060',fontsize=8)
        ax.plot([0],[.4],marker='x',color='#606060')
        ax.set_title('Transformer' if arch=='transformer'else 'GRU')
        ax.set_xticks(x,['Input-end\nembedding','1','2','3','4','Final\nnorm'])
        ax.set_ylim(0,17);ax.set_yticks([0,4,8,12,16]);ax.set_xlabel('Declared layer / stage')
    axes[0].set_ylabel('Positive conditions, out of 16')
    axes[1].legend(loc='upper left',fontsize=8)
    save(fig,'figure4_layers')
    write_table('layer_counts',list(layer_rows[0]),layer_rows)

    matrices=read(MATRIX/'geometry_summary.csv')
    fig,axes=plt.subplots(2,3,figsize=(6.9,4.1),constrained_layout=True)
    metric_labels={'linear_cka':'CKA','knn_overlap':'10-neighbor overlap','procrustes':'Negative Proc. error'}
    r1rows=[]
    for i,unit in enumerate(['unit001','unit002']):
        for j,metric in enumerate(metric_labels):
            ax=axes[i,j]
            selected=[x for x in matrices if x['unit']==unit and x['relation']=='R1'and x['branch']=='prefix_final'and x['metric']==metric]
            assert len(selected)==2
            for q,control in enumerate(['raw_within_n','answer_strata']):
                row=next(x for x in selected if x['control']==control)
                vals=[number(row[k])for k in ['Bt','Delta']]
                bars=ax.bar(np.arange(2)+(q-.5)*.36,vals,width=.34,
                            color='#a1a1aa'if q==0 else '#256b7a',label='Raw'if q==0 else 'Answer control')
                for bar,val in zip(bars,vals):
                    ax.annotate(f'{val:+.3f}',(bar.get_x()+bar.get_width()/2,val),xytext=(0,3 if val>=0 else -11),textcoords='offset points',ha='center',fontsize=7)
                r1rows.append({k:row[k]for k in ['unit','relation','control','metric','initial_correct','initial_wrong','trained_correct','trained_wrong','B0','Bt','dc','du','Delta']})
            ax.axhline(0,color='black',linewidth=.7)
            ax.set_xticks([0,1],[r'$B_t$',r'$\Delta$']);ax.set_title(f'Unit {i+1}: {metric_labels[metric]}',fontsize=9)
            ax.margins(y=.28)
            if j==0: ax.set_ylabel('Signed score contrast')
    axes[0,0].legend(fontsize=7)
    save(fig,'figure5_matrix_r1')
    write_table('matrix_r1',list(r1rows[0]),r1rows)
    lines=[]
    for metric,label in metric_labels.items():
        selected=[x for x in r1rows if x['control']=='answer_strata'and x['metric']==metric]
        one=next(x for x in selected if x['unit']=='unit001');two=next(x for x in selected if x['unit']=='unit002')
        lines.append(label+' & '+' & '.join('$'+fmt(number(x[k]))+'$'for x in [one,two]for k in ['Bt','Delta'])+r' \\')
    (HERE/'tables/matrix_r1_main.tex').write_text('\n'.join(lines)+'\n')
    (HERE/'tables/matrix_r1_main_tabular.tex').write_text(
        r'\begin{tabular}{lrrrr}'+'\n'+r'\toprule'+'\n'+
        r'& \multicolumn{2}{c}{Matrix unit 1} & \multicolumn{2}{c}{Matrix unit 2}\\'+'\n'+
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}'+'\n'+
        r'Answer-controlled R1 readout & $B_t$ & $\Delta$ & $B_t$ & $\Delta$\\'+'\n'+
        r'\midrule'+'\n'+'\n'.join(lines)+'\n'+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n')
    graph=read(REPO/'diagnostics/graphworld_pilot_20261010/geometry_summary.csv')
    write_table('matrix_all_hidden',list(matrices[0]),[x for x in matrices if x['branch']=='prefix_final'])
    matrix_ranks=read(MATRIX/'designated_rank.csv')
    write_table('matrix_fixed_transform_ranks',list(matrix_ranks[0]),matrix_ranks)
    read(PAIR/'W_actual_summary.csv')
    read(PAIR/'gru_exceedance_encoding_case.csv')
    output_rows=[]
    for control in ['raw_within_length','answer_strata']:
        for branch,prefix in [('hidden',control+'_linear_cka_'),('output_prob','output_'+control+'_linear_cka_')]:
            row={'control':control,'branch':branch,'declared_conditions':32}
            for key in ['Bt','Delta']:
                vals=[number(x[prefix+key])for x in ordered]
                row[key+'_positive']=sum(v>0 for v in vals)
                row[key+'_finite']=sum(math.isfinite(v)for v in vals)
            output_rows.append(row)
    write_table('output_baseline_counts',list(output_rows[0]),output_rows)
    INPUTS.update({str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [REPO/'diagnostics/permworld_eight_relations_20261010/protocol.json',
                    REPO/'diagnostics/graphworld_pilot_20261010/protocol.json',
                    MATRIX/'protocol.json', REPO/'diagnostics/representation_pairing_review_20261011/protocol.json',
                    MATRIX/'actual_numerical_verification.json', MATRIX/'completion.json']})
    (HERE/'provenance/figure_sources.json').write_text(json.dumps({'status':'replotted_existing_scores_only',
        'git_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        'input_SHA256':INPUTS,'feature_arrays_opened':0,'model_forwards':0,'training_updates':0,
        'scope':'Saved CSV formatting/plotting and descriptive counts only. No new similarity measure or sample selection.'},indent=2)+'\n')
    print('Saved four score figures, seven CSV tables and source hashes; no model/activation access.')

if __name__=='__main__':main()
