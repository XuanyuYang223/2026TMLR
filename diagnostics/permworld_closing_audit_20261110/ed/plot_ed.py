"""Post-results scientific display of every predefined E/D version/unit."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
UNITS=('unit02','unit03','unit04','unit05')
ENCS=('one_line','canonical_cycle')
VERSIONS=('pooled_raw','within_length_debiased','within_length_dual_answer_debiased')
COLORS=('#2868a4','#d76b4e')


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def rows(path):
    with Path(path).open(newline='') as handle:return list(csv.DictReader(handle))


def main():
    files=[HERE/'cka_per_unit.csv',HERE/'cka_four_unit_summary.csv',HERE/'probe_per_unit.csv',HERE/'probe_four_unit_summary.csv']
    cka,cka_summary,probe,probe_summary=map(rows,files)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,5.6),gridspec_kw={'width_ratios':[1.3,1.3,1]})
    checks=0
    for panel,arch in enumerate(['gru','transformer']):
        axis=axes[panel]
        for offset,(enc,color) in enumerate(zip(ENCS,COLORS)):
            for x,version in enumerate(VERSIONS):
                values=np.array([float(next(r for r in cka if (r['unit'],r['architecture'],r['encoding'],r['cka_version'])==(unit,arch,enc,version))['trained_transform_advantage']) for unit in UNITS])
                summary=next(r for r in cka_summary if (r['architecture'],r['encoding'],r['cka_version'],r['effect'])==(arch,enc,version,'trained_transform_advantage'))
                assert abs(values.mean()-float(summary['mean']))<=1e-12
                assert abs(values.std(ddof=1)-float(summary['sample_sd']))<=1e-12
                checks+=2;position=x+(-.15 if offset==0 else .15)
                axis.scatter(position+np.linspace(-.035,.035,4),values,color=color,s=28,alpha=.6,zorder=3)
                axis.errorbar(position,values.mean(),yerr=values.std(ddof=1),color=color,marker='D',markersize=6,capsize=4,elinewidth=1.3,zorder=4,label=enc.replace('_',' ') if x==0 else None)
        axis.axhline(0,color='#777777',linewidth=.8)
        axis.set_xticks(range(3),['Pooled\nordinary CKA','Within-length\nU-CKA','Dual-answer\nresidual U-CKA'])
        axis.set_title(f"{'AB'[panel]}  {arch.upper() if arch=='gru' else 'Transformer'} geometric selectivity",loc='left')
        axis.set_ylabel('Inverse CKA − best eligible alternative CKA')
        axis.grid(axis='y',alpha=.18)
    axis=axes[2]
    for x,(enc,color) in enumerate(zip(ENCS,COLORS)):
        values=np.array([float(next(r for r in probe if (r['unit'],r['encoding'],r['support'])==(unit,enc,'eligible'))['margin_vs_best_common']) for unit in UNITS])
        summary=next(r for r in probe_summary if (r['encoding'],r['support'],r['effect'])==(enc,'eligible','margin_vs_best_common'))
        assert abs(values.mean()-float(summary['mean']))<=1e-12
        assert abs(values.std(ddof=1)-float(summary['sample_sd']))<=1e-12
        checks+=2
        axis.scatter(x+np.linspace(-.045,.045,4),values,color=color,s=30,alpha=.6,zorder=3)
        axis.errorbar(x,values.mean(),yerr=values.std(ddof=1),color=color,marker='D',markersize=6,capsize=4,elinewidth=1.3,zorder=4)
    axis.axhline(0,color='#777777',linewidth=.8)
    axis.set_xticks([0,1],['One-line','Canonical\ncycle'])
    axis.set_xlim(-.45,1.45)
    axis.set_ylabel('Correct-map AUROC − best common-control AUROC')
    axis.set_title('C  GRU position-readout transfer',loc='left')
    axis.grid(axis='y',alpha=.18)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.92),ncol=2,frameon=False)
    fig.suptitle('E/D encoding rankings depend on nuisance control and measurement',fontsize=14,y=.99)
    fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.30,wspace=.37)
    caption=('Dots: all four existing units02–05; diamonds / error bars: mean ± sample SD. Historical unit excluded.\n'
             'A–B: inverse CKA minus max(identity, complement), after the registered aggregation; C has a separate AUROC scale.\n'
             'C: full eligible support; max of 11 common mapping controls is taken after fixed 1/21-length averaging.\n'
             'All three CKA versions are shown. Additive answer residualization does not remove all answer information.\n'
             'These observational rankings do not identify an encoding-caused mechanism or rescue the failed double-dissociation prediction.')
    fig.text(.065,.035,caption,ha='left',va='bottom',fontsize=9)
    png,pdf=HERE/'ed_encoding_comparison.png',HERE/'ed_encoding_comparison.pdf'
    fig.savefig(png,dpi=220);fig.savefig(pdf)
    plt.close(fig)
    receipt=dict(status='completed',created_utc=datetime.now(timezone.utc).isoformat(),post_results_reader_only=True,
                 no_new_metrics_or_conditions=True,no_model_or_probe_fitting=True,
                 all_four_formal_units_shown=True,scalar_summary_checks=checks,tolerance=1e-12,
                 source_sha256=sha(__file__),inputs={str(p):sha(p) for p in files},outputs={str(p):sha(p) for p in [png,pdf]})
    (HERE/'plot_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
