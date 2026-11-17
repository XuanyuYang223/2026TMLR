"""Standalone scientific figure from completed per-unit scalar tables only."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'headlines'
UNITS=('unit02','unit03','unit04','unit05')
CELLS=tuple((t,g,s) for t in ('peaks','valleys') for g in ('main','matched') for s in (-1,1))


def load(name):
    with (OUT/name).open(newline='') as f:
        rows=list(csv.DictReader(f))
    ix={(r['unit'],r['task'],r['group'],int(r['sign'])):r for r in rows}
    assert len(ix)==32 and len(rows)==32
    return ix


def main():
    main_rows=load('main_by_unit.csv');aux_rows=load('auxiliary_by_unit.csv')
    fig,axes=plt.subplots(1,2,figsize=(15,6.4),sharey=True)
    fig.subplots_adjust(left=.055,right=.99,top=.83,bottom=.35,wspace=.07)
    colors={'late':'#2166ac','early':'#d6604d'}
    offsets={'late':-.17,'early':.17}
    jitter=np.array([-.065,-.022,.022,.065])
    for ax,rows,fields,title in [
        (axes[0],main_rows,{'late':'late_relative_step','early':'early_relative_step'},'A  One-dimensional local-probe edit'),
        (axes[1],aux_rows,{'late':'late_whole_layer1_relative_step_hit','early':'early_whole_layer1_relative_step_hit'},'B  Complete first-layer state (256 coordinates)'),
    ]:
        for timing in ('late','early'):
            means=[];sds=[]
            for j,(task,group,sign) in enumerate(CELLS):
                values=np.array([float(rows[u,task,group,sign][fields[timing]])*100 for u in UNITS])
                ax.scatter(j+offsets[timing]+jitter,values,s=25,color=colors[timing],alpha=.55,zorder=3)
                means.append(float(values.mean()));sds.append(statistics.stdev(values.tolist()))
            ax.errorbar(np.arange(8)+offsets[timing],means,yerr=sds,fmt='D',ms=5,
                        color=colors[timing],capsize=3,elinewidth=1.3,zorder=4,
                        label=timing.capitalize()+' (mean ± sample SD)')
        ax.axhline(0,color='.5',lw=.8,zorder=1)
        ax.axvline(3.5,color='.7',lw=.8)
        ax.set_xticks(np.arange(8),[f'{"P" if t=="peaks" else "V"} · {"main" if g=="main" else "matched"}\n{s:+d}' for t,g,s in CELLS],rotation=35,ha='right')
        ax.set_ylim(-3,105);ax.set_xlim(-.65,7.65)
        ax.set_title(title,loc='left',fontsize=11,pad=12)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Exact requested relative count step (%)')
    axes[0].text(3.5,58,'All four units: 0%\nin both timing conditions',ha='center',va='center',fontsize=11)
    axes[0].text(3.5,37,'Independent local-checker crossing:\n98.82–100%, identical across timings',ha='center',va='center',fontsize=9,color='.25')
    axes[1].legend(loc='lower left',fontsize=8,frameon=False)
    fig.suptitle('Changing update timing does not rescue the chosen 1D edit; complete-L1 replacement can change counts',fontsize=13,y=.97)
    fig.text(.52,.175,'Task · donor construction · requested count change',fontsize=9.5,ha='center')
    fig.text(.055,.035,'Dots: individual existing units02–05; diamonds/error bars: mean ± sample SD. Historical unit excluded.\n'
             'Each unit uses equal registered centers within length and all n=10–30 with fixed 1/21 weights. P: peaks; V: valleys.\n'
             'Early replays upper current-token layers from recipient previous states; late retains recipient current upper states.\n'
             'Relative-step success is distinct from true-answer accuracy. The two payloads differ in dimension and transported information.',fontsize=8.5,ha='left',va='bottom')
    for suffix in ('png','pdf'):
        fig.savefig(OUT/f'timing_payload_comparison.{suffix}',dpi=200)
    plt.close(fig)
    sources={name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in ('main_by_unit.csv','auxiliary_by_unit.csv')}
    (OUT/'timing_figure_receipt.json').write_text(json.dumps(dict(status='completed',source_tables=sources,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),no_new_metrics=True,
        no_fits=True,no_forwards=True,all_eight_registered_cells=True,historical_excluded=True),indent=2)+'\n')


if __name__=='__main__':main()
