"""Requested +/-1 modal displacement, distinct from oracle-answer accuracy.

This is a deterministic additional view of already saved predictions, with no
new fitting, condition selection, metric search or source-result mutation.
"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import statistics

HERE=Path(__file__).resolve().parent
UNITS=['archdev001','unit02','unit03','unit04','unit05']


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    rows=[];counts={}
    for unit in UNITS:
        cells=defaultdict(lambda:[0,0,0,0,0])
        with (HERE/'results'/unit/'pair_results.csv').open() as stream:
            for r in csv.DictReader(stream):
                if r['condition_available'].lower()!='true':continue
                key=(r['task'],r['group'],int(r['sign']),r['condition'],int(r['length']),int(r['position']))
                a,b=int(r['recipient_argmax_count']),int(r['edited_argmax_count'])
                numeric=a>=0 and b>=0;sign=int(r['sign'])
                c=cells[key];c[0]+=1;c[1]+=int(numeric)
                c[2]+=int(numeric and b-a==sign)
                c[3]+=int(numeric and b!=a)
                c[4]+=int(r['recipient_argmax_token']!=r['edited_argmax_token'])
        groups=sorted(set(key[:4] for key in cells))
        for task,group,sign,condition in groups:
            expected=[(n,i) for n in range(10,31) for i in range(6 if group=='matched' else 2,n-1)]
            values={}
            for mi,name in [(1,'numeric_pair_coverage'),(2,'exact_signed_modal_delta_rate'),(3,'any_numeric_count_change_rate'),(4,'any_argmax_token_change_rate')]:
                by_n=[]
                for n in range(10,31):
                    positions=range(6 if group=='matched' else 2,n-1)
                    x=[]
                    for i in positions:
                        c=cells.get((task,group,sign,condition,n,i))
                        if c is None:break
                        x.append(c[mi]/c[0])
                    else:
                        by_n.append(statistics.mean(x));continue
                    break
                values[name]=statistics.mean(by_n) if len(by_n)==21 else ''
            total=[sum(cells[(task,group,sign,condition,n,i)][j] for n,i in expected if (task,group,sign,condition,n,i) in cells) for j in range(5)]
            rows.append(dict(unit=unit,task=task,group=group,sign=sign,condition=condition,
                **values,available_pairs=total[0],exact_signed_delta_n=total[2],numeric_change_n=total[3],token_change_n=total[4],
                aggregation='Fixed equal centers then1/21 lengths; no missing-cell reweight'))
        count_rows=[r for r in rows if r['unit']==unit and r['condition']=='local_projection']
        counts[unit]=dict(local_pairs=sum(r['available_pairs'] for r in count_rows),
            exact_signed_delta_hits=sum(r['exact_signed_delta_n'] for r in count_rows),
            any_numeric_count_changes=sum(r['numeric_change_n'] for r in count_rows),
            any_argmax_token_changes=sum(r['token_change_n'] for r in count_rows))
    write_csv(HERE/'exact_modal_delta_by_unit.csv',rows)
    summary=[]
    for key in sorted(set((r['task'],r['group'],r['sign'],r['condition']) for r in rows)):
        members=[r for r in rows if (r['task'],r['group'],r['sign'],r['condition'])==key and r['unit']!='archdev001']
        for metric in ['exact_signed_modal_delta_rate','any_numeric_count_change_rate','numeric_pair_coverage']:
            v=[r[metric] for r in members]
            summary.append(dict(zip(['task','group','sign','condition'],key),metric=metric,
                mean=statistics.mean(v) if len(v)==4 and '' not in v else '',
                sample_sd=statistics.stdev(v) if len(v)==4 and '' not in v else '',
                minimum=min(v) if len(v)==4 and '' not in v else '',maximum=max(v) if len(v)==4 and '' not in v else ''))
    write_csv(HERE/'exact_modal_delta_statistics.csv',summary)
    report=dict(status='completed',definition='Exact signed modal displacement: both full argmax outputs numeric AND edited_count-recipient_count==intended_sign. Oracle desired-answer hit remains a different metric.',
                per_unit_local_projection=counts,no_fitting=True,no_forward=True,no_new_test_or_condition=True,
                frozen_main_helper_unchanged=True)
    (HERE/'exact_modal_delta_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
