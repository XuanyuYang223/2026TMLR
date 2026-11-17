"""Independent Decimal arithmetic and current checkpoint-file identity audit.

Does not import the frozen derivation helper, load model tensors, or evaluate
models. The fixed subset uses units02/05, both architectures/encodings/phases.
"""
from decimal import Decimal, localcontext
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'diagnostics/permworld_answer_controls_20261109'


def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def rows(path):
    with Path(path).open(newline='') as handle:return list(csv.DictReader(handle))


def main():
    completion=json.loads((HERE/'completion.json').read_text())
    assert completion['status']=='completed'
    config=json.loads((OLD/'protocol.json').read_text())
    by_phase={(r['unit'],r['architecture'],r['encoding'],r['phase'],r['cka_version']):r for r in rows(HERE/'cka_by_phase.csv')}
    checks=0;max_error=Decimal(0);selected=[]
    with localcontext() as context:
        context.prec=60
        for unit in ['unit02','unit05']:
            for arch in ['gru','transformer']:
                for enc in ['one_line','canonical_cycle']:
                    for phase in ['initial','trained']:
                        folder=Path(config['output'])/'scores'/f'{unit}_{arch}_{enc}_{phase}'
                        table=[]
                        for n in range(10,31):
                            table.extend(r for r in rows(folder/f'n{n}.csv') if r['source_task']=='exceedances' and r['candidate_task']=='deficiencies')
                        for version in ['within_length_debiased','within_length_dual_answer_debiased']:
                            means={t:sum(Decimal(r['cka']) for r in table if r['transform']==t and r['cka_version']==version)/Decimal(21) for t in ['identity','complement','inverse']}
                            result=means['inverse']-max(means['identity'],means['complement'])
                            observed=Decimal(by_phase[unit,arch,enc,phase,version]['transform_advantage'])
                            error=abs(observed-result);max_error=max(max_error,error);checks+=1
                            assert error<=Decimal('1e-12'), (unit,arch,enc,phase,version,str(error))
                            selected.append(dict(unit=unit,architecture=arch,encoding=enc,phase=phase,cka_version=version,decimal_reconstruction=str(result),saved_float=by_phase[unit,arch,enc,phase,version]['transform_advantage'],absolute_error=str(error)))
        original=rows(HERE/'cka_per_unit.csv')
        for summary in rows(HERE/'cka_four_unit_summary.csv'):
            cell=[r for r in original if r['unit'] in ['unit02','unit03','unit04','unit05'] and all(r[k]==summary[k] for k in ['architecture','encoding','cka_version'])]
            values=[Decimal(r[summary['effect']]) for r in cell];assert len(values)==4
            mean=sum(values)/Decimal(4);sd=(sum((v-mean)**2 for v in values)/Decimal(3)).sqrt()
            for field,expected in [('mean',mean),('sample_sd',sd),('minimum',min(values)),('maximum',max(values))]:
                error=abs(Decimal(summary[field])-expected);checks+=1;max_error=max(max_error,error)
                assert error<=Decimal('1e-12'), (summary,field,str(error))
    inventory=[r for r in rows(OLD/'checkpoint_inventory.csv') if r['task'] in ['exceedances','deficiencies']]
    assert len(inventory)==40
    physical={}
    for record in inventory:
        for prefix in ['checkpoint','initial_checkpoint']:
            path=record[prefix+'_path'];expected=record[prefix+'_sha256']
            if path in physical:assert physical[path]['expected_sha256']==expected;continue
            actual=sha(path);checks+=1
            assert actual==expected, path
            physical[path]=dict(path=path,expected_sha256=expected,actual_sha256=actual,size_bytes=Path(path).stat().st_size,passed=True)
    assert len(physical)==50
    with (HERE/'checkpoint_source_manifest.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(inventory[0]));writer.writeheader();writer.writerows(inventory)
    with (HERE/'independent_arithmetic_rows.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    clarification=('Post-results reader clarification: the residual CKA cycle-minus-one-line contrast is positive in all four GRU units, but negative in only three of four Transformer units. Transformer unit02 is positive (+0.0273509823511984). Thus the opposite architecture ordering stated in report.md refers to the four-unit means, not a uniform Transformer direction. The unchanged per-unit contrast tables retain this exception. This clarification adds no metric, model, sample or computational condition.\n')
    (HERE/'reader_clarification.md').write_text(clarification)
    result=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks=checks,arithmetic_checks=checks-50,maximum_Decimal_absolute_error=str(max_error),tolerance='1e-12',
                fixed_subset='units02/05 × botharchitectures × bothencodings × bothphases × bothperlengthCKAversions; Decimalprecision60; plusall36means/SD/min/max',
                fresh_checkpoint_file_hashes=list(physical.values()),checkpoint_manifest_rows=40,
                no_model_tensor_loading=True,no_forward=True,no_fitting=True,no_CKA_computation=True,
                initialization_identity_scope='Current actual savedstep0 file hashes match original audited inventory; initialization state_dict identities are retained from the original inventory, not reconstructed or re-estimated.',
                frozen_completion_sha256=sha(HERE/'completion.json'),source_sha256=sha(__file__),
                outputs={p.name:sha(p) for p in [HERE/'checkpoint_source_manifest.csv',HERE/'independent_arithmetic_rows.csv',HERE/'reader_clarification.md']})
    (HERE/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='fresh_checkpoint_file_hashes'},indent=2))


if __name__=='__main__':main()
