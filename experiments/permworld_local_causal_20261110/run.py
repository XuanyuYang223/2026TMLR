"""Two freezes: historical geometry development, then fixed formal evaluation."""
from datetime import datetime,timezone
import argparse
import subprocess

import torch
from threadpoolctl import threadpool_limits

from .shared import ROOT,PACKAGE,DIAG,OUT,TASKS,CONDITIONS,RANDOM_SEEDS,read,write,sha


def engineering_freeze():
    path=DIAG/'engineering_protocol.json'
    if path.exists():return verify_engineering()
    source=read(DIAG/'source_audit.json')
    assert source['status']=='passed' and not source['failures']
    assert read(DIAG/'math_audit.json')['status']=='passed'
    assert read(DIAG/'data/archdev001/completion.json')['status']=='complete'
    protocol=dict(version='local-extremum-recurrent-causal/v1',created_utc=datetime.now(timezone.utc).isoformat(),
        git_head=source['git_head'],source_audit_sha256=sha(DIAG/'source_audit.json'),
        model=source['model'],units=source['units'],output=str(OUT),tasks=list(TASKS),encoding='one_line',
        lengths=list(range(10,31)),centers='main2..n-2; matched6..n-2; main_i_ge6 sensitivity',
        data='Fixed64candidate roots/cell/origin, fit12/dev6/test8 family quotas, entire generated family/complement components, ownoriginaltrainingp/C exclusion; all derived actualCF and same-label nodes included. Keepallshortfalls.',
        random_control_seeds=list(RANDOM_SEEDS),conditions=list(CONDITIONS),
        local_endpoint='Recipient true full count + intended sign for BOTH main and matched. Matched donor full/resolved counts equalrecipient, but differs at targeti and earlierk2. Full-state donor wiring target differs from local-only goal in matched.',
        source_probes='Float64 StandardScaler+Ridge alpha10 globally across lengths/positions per task/CP/statepoint. Only count-matched FIT_A natural states train editing direction; component-disjoint FIT_B trains independent checker and a diagnostic prefix-count reader. Binary boundary.5. Probes own source labels only; no final count-output optimization.',
        component='Raw actual GRU h_n at ONE layer, direction equal normalized raw editing classifier weight. Dose1 copies true donor projection. Independent checker is the primary manipulation check; editing probe equality alone is tautological.',
        development='Historical dev only. Both BEFORE-right-value and AFTER-right-value readouts reported; onlyAFTER eligible forprimarycut. Choose one COMMON layer maximizing minimum independent-checker true boundarycrossing across tasks×groups×signs, then min directedscorechange, then higherlayer. Dose1 fixed. Neverselectusingfinalcountchanges.',
        manipulation_criterion='Descriptive per-unit/group/task/sign reliability: independent-checker true boundarycrossing>=.8 withpositive directedscorechange. Crossingrequires recipient starts on own-label side and ends on intended opposite side. Ifweak, keepoutcomes but callcausalinterpretationinconclusive; no retuning.',
        controls='Same-label natural donor: same task/n/i,label,resolvedprefixcount,fullcount,postcut suffix. Earlier-only or full-read-prefix variants recorded. Eight global 1D Gaussian directions with exact planned treatment norm/coordinate sign; paired within each example. Noop; full4layer donor state; real local-only CFinput. Allrealpositivecontrols separatelyreported.',
        recurrence='Cut immediately aftervaluepi_(i+1), token4+2i. Patch actual one-layer GRU h_n,seedallfour recurrenth0 withtheir corresponding carriedstates forNEXTtoken; no LayerNorm feedback. Correctabsolute positions; originalprefix/tail/querytaskretained.',
        numerical=dict(dtype='FP32recurrence and actualcarriededits;FP64linearprobes/planneddeltas',
            native_noop_logits_absolute=1e-4,fullswap_donor_logits_absolute=1e-4,
            planned_norm_absolute=1e-10,actual_norm_absolute=1e-5,actual_norm_relative=1e-5),
        count='Baselineunmodifiedcontinuation fromsamecarriedstateengine. Rawfull-vocabularyargmax; numeric0..30else−1; signedargmaxdelta onlywhenbothnumeric. Conditionalnumeric0..30expectation plusfullvocabnumericmass. Allpairsprimary; baselinecorrect subsets diagnosticonly.',
        statistics='Fournewunitsindependentbundles; tasks/signs/groups separate. Pairtreatmentminuscontrolbeforecellaggregation, random8meanwithinpair requiringall8. Fixed21equal lengths andequalpredeclaredcenterswithinlength;missingrequiredcellsundefined. Availablecell/pooled sensitivities explicit. No perpairindependenceornewpvalues.',
        limitations='Countmatched donors reduce existingcountertransport but lowrankmix may overlap genericcountaxes or beoffmanifold. Prefix-countprobe diagnostic isnotcomplete informationremoval. Afterupdate interventions cannotestablishearliernecessity. No cross-modelexchange/newcombinationtraining inthisphase.',
        actual_operations='Newcounterfactualinputsfromexistinganchors,frozenforward/Ridgeprobe/realstatepatch/scoring only. No optimizer/backward/basecheckpointupdate,oldreportorF17edit,autopublish.',
        no_formal_test_output_access_before_final_freeze=True)
    write(path,protocol)
    sources=[PACKAGE/name for name in ['__init__.py','shared.py','pairs.py','recurrent.py','features.py','probes.py','calibrate.py','run.py']]
    sources += [DIAG/'source_audit.json',DIAG/'math_audit.json',DIAG/'pair_generation_protocol.json',
                ROOT/'tests/test_permworld_local_causal_pairs.py',ROOT/'tests/test_permworld_local_causal_recurrent.py']
    write(DIAG/'engineering_freeze.json',dict(protocol_sha256=sha(path),created_utc=datetime.now(timezone.utc).isoformat(),
        sources={str(p.relative_to(ROOT)):sha(p) for p in sources},before_historical_probe_fits=True))
    return protocol


def verify_engineering():
    frozen=read(DIAG/'engineering_freeze.json')
    assert sha(DIAG/'engineering_protocol.json')==frozen['protocol_sha256']
    for path,digest in frozen['sources'].items():assert sha(ROOT/path)==digest,('Engineering dependency changed',path)
    return read(DIAG/'engineering_protocol.json')


def formal_freeze():
    protocol=verify_engineering()
    source=read(DIAG/'source_audit.json')
    for unit in source['units']:
        receipt=read(DIAG/'data'/unit['unit']/'completion.json')
        assert receipt['status']=='complete'
        assert receipt['data_sha256_before']==receipt['data_sha256_after']==unit['data_sha256']
        for path,digest in receipt['output_sha256'].items():assert sha(path)==digest
    selection=read(DIAG/'selection.json')
    path=DIAG/'protocol.json'
    if path.exists():
        return verify_formal()
    protocol.update(created_formal_freeze_utc=datetime.now(timezone.utc).isoformat(),
        selected_layer=selection['layer_zero_based'],point=1,dose=1.,dimension=1,
        selection=selection,selection_sha256=sha(DIAG/'selection.json'),
        classification='Exploratory causaldiagnostic; historicaldevelopment fixed beforefournewtestoutcomes. Preserveweakmanipulation/adverseeffects.',
        stage_boundary='Complete andstop withinmodelstage1 evenifpositive; nextcross-modelphase requiresseparateauthorization.')
    write(path,protocol)
    files=[*PACKAGE.glob('*.py'),DIAG/'summarize_results.py',DIAG/'selection.json',DIAG/'development_geometry.csv']
    files += [DIAG/'data'/u['unit']/'completion.json' for u in source['units']]
    write(DIAG/'freeze.json',dict(protocol_sha256=sha(path),created_utc=datetime.now(timezone.utc).isoformat(),
        sources={str(p.relative_to(ROOT)):sha(p) for p in files},before_any_formal_test_forward_or_count_scoring=True))
    return protocol


def verify_formal():
    verify_engineering()
    frozen=read(DIAG/'freeze.json')
    assert sha(DIAG/'protocol.json')==frozen['protocol_sha256']
    for path,digest in frozen['sources'].items():assert sha(ROOT/path)==digest,('Formal dependency changed',path)
    return read(DIAG/'protocol.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['engineering-freeze','calibrate','freeze','historical','new','all'])
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    with threadpool_limits(limits=2):
        if args.stage=='engineering-freeze':engineering_freeze();return
        if args.stage=='calibrate':
            from .calibrate import run
            run();return
        if args.stage=='freeze':formal_freeze();return
        protocol=verify_formal()
        from .formal import execute
        source=read(DIAG/'source_audit.json')
        units=source['units'] if args.stage=='all' else source['units'][:1] if args.stage=='historical' else source['units'][1:]
        for unit in units:verify_formal();execute(unit,protocol)
        write(DIAG/'run_completion.json',dict(status='completed',stage=args.stage,units=[u['unit'] for u in units],
              no_base_training=True,stopped_after_within_model_stage=True))


if __name__=='__main__':main()
