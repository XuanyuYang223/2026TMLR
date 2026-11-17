"""Freeze all reused sources and timing rules before new formal outcomes."""
import argparse
from datetime import datetime,timezone

import torch
from threadpoolctl import threadpool_limits

from .shared import ROOT,PACKAGE,DIAG,OUT,PRIOR,TASKS,SEEDS,COMMON,AUX_LATE,sha,read,write


def freeze():
    path=DIAG/'protocol.json'
    if path.exists():return verify()
    audit=read(DIAG/'input_audit.json')
    assert audit['status']=='passed'
    selection=read(PRIOR/'selection.json')
    assert selection['layer_zero_based']==0 and selection['point']==1 and selection['dose']==1
    source=read(PRIOR/'source_audit.json')
    p=dict(version='local-current-token-interlayer-timing/v1',created_utc=datetime.now(timezone.utc).isoformat(),
        exploratory=True,prior_stage_results_seen=True,new_formal_timing_outcomes_seen=False,
        git_head=source['git_head'],input_audit_sha256=sha(DIAG/'input_audit.json'),
        parent_protocol_sha256=sha(PRIOR/'protocol.json'),model=source['model'],units=source['units'],output=str(OUT),
        tasks=list(TASKS),lengths=list(range(10,31)),encoding='one_line',layer=1,dimension=1,dose=1.,point=1,
        sample_policy='Reuse every prior test pair/ID/input/split/complement/direction in all5units; historicalreferenceexcludedfromfour-unitmeans. No newpairs/resampling/outcome-basedexclusion.',
        readouts='Load exact saved edit/checker/prefixcounter weights, scalers,biases,direction and eight random directions; no refitting,alpha/threshold/dose/layer selection. Checkerboundary.5.',
        carry_late='Afterallfourlayersprocesscurrentright-valuetokent=4+2i,replaceL1componentandretainrecipientL2–4statesat t. Resume originaltailatt+1.',
        carry_early='UseTHESAMEFP32editedh1(t). RecomputeONLYL2,L3,L4at t sequentially, each seededbyitsownrecipienth_l(t−1), usingnewlowercurrentoutput; doNOTembed/processvalueagainthroughL1. Carryrecomputedfourstatesintot+1.',
        invariance='Primaryearly/lateh1(t)bit-identicalforeachpair/condition; sameindependentchecker/edit-probescores andinjectedL1L2norm. Fullfour-stateeditnormmaydifferbecauseupperreplayisintended; no rescalingofearlyupperstates.',
        common_conditions=list(COMMON),common_timings=['late','early'],random_seeds=list(SEEDS),
        auxiliary_late=list(AUX_LATE),auxiliary_early='early_full_layer1',
        positives=['full_state_swap','actual_counterfactual'],
        condition_inventory='29rowsperpair:11commonconditions×2timings+4latelayerswaps+1earlyfullL1+2separatereferences. Streamcase-majorwithinmonotonictask,n,i blocks.',
        single_layer_late='ReplaceoneWHOLEdonorrecurrenth_l(t),l1..4,keeprecipientothers. Sufficientinthismixedstateonly; notcausalaccumulationlocation/necessity.',
        full_layer_early='Replacewholeh1(t)withdonorh1(t); recipientpreviousL2–4seedscurrentreplay. H1identicaltolate_layer_swap_1. Thisisnotafullcounterfactualprefixandneednotmatchdonorwholecontinuation.',
        references='Full4layerstateswapverifiedagainstnativeDONORlogits. RealCFinputusespreviouslyverifiednativeCFlogits. Matchedpairlocal-onlygoalALWAYSrecipienttruecount+sign; fullswapdonortruthmayremainrecipientcount.',
        baseline='Sameunmodifiedlatecontinuationlogits/argmaxforallconditions. Earlyno-op isaseparatewiring/referencecheck; anyearlyno-opmodaldriftreportednotusedtofilterpairs.',
        metrics=dict(relative_exact='Numericbothandedited_argmax_count−recipient_argmax_count==intendedsign; nonnumericfailuresretained.',
                     other_change='Numericboth,nonzerodeltaANDnotintendedsign; alsoanynumericchangeandfullargmaxtokenchanges.',
                     oracle='Editednumericcount==recipienttruecount+intendedsign; distinctfromrelativeexact.',
                     expected='Intendedsign×(edited−baseline)conditionalnumericexpectationover0..30; fullvocabnumericmassandnonnumericcoverage retained.',
                     checker='IndependentFIT_Bclassifiertruerecipient-to-donorboundarycrossing; unchangedreadoutandh1acrosstimings,notindependentreplications.'),
        primary_comparisons='WithinpairEARLY−LATEforlocaledit; DiDsubtractsame-labeltimingdifferenceandmean8randomtimingdifference(averagewithinpairfirst); controlsnotmaximized. Fullrawconditionaltables.',
        statistics='Task/group/signseparate. Fixed21equal-lengthmeans afterequalpredeclaredcenterswithinlength (main2..n−2,matched6..n−2). Pairedcasecontrastsbeforeaggregation. Missingcellsundefined; sensitivityavailable/pooledexplicit. Fourunitrawvalues,mean,sampleSD,minmax,directions; pairs/positions/conditionsnotindependentunits.',
        judgment='Supportcurrent-stepinterlayerrouteonlyiflocalmanipulationreliableandmatchedanticipatedmodalresponsesincreaseconsistentlybeyondsame-label/randomcontrolswithreportedeffectsize. Positivecountprobabilitydriftalonewithoutmodalchangeisnotasuccess. OnlyfullL1successful=>testedinterfacecanconveychangebut1Deditdoesnotreproduceit. Bothnull=>scopeboundnegative; noequivalenceorabsenceofearlieruseclaim.',
        numerical=dict(state_noop_absolute=2e-5,logits_noop_old_late_and_donor_absolute=1e-4,
            same_h1='ExactFP32byteidentity',paired_readout_absolute=1e-10,planned_norm_absolute=1e-10,
            injected_L1_norm_absolute=1e-5,injected_L1_norm_relative=1e-5,
            model_dtype='float32',scoring_dtype='float64'),
        runtime=dict(threads=2,cuda_matmul_tf32=False,cudnn_tf32=False,cudnn_deterministic=True),
        limitations='Quartetlabelcoupledtoadjacentorder/tokenrank; mixedstatescanbeoffmanifold. Earlylatecontrastidentifiescurrentinterlayerpathunderthisintervention,notpreciseaccumulationtime/naturalone-dimensionalmechanism. Auxiliarysinglelayerswapdoesnotlocatenecessarycounter.',
        boundary='Onlytwo within-model experiments. Nofits/training/oldresultwrites/DAS/crossmodelexchange/newtaskgeneralization/publication.')
    write(path,p)
    files=[*PACKAGE.glob('*.py'),DIAG/'input_audit.json',DIAG/'summarize_results.py',
           ROOT/'tests/test_permworld_local_timing_replay.py',PRIOR/'protocol.json',PRIOR/'selection.json',
           ROOT/'experiments/permworld_local_causal_20261110/formal.py',
           ROOT/'experiments/permworld_local_causal_20261110/recurrent.py',
           ROOT/'experiments/permworld_local_causal_20261110/probes.py',
           ROOT/'experiments/permworld_local_causal_20261110/shared.py',
           ROOT/'experiments/permworld_local_causal_20261110/pairs.py']
    for f in files:assert f.is_file(),f
    write(DIAG/'freeze.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(path),
          sources={str(f.relative_to(ROOT)):sha(f) for f in files},before_new_formal_outcomes=True))
    return p


def verify():
    f=read(DIAG/'freeze.json')
    assert sha(DIAG/'protocol.json')==f['protocol_sha256']
    for path,digest in f['sources'].items():assert sha(ROOT/path)==digest,('Frozen source changed',path)
    return read(DIAG/'protocol.json')


def setup():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','historical','new','all'])
    args=parser.parse_args();setup()
    if args.stage=='freeze':freeze();return
    p=verify()
    from .experiment import execute
    selected=p['units'] if args.stage=='all' else p['units'][:1] if args.stage=='historical' else p['units'][1:]
    with threadpool_limits(limits=2):
        for unit in selected:verify();execute(unit,p)
    write(DIAG/'run_completion.json',dict(status='completed',stage=args.stage,units=[u['unit'] for u in selected],
        no_probe_fitting=True,no_base_training=True,no_extra_stage=True))


if __name__=='__main__':main()
