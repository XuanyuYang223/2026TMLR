"""Frozen final-state exceedance/inverse-deficiency transfer experiment.

Existing data, real checkpoints and prefix_final caches are read-only. Only
source-labelled ridge readouts and validation-only orthogonal maps are fitted.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import time

import numpy as np
from threadpoolctl import threadpool_limits

from .shared import (ROOT, PACKAGE, DIAG, OUT, PV, VENDOR, ALIGN_VENDOR,
                     ENCODINGS, LENGTHS, SHUFFLE_SEEDS, sha, array_sha, read,
                     write, write_csv, read_csv, vendor, load_data, raw_arrays,
                     split_rows)
from .oracles import pullback_target_owned_batch

METHODS = ['direct', 'means_only', 'within_source', 'within_target_reindexed',
           'source_position_prior', 'target_position_prior_reindexed',
           'oracle_count_source_position_prior', 'oracle_count_target_position_prior_reindexed',
           *[f'fit_{a}__test_inverse' for a in ['identity', 'complement', 'inverse']],
           *[f'fit_shuffle_{seed}__test_inverse' for seed in SHUFFLE_SEEDS],
           'aligned_step0_target', 'aligned_fixed_points_target']


def freeze():
    if (DIAG/'protocol.json').exists():
        return config()
    audit = read(DIAG/'cache_audit.json')
    assert audit['status'] == 'passed' and not audit['required_missing_files']
    math = read(DIAG/'math_design_audit.json')
    assert math['status'] == 'passed'
    base = read(ROOT/'diagnostics/permworld_repetition_20261106/protocol.json')
    c = dict(
        version='exceedance-deficiency-orthogonal-transfer/v1',
        created_utc=datetime.now(timezone.utc).isoformat(), exploratory=True,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        output=str(OUT), units=[dict(unit=u['unit'], historical=u['historical'],
                                  data_path=u['data_path'], data_sha256=u['data_sha256']) for u in audit['units']],
        model=base['model'], architecture='gru', encodings=list(ENCODINGS),
        lengths=list(LENGTHS), length_weights={str(n):1/21 for n in LENGTHS},
        actions=['identity', 'complement', 'inverse'], shuffle_seeds=list(SHUFFLE_SEEDS),
        source='exceedances', target='deficiencies', correct_transform='inverse', wrong_transform='complement',
        labels='E_i(pi)=[pi_i>i], D_j(invpi)=[invpi_j<j]; j=pi_i, fixed points negative. Transferred source vector already indexes i; own target scores/priors gathered by pi_i-1 exactly once.',
        probe='Original fit_probe unchanged: one vector-valued linear Ridge per n/source task, StandardScaler fitted only source validation-fit; alphas1,10,100,1000 selected by source validation-dev per-example position AUROC.',
        split='Reuse original read_length orbit keys and source validation-fit/dev/test exclusion exactly. Encodings and mapping controls share anchor IDs. No new data generation.',
        matched='Sensitivity: eligible anchors where E(pi)count=D(pi)count=D(Cpi)count; inverse equality is exact. PV retains its original saved matched mask; pair-specific supports are not called identical samples across task pairs.',
        maps='One O(256) per encoding/unit/method; per-length fit-only means and cross-products divided by fit anchor count, then sum over all21 lengths. Three native target transform fits plus eight source-E-count-preserving shuffled correct-inverse fits. Same original shuffled_indices and persistent seed1026..1033 RNG stream in increasing n.',
        evaluation='All maps score SAME held-out deficiencies(invpi), centered by inverse fit mean and restored source mean; wrong-transform means are not substituted. Step0 and fixed_points controls align inverse features to SAME trained source with own inverse fit-only means. Source probe never refitted per transfer method.',
        priors='Source and native target unconditional position frequencies fitted only validation-fit. Native target prior gathered to i exactly once. Additional oracle-count priors use test true counts explicitly as diagnostic oracle references, not common-margin controls.',
        methods=METHODS,
        main_statistic='Per-sample position AUROC; within-length finite-sample mean; fixed1/21 length mean. Full eligible primary; original-style matched sensitivity. Undefined samples/counts retained. Any undefined length leaves aggregate undefined; no reweighting.',
        common_controls=['means_only', 'fit_identity', 'fit_wrong_transform', *[f'fit_shuffle_{s}' for s in SHUFFLE_SEEDS]],
        margin='Correct map minus maximum of11 individual common mapping controls AFTER equal-length averaging. Direct, within-model,priors,step0-target,fixed_points excluded from that maximum and reported separately.',
        cross_pair='PV margin(one_line)-PV margin(canonical_cycle); ED margin(canonical_cycle)-ED margin(one_line). Report both and joint positive direction per unit, newfour mean sampleSD minmax. No extra scientific repetitions from controls,lengths,positions,metrics or shared models.',
        comparison_pv='Reaggregate original trained PV per-length results under same rule; rebuild unsaved per-position scores with unchanged probes and saved maps, verify registered cells1e-12. Original historical unit reference only.',
        learning='Source/target trained count exact accuracy from cached raw-vocabulary predictions against own oracle counts; training-only per-length count-mode baseline; report identity and target inverse separately.',
        token_stage=dict(
            timing='Separate diagnostic only AFTER completed final-state cross-pair table; no replacement of main endpoint.',
            point='Final-normalized sequence hidden state immediately before and at local explicit-edge reveal. One-line value token; cycle second numeric endpoint, closing delimiter for closing edges/singletons. Separate source/target internal,closure,singleton. Not a claim of earliest logical entailment; canonical/permutation/position priors may inform before states.',
            scalar_probe='Per n, one source scalar Ridge over validation-fit anchors/all edge positions AFTER reveal; source-dev after-reveal per-anchor AUROC picks alpha in1,10,100,1000; same coefficients/scaler applied before and after, all methods. Anchor-orbit split retained, no position-level random splitting.',
            mapping='After-reveal state fits only; global256 raw-coordinate maps, equal-length normalized by fitanchors*n. Identity/complement target native i; inverse target j=pi_i; shuffle anchor pairs within source count. Same inverse test states at j for all maps. Fixed after-fit means used for before/after; state-distribution difference noted.',
            readouts='Own target scalar probe only target identity labels; reindex inverse to source i once. Source/target validation position priors retained. Scores, i/j, input IDs, edge kinds and before/after token offsets saved.',
            metrics='Per-anchor position AUROC, fixed21 equal means; supplementary pooled-edge AUROC per length and kind. All-negative/small strata undefined, coverage preserved; source and target edge kinds reported separately.',
            forwards='Only frozen existing checkpoints, eval/inference_mode, parameter hashes unchanged; source identity, targetI/C/Inv,initialInv,fixed_pointsInv. RawFP32 state shards external; endstate versus old prefix tolerance2e-5 fixed from old extraction protocol. No optimizer or backward.',
        ),
        numerical=dict(raw_dtype='float32', fit_dtype='float64', map_orthogonality_absolute=2e-12,
                       previous_pv_auroc_absolute=1e-12, token_prefix_absolute=2e-5),
        boundaries='All five units and adverse/undefined cells retained. No base training, adaptive model/seed/metric choices, F17/paper mutation, old output overwrite or automatic publication.',
        prior_pv_scores_seen=True, new_ed_test_scores_seen=False,
    )
    write(DIAG/'protocol.json', c)
    paths = [PACKAGE/name for name in ['__init__.py', 'shared.py', 'run.py', 'oracles.py', 'rebuild_pv_scores.py']]
    paths += [VENDOR/'analyze.py', ALIGN_VENDOR/'alignment.py', DIAG/'cache_audit.json', DIAG/'math_design_audit.json', DIAG/'token_stage_spec.json',
              DIAG/'summarize_results.py', ROOT/'tests/test_permworld_exceedance_oracles.py']
    for path in paths:
        assert path.is_file(), path
    write(DIAG/'freeze.json', dict(protocol_sha256=sha(DIAG/'protocol.json'),
         source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
         before_new_ed_probe_fits_and_test_scoring=True,
         token_code_separate_freeze_before_token_fits=True))
    protected = {p:r['sha256'] for p,r in audit['files_hashed'].items()}
    for previous in [PV/'delivery_manifest.json', ROOT/'diagnostics/permworld_lemma_composition_20261110/delivery_manifest.json']:
        for p,r in read(previous)['files'].items():
            protected[str(ROOT/p if not Path(p).is_absolute() else Path(p))] = r['sha256']
    for path in [ROOT/'README.md', ROOT/'CURRENT_RESEARCH.txt',
                 ROOT/'manuscripts/representation_selectivity_20261012/manuscript.tex',
                 ROOT/'manuscripts/representation_selectivity_20261012/manuscript.pdf']:
        protected[str(path)] = sha(path)
    write(DIAG/'protection_start.json', dict(files=protected, git_head=c['git_head'],
          scope='Start/end SHA check of explicitly named source caches/checkpoints, upstream delivery files and paper/index; not all files or continuous monitoring.'))
    return c


def config():
    c = read(DIAG/'protocol.json')
    frozen = read(DIAG/'freeze.json')
    assert sha(DIAG/'protocol.json') == frozen['protocol_sha256']
    for path,digest in frozen['source_sha256'].items():
        assert sha(ROOT/path) == digest, ('Frozen dependency changed', path)
    return c


def save_probe(probe, prefix, arrays):
    closure = dict(zip(probe.__code__.co_freevars, [cell.cell_contents for cell in probe.__closure__]))
    model, scaler = closure['best_model'], closure['scaling']
    for key, value in dict(coef=model.coef_, intercept=model.intercept_, mean=scaler.mean_, scale=scaler.scale_).items():
        arrays[prefix+'_'+key] = np.asarray(value).copy()


def aggregate(records):
    groups = {}
    for r in records:
        key = tuple(r[k] for k in ['encoding','phase','support','method'])
        groups.setdefault(key, []).append(r)
    result = []
    for key, values in groups.items():
        assert sorted(r['length'] for r in values) == list(LENGTHS)
        finite = sum(np.isfinite(r['auc']) for r in values)
        result.append(dict(zip(['encoding','phase','support','method'],key),
                           auc=float(np.mean([r['auc'] for r in values])) if finite == 21 else np.nan,
                           lengths=finite, anchors=sum(r['n'] for r in values),
                           finite=sum(r['finite'] for r in values), undefined_lengths=';'.join(str(r['length']) for r in values if not np.isfinite(r['auc']))))
    return result


def execute(unit):
    from .oracles import batch_labels
    c = config()
    destination = DIAG/'results'/unit['unit']
    external = OUT/'main'/unit['unit']
    done = destination/'completion.json'
    if done.exists():
        receipt = read(done)
        assert receipt['protocol_sha256'] == sha(DIAG/'protocol.json')
        for path,digest in receipt['files'].items():
            assert sha(path) == digest, path
        return receipt
    destination.mkdir(parents=True, exist_ok=True)
    external.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    data, splits = load_data(unit), None
    splits = split_rows(data)
    fit_probe, _, example_auc, average, position_prior, rotation, shuffled_indices = vendor()
    records, learning, coverage, fitting, coefficient_records = [], [], [], [], []
    artifacts = []
    for enc in ENCODINGS:
        raw = {}
        for label,task,phase in [('source','exceedances','trained'), ('target','deficiencies','trained'),
                                  ('initial','deficiencies','initial'), ('fixed','fixed_points','trained')]:
            for split in ['validation','test']:
                raw[label,split] = raw_arrays(unit, enc, task, phase, split)
        cross = {name:np.zeros((256,256),np.float64) for name in ['identity','complement','inverse',
                 *[f'shuffle_{s}' for s in SHUFFLE_SEEDS], 'step0', 'fixed']}
        rngs = [np.random.default_rng(seed) for seed in SHUFFLE_SEEDS]
        row_data, parameters = [], {}
        for n,row in splits.items():
            vi,ti,fit,dev = row['val_indices'],row['test_indices'],row['fit'],row['dev']
            x = raw['source','validation'][0][vi,0]
            ys = [raw['target','validation'][0][vi,a] for a in range(3)]
            ys += [raw['initial','validation'][0][vi,2], raw['fixed','validation'][0][vi,2]]
            source, alpha = fit_probe(x, row['source_val_labels'], fit, dev)
            own, own_alpha = fit_probe(ys[0], row['target_val_labels'], fit, dev)
            save_probe(source, f'source_n{n}', parameters)
            save_probe(own, f'target_n{n}', parameters)
            mx = x[fit].astype(np.float64).mean(0)
            my = [y[fit].astype(np.float64).mean(0) for y in ys]
            cx = x[fit]-mx
            cy = [y[fit]-mu for y,mu in zip(ys,my)]
            parameters[f'n{n}_source_mean'] = mx
            for name,mu in zip(['identity','complement','inverse','step0','fixed'],my):
                parameters[f'n{n}_{name}_mean'] = mu
            parameters[f'n{n}_fit_indices'] = vi[fit]
            parameters[f'n{n}_dev_indices'] = vi[dev]
            for name, y in zip(['identity','complement','inverse','step0','fixed'],cy):
                cross[name] += y.T@cx/fit.sum()
            counts = row['source_val_labels'][fit].sum(1)
            for seed,rng in zip(SHUFFLE_SEEDS,rngs):
                permutation = shuffled_indices(counts,rng)
                assert np.array_equal(counts[permutation],counts)
                parameters[f'n{n}_shuffle_{seed}_indices'] = vi[fit][permutation]
                parameters[f'n{n}_shuffle_{seed}_local'] = permutation
                cross[f'shuffle_{seed}'] += cy[2][permutation].T@cx/fit.sum()
                fitting.append(dict(unit=unit['unit'],encoding=enc,length=n,seed=seed,fit_n=int(fit.sum()),
                                    moved=int(np.count_nonzero(permutation!=np.arange(len(counts)))),
                                    counts_sha256=array_sha(counts),permutation_sha256=array_sha(permutation)))
            row_data.append(dict(n=n,row=row,probe=source,own=own,mx=mx,my=my))
            coefficient_records.append(dict(unit=unit['unit'],encoding=enc,length=n,source_alpha=alpha,target_alpha=own_alpha,
                                             fit_n=int(fit.sum()),dev_n=int(dev.sum())))
            if enc == ENCODINGS[0]:
                coverage.append(dict(unit=unit['unit'],length=n,fit_n=int(fit.sum()),dev_n=int(dev.sum()),
                                     eligible_n=int(row['eligible'].sum()),matched_n=int(row['matched'].sum()),
                                     excluded_validation_n=int(len(fit)-fit.sum()-dev.sum()),excluded_test_n=int(len(row['eligible'])-row['eligible'].sum())))
        maps = {name:rotation(value) for name,value in cross.items()}
        for matrix in maps.values():
            assert np.max(np.abs(matrix.T@matrix-np.eye(256))) <= 2e-12
        for name,matrix in maps.items():
            parameters[f'R_{name}'] = matrix
            parameters[f'cross_{name}'] = cross[name]
        size = len(data['test_lengths'])
        scores = np.full((len(METHODS),size,30),np.nan,np.float64)
        per_sample = np.full((len(METHODS),size),np.nan,np.float64)
        labels = np.zeros((size,30),bool)
        labels_native = np.zeros_like(labels)
        eligible, matched = np.zeros(size,bool),np.zeros(size,bool)
        for item in row_data:
            n,row,p,q,mx,my = item['n'],item['row'],item['probe'],item['own'],item['mx'],item['my']
            ti = row['test_indices']
            x = raw['source','test'][0][ti,0]
            y = raw['target','test'][0][ti,2]
            e,dt,pt = row['source_test_labels'],row['target_test_labels'],row['test_permutations']
            own_native = q(y)
            own_reindexed = pullback_target_owned_batch(own_native,pt)
            assert np.array_equal(pullback_target_owned_batch(dt[2],pt),e)
            np.testing.assert_allclose(example_auc(own_native,dt[2]),example_auc(own_reindexed,e),rtol=0,atol=1e-15,equal_nan=True)
            source_prior = np.broadcast_to(row['source_val_labels'][row['fit']].mean(0),e.shape)
            target_prior_native = np.broadcast_to(row['target_val_labels'][row['fit']].mean(0),e.shape)
            values = dict(direct=p(y),means_only=p(y-my[2]+mx),within_source=p(x),within_target_reindexed=own_reindexed,
                          source_position_prior=source_prior,
                          target_position_prior_reindexed=pullback_target_owned_batch(target_prior_native,pt),
                          oracle_count_source_position_prior=position_prior(row['source_val_labels'][row['fit']],e),
                          oracle_count_target_position_prior_reindexed=pullback_target_owned_batch(position_prior(row['target_val_labels'][row['fit']],dt[2]),pt))
            for name in ['identity','complement','inverse',*[f'shuffle_{s}' for s in SHUFFLE_SEEDS]]:
                values[f'fit_{name}__test_inverse'] = p((y-my[2])@maps[name]+mx)
            values['aligned_step0_target'] = p((raw['initial','test'][0][ti,2]-my[3])@maps['step0']+mx)
            values['aligned_fixed_points_target'] = p((raw['fixed','test'][0][ti,2]-my[4])@maps['fixed']+mx)
            assert set(values) == set(METHODS)
            labels[ti,:n],labels_native[ti,:n] = e,dt[2]
            eligible[ti],matched[ti] = row['eligible'],row['matched']
            for mi,method in enumerate(METHODS):
                auc = example_auc(values[method],e)
                scores[mi,ti,:n],per_sample[mi,ti] = values[method],auc
                for support in ['eligible','matched']:
                    selection = row[support]
                    finite = np.isfinite(auc[selection])
                    chosen = e[selection]
                    records.append(dict(unit=unit['unit'],encoding=enc,phase='trained',length=n,support=support,
                                        method=method,n=int(selection.sum()),finite=int(finite.sum()),
                                        undefined=int((~finite).sum()),all_negative_n=int((chosen.sum(1)==0).sum()),
                                        all_positive_n=int((chosen.sum(1)==n).sum()),auc=average(auc[selection])))
            for task,label,action,truth in [('exceedances','source',0,e),('deficiencies','target',0,dt[0]),('deficiencies','target',2,dt[2])]:
                train = data['train_permutations'][data['train_lengths']==n,:n]
                train_counts = batch_labels(train)[task].sum(1)
                mode = int(np.bincount(train_counts).argmax())
                predicted = raw[label,'test'][1][ti,action]
                selection = row['eligible']
                learning.append(dict(unit=unit['unit'],encoding=enc,length=n,task=task,transform=['identity','complement','inverse'][action],
                                     n=int(selection.sum()),accuracy=float(np.mean(predicted[selection]==truth[selection].sum(1))),
                                     baseline=float(np.mean(mode==truth[selection].sum(1))),train_mode=mode,
                                     within_position_auroc=average(example_auc(p(x) if label=='source' else q(raw['target','test'][0][ti,action]), truth)[selection])))
        checkpoint_info = {label:raw[label,'test'][2]['checkpoint_sha256'] for label in ['source','target','initial','fixed']}
        scorefile = external/f'{enc}-scores.npz'
        np.savez_compressed(scorefile,scores=scores,per_sample_auroc=per_sample,labels=labels,target_native_labels=labels_native,
                            lengths=data['test_lengths'],anchor_indices=np.arange(size),sample_ids=np.array([f'{unit["unit"]}:test:{i}' for i in range(size)]),
                            eligible=eligible,matched=matched,method_names=np.array(METHODS),data_sha256=np.asarray(unit['data_sha256']),
                            source_to_target=data['test_permutations']-1,checkpoint_bindings=np.asarray(str(checkpoint_info)),
                            valid_positions=np.arange(30)[None,:]<data['test_lengths'][:,None])
        paramfile = external/f'{enc}-parameters.npz'
        np.savez_compressed(paramfile,**parameters)
        artifacts.extend([scorefile,paramfile])
        write(external/f'{enc}-binding.json',dict(unit=unit['unit'],encoding=enc,data_sha256=unit['data_sha256'],
              checkpoint_bindings=checkpoint_info,output_scores_already_source_indexed=True,
              own_target_and_prior_reindexed_once=True,source_file_sha256=sha(scorefile),parameter_file_sha256=sha(paramfile),
              orthogonality={name:float(np.max(np.abs(r.T@r-np.eye(256)))) for name,r in maps.items()},
              no_neural_forward=True,no_base_training=True))
        artifacts.append(external/f'{enc}-binding.json')
        del raw,scores,parameters,per_sample,row_data
    for name,rows in [('by_length.csv',records),('summary.csv',aggregate(records)),('coverage.csv',coverage),
                       ('model_diagnostics_by_length.csv',learning),('probe_fit.csv',coefficient_records),('shuffle_fit.csv',fitting)]:
        path = destination/name
        write_csv(path,rows)
        artifacts.append(path)
    receipt = dict(status='completed',unit=unit['unit'],protocol_sha256=sha(DIAG/'protocol.json'),
                   elapsed_seconds=time.monotonic()-started,files={str(p):sha(p) for p in artifacts},
                   all_lengths=list(LENGTHS),methods=METHODS,no_neural_forward=True,no_base_training=True)
    write(done,receipt)
    print(dict(unit=unit['unit'],status='completed',elapsed_seconds=receipt['elapsed_seconds']),flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','historical','new','all','pv'])
    args = parser.parse_args()
    if args.stage == 'freeze':
        freeze()
        return
    config()
    units = read(DIAG/'cache_audit.json')['units']
    try:
        with threadpool_limits(limits=2):
            if args.stage in ['historical','all']:
                execute(units[0])
            if args.stage in ['new','all']:
                assert read(DIAG/'results'/units[0]['unit']/'completion.json')['status']=='completed'
                for unit in units[1:]:
                    execute(unit)
            if args.stage in ['all','pv']:
                from .rebuild_pv_scores import run_unit
                for unit in units:
                    run_unit(unit)
        write(DIAG/'main_run_completion.json',dict(status='completed',stage=args.stage,
              no_neural_forward=True,no_base_training=True,token_stage_not_started=True))
    except Exception as error:
        write(DIAG/'main_failure.json',dict(status='failed',stage=args.stage,type=type(error).__name__,error=str(error)))
        raise


if __name__ == '__main__':
    main()
