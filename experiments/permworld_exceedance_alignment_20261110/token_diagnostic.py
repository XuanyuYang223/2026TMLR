"""Separately frozen explicit-edge diagnostic using existing GRU checkpoints.

Run ``python -m experiments.permworld_exceedance_alignment_20261110.token_diagnostic all``.
No original checkpoint or old result is modified. No optimizer is constructed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import sys
import time

import numpy as np
from scipy.linalg import svd
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from .shared import DIAG, OUT, PACKAGE, VENDOR, ENCODINGS, LENGTHS, SHUFFLE_SEEDS
from .shared import sha, array_sha, read, write, write_csv, load_data, split_rows, vendor
from .token_features import BASE, CONDITIONS, KIND_NAMES, points, native_permutations
from .token_features import load_states, extract_condition, preflight

RESULTS = DIAG/'token_results'
ALPHAS = (1., 10., 100., 1000.)


def finite_mean(value):
    value = np.asarray(value)
    return float(value[np.isfinite(value)].mean()) if np.isfinite(value).any() else np.nan


def masked_auc(scores, targets, mask=None):
    """Per-anchor AUROC; a stratum lacking either class remains undefined."""
    scores, targets = np.asarray(scores), np.asarray(targets, dtype=bool)
    if mask is None:
        mask = np.ones_like(targets, dtype=bool)
    valid = np.where(mask, scores, np.nan)
    ranks = rankdata(valid, axis=1, method='average', nan_policy='omit')
    pos = (targets & mask).sum(1)
    neg = ((~targets) & mask).sum(1)
    numerator = np.nansum(ranks*targets, axis=1)-pos*(pos+1)/2
    return np.divide(numerator, pos*neg, out=np.full(len(scores), np.nan), where=pos*neg>0)


def fit_scalar(states, labels, fit, dev):
    d = states.shape[-1]
    scaler = StandardScaler().fit(np.asarray(states[fit]).reshape(-1,d))
    xf = scaler.transform(np.asarray(states[fit]).reshape(-1,d)).astype(np.float64)
    xd = scaler.transform(np.asarray(states[dev]).reshape(-1,d)).astype(np.float64)
    best, records = None, []
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver='cholesky').fit(xf, labels[fit].reshape(-1).astype(np.float64))
        auc = finite_mean(masked_auc(model.predict(xd).reshape(labels[dev].shape), labels[dev]))
        records.append(dict(alpha=alpha, dev_per_anchor_auc=auc))
        if np.isfinite(auc) and (best is None or auc > best[0]):
            best = (auc, alpha, model)
    if best is None:
        raise ValueError('All scalar source-dev AUROCs undefined')
    auc, alpha, model = best
    weight = model.coef_/scaler.scale_
    bias = float(model.intercept_-scaler.mean_@weight)
    check = np.asarray(states[dev]).reshape(-1,d).astype(np.float64)
    np.testing.assert_allclose(check@weight+bias, model.predict(scaler.transform(check)), atol=2e-12, rtol=0)
    return dict(weight=weight, bias=bias, alpha=alpha, dev_auc=auc,
                scaler_mean=scaler.mean_, scaler_scale=scaler.scale_, scaler_var=scaler.var_,
                ridge_coef=model.coef_, ridge_intercept=np.asarray(model.intercept_)), records


def predict(probe, states):
    # Match the pinned StandardScaler.transform path exactly: its in-place
    # subtraction/division casts fitted parameters to the incoming dtype.
    value=np.array(states,copy=True)
    value-=probe['scaler_mean'].astype(value.dtype)
    value/=probe['scaler_scale'].astype(value.dtype)
    return value.astype(np.float64)@probe['ridge_coef']+probe['ridge_intercept']


def pull_states(states, permutation):
    """Native inverse edge j -> source edge i exactly once."""
    return np.take_along_axis(states, (permutation-1)[...,None,None], axis=1)


def rotation(cross):
    left, _, right = svd(cross, full_matrices=False)
    result = left@right
    np.testing.assert_allclose(result.T@result, np.eye(result.shape[0]), atol=2e-12, rtol=0)
    return result


def synthetic_checks():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(200,8))
    q, _ = np.linalg.qr(rng.normal(size=(8,8)))
    y = x@q
    r = rotation(y.T@x)
    np.testing.assert_allclose(y@r,x,atol=2e-12,rtol=0)
    lab = rng.integers(0,2,size=(20,10)).astype(bool)
    scores = rng.normal(size=lab.shape)
    auc = masked_auc(scores,lab)
    for i in range(len(lab)):
        if lab[i].any() and (~lab[i]).any():
            np.testing.assert_allclose(auc[i],roc_auc_score(lab[i],scores[i]),atol=1e-14,rtol=0)
    assert np.isnan(masked_auc(scores,lab,np.zeros_like(lab))).all()
    p = np.array([[2,3,1]])
    s = np.arange(1*3*2*4).reshape(1,3,2,4)
    assert np.array_equal(pull_states(s,p),s[:,[1,2,0]])
    fake = rng.normal(size=(20,10,8)).astype(np.float32)
    probe, _ = fit_scalar(fake,lab,np.arange(20)<14,np.arange(20)>=14)
    assert predict(probe,fake).shape == lab.shape
    return dict(status='passed', orientation=True, scalar_serialization=True,
                masked_auroc=True, inverse_order_once=True, no_checkpoint_forward=True)


def freeze():
    RESULTS.mkdir(parents=True, exist_ok=True)
    supplement = dict(version='token-stage-supplement/v1', created_utc=datetime.now(timezone.utc).isoformat(),
        status='frozen_before_any_token_forward_fit_or_outcome',
        supplemental_controls=['both_internal stratum', 'fit-only native token-kind priors'],
        both_internal='source cycle_internal AND paired inverse-target cycle_internal; original overall metric stays primary',
        kind_prior='source training-fit E fraction by source edge kind; target validation-fit D fraction by native identity kind, evaluated at native inverse kind then gathered once',
        undefined='single-class and missing-length aggregates remain undefined, not zero; no weight renormalization',
        before='AFTER-only scalar probe, scaler and centering means applied unchanged to BEFORE; before can already informative')
    supplement_path = DIAG/'token_stage_supplement.json'
    if not supplement_path.exists():
        write(supplement_path,supplement)
    files = [PACKAGE/'token_features.py',PACKAGE/'token_diagnostic.py',PACKAGE/'oracles.py',PACKAGE/'shared.py',
             DIAG/'token_stage_spec.json',supplement_path,DIAG/'protocol.json',DIAG/'cache_audit.json',
             VENDOR/'infer.py',VENDOR/'archdev_common.py',VENDOR/'encoding_core.py',VENDOR/'analyze.py']
    hashes = {str(p):sha(p) for p in files}
    target = RESULTS/'freeze.json'
    if target.exists():
        assert read(target)['source_sha256'] == hashes, 'Token source changed after freeze'
    else:
        write(target,dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=hashes,
                          synthetic_checks=synthetic_checks(),no_token_outcomes_before_freeze=True))
    return read(target)


def verify_freeze():
    frozen = read(RESULTS/'freeze.json')
    for path, digest in frozen['source_sha256'].items():
        assert sha(path)==digest, ('Frozen token dependency changed',path)


def fitting(unit, encoding, splits):
    _, _, _, _, _, _, shuffle = vendor()
    rngs = {seed:np.random.default_rng(seed) for seed in SHUFFLE_SEEDS}
    names = ['target_identity','target_complement','target_inverse',
             *[f'shuffle_{s}' for s in SHUFFLE_SEEDS], 'initial_inverse','fixed_inverse']
    cross = {name:np.zeros((256,256)) for name in names}
    rows, fits, shuffle_records = {}, [], []
    save = BASE/'parameters'/unit['unit']/encoding
    save.mkdir(parents=True,exist_ok=True)
    for n in LENGTHS:
        split=splits[n]
        fit,dev=split['fit'],split['dev']
        x=load_states(unit['unit'],encoding,'source','validation',n)
        source, grid=fit_scalar(x[:,:,1],split['source_val_labels'],fit,dev)
        y0=load_states(unit['unit'],encoding,'target_identity','validation',n)
        target, own_grid=fit_scalar(y0[:,:,1],split['target_val_labels'],fit,dev)
        np.savez(save/f'n{n}_probes.npz',**{f'source_{k}':v for k,v in source.items()},
                 **{f'target_{k}':v for k,v in target.items()})
        for role, g in [('source',grid),('target',own_grid)]:
            for item in g:
                fits.append(dict(unit=unit['unit'],encoding=encoding,n=n,role=role,**item,
                                 selected=item['alpha']==(source if role=='source' else target)['alpha']))
        xf=np.asarray(x[fit,:,1],dtype=np.float64)
        mx=xf.mean((0,1))
        cx=(xf-mx).reshape(-1,256)
        means={}
        for condition in ('target_identity','target_complement','target_inverse','initial_inverse','fixed_inverse'):
            y=load_states(unit['unit'],encoding,condition,'validation',n)
            yf=np.asarray(y[fit,:,1],dtype=np.float64)
            if condition.endswith('inverse'):
                yf=np.take_along_axis(yf,(split['val_permutations'][fit]-1)[...,None],axis=1)
            mu=yf.mean((0,1))
            means[condition]=mu
            cy=yf-mu
            cross[condition]+=cy.reshape(-1,256).T@cx/(fit.sum()*n)
            if condition=='target_inverse':
                counts=split['source_val_labels'][fit].sum(1)
                for seed,rng in rngs.items():
                    order=shuffle(counts,rng)
                    cross[f'shuffle_{seed}']+=cy[order].reshape(-1,256).T@cx/(fit.sum()*n)
                    np.save(save/f'n{n}_shuffle_{seed}.npy',order,allow_pickle=False)
                    shuffle_records.append(dict(unit=unit['unit'],encoding=encoding,n=n,seed=seed,
                                                anchors=len(order),moved=int(np.count_nonzero(order!=np.arange(len(order))))))
        source_coords=points(split['val_permutations'],encoding)
        native_kind=source_coords['kind']
        kind_source=np.array([finite_mean(split['source_val_labels'][fit][native_kind[fit]==k]) for k in range(len(KIND_NAMES))])
        kind_target=np.array([finite_mean(split['target_val_labels'][fit][native_kind[fit]==k]) for k in range(len(KIND_NAMES))])
        rows[n]=dict(probe=source,own=target,mx=mx,my=means,
                     source_prior=split['source_val_labels'][fit].mean(0),target_prior=split['target_val_labels'][fit].mean(0),
                     kind_source=kind_source,kind_target=kind_target)
        np.savez(save/f'n{n}_means_priors.npz',source_mean=mx,**{f'{k}_mean':v for k,v in means.items()},
                 source_prior=rows[n]['source_prior'],target_prior=rows[n]['target_prior'],
                 source_kind_prior=kind_source,target_kind_prior=kind_target)
    maps={name:rotation(value) for name,value in cross.items()}
    np.savez(save/'maps.npz',**maps)
    return rows,maps,fits,shuffle_records


def evaluate(unit,encoding,splits,rows,maps):
    _,_,_,_,count_prior,_,_=vendor()
    records=[]
    score_dir=BASE/'scores'/unit['unit']/encoding
    score_dir.mkdir(parents=True,exist_ok=True)
    for n in LENGTHS:
        split,row=splits[n],rows[n]
        p=split['test_permutations']; y=split['source_test_labels']
        x=load_states(unit['unit'],encoding,'source','test',n)
        yi=pull_states(load_states(unit['unit'],encoding,'target_inverse','test',n),p)
        csource=points(p,encoding)
        ctargetnative=points(native_permutations(p,2),encoding)
        ctarget={k:np.take_along_axis(v,p-1,axis=1) for k,v in ctargetnative.items()}
        probe=row['probe']; mx=row['mx']; mi=row['my']['target_inverse']
        scores={
            'source_own':predict(probe,x),
            'direct':predict(probe,yi),
            'means_only':predict(probe,yi-mi+mx),
            'target_own':predict(row['own'],yi),
        }
        prior=np.broadcast_to(row['source_prior'],y.shape)
        tprior=np.take_along_axis(np.broadcast_to(row['target_prior'],y.shape),p-1,axis=1)
        oracle_prior=count_prior(split['source_val_labels'][split['fit']],y)
        for name,val in [('source_n_i_prior',prior),('target_n_j_prior',tprior),
                         ('oracle_count_position_prior',oracle_prior),
                         ('source_token_kind_prior',row['kind_source'][csource['kind']]),
                         ('target_token_kind_prior',row['kind_target'][ctarget['kind']])]:
            # A source/target kind absent from validation fit is undefined, never assigned zero.
            scores[name]=np.repeat(val[:,:,None],2,axis=2)
        for name,r in maps.items():
            if name in ('initial_inverse','fixed_inverse'):
                states=pull_states(load_states(unit['unit'],encoding,name,'test',n),p)
                mu=row['my'][name]
            else:
                states,mu=yi,mi
            # Scalar probe linearity avoids materializing a rotated [N,n,2,256] tensor.
            rw=r@probe['weight']
            scores[f'fit_{name}']=np.asarray(states,dtype=np.float64)@rw+float(mx@probe['weight']-mu@rw+probe['bias'])
        all_scores=np.stack(list(scores.values())) # method,anchor,edge,point
        assert np.isfinite(all_scores[:4]).all()
        path=score_dir/f'n{n}.npz'
        np.savez_compressed(path,scores=all_scores,methods=np.asarray(list(scores)),points=np.array(['before','after']),
             anchor_indices=split['test_indices'],source_position=np.arange(1,n+1),target_position=p,
             source_before_token=csource['before'],source_after_token=csource['after'],
             target_before_token=ctarget['before'],target_after_token=ctarget['after'],
             source_kind=csource['kind'],target_kind=ctarget['kind'],kind_names=np.array(KIND_NAMES),
             labels=y,eligible=split['eligible'],matched=split['matched'])
        strata={'all':np.ones_like(y,dtype=bool)}
        for side,coords in [('source',csource),('target',ctarget)]:
            for code,kind in enumerate(KIND_NAMES):
                strata[f'{side}_{kind}']=coords['kind']==code
        strata['both_internal']=(csource['kind']==1)&(ctarget['kind']==1)
        for name,values in scores.items():
            for point,point_name in enumerate(('before','after')):
                value=values[:,:,point]
                for stratum,edge_mask in strata.items():
                    valid_mask=edge_mask&np.isfinite(value)
                    auc=masked_auc(value,y,valid_mask)
                    for support in ('eligible','matched'):
                        use=split[support]
                        edges=valid_mask&use[:,None]
                        truth=y[edges]
                        pooled=float(roc_auc_score(truth,value[edges])) if len(truth) and truth.any() and (~truth).any() else np.nan
                        records.append(dict(unit=unit['unit'],historical=unit['historical'],encoding=encoding,n=n,
                            support=support,stratum=stratum,method=name,point=point_name,anchors=int(use.sum()),
                            finite_anchors=int(np.isfinite(auc[use]).sum()),edges=int(edges.sum()),
                            per_anchor_auc=finite_mean(auc[use]),pooled_edge_auc=pooled,
                            undefined_anchors=int((~np.isfinite(auc[use])).sum())))
        print(f'TOKEN SCORE {unit["unit"]} {encoding} n={n}',flush=True)
    return records


def summaries(records):
    import pandas as pd
    frame=pd.DataFrame(records)
    keys=['unit','historical','encoding','support','stratum','method','point']
    rows=[]
    for key,group in frame.groupby(keys,sort=False):
        assert len(group)==21
        row=dict(zip(keys,key))
        for metric in ('per_anchor_auc','pooled_edge_auc'):
            values=group[metric].to_numpy()
            row[metric]=float(values.mean()) if np.isfinite(values).all() else np.nan
            row[f'{metric}_finite_lengths']=int(np.isfinite(values).sum())
        rows.append(row)
    return rows


def summarize_all(units):
    import pandas as pd
    frames=[pd.read_csv(RESULTS/u['unit']/'summary.csv') for u in units
            if (RESULTS/u['unit']/'completion.json').exists()]
    if not frames:
        return
    frame=pd.concat(frames,ignore_index=True)
    frame.to_csv(RESULTS/'per_unit.csv',index=False)
    cohorts=[]
    for cohort,use in [('new_four',~frame['historical']),('all_five',np.ones(len(frame),dtype=bool))]:
        for key,g in frame.loc[use].groupby(['encoding','support','stratum','method','point'],sort=False):
            values=g['per_anchor_auc'].to_numpy()
            finite=np.isfinite(values)
            cohorts.append(dict(cohort=cohort,**dict(zip(['encoding','support','stratum','method','point'],key)),
                available_units=len(g),finite_units=int(finite.sum()),
                mean=float(values.mean()) if finite.all() else np.nan,
                sample_sd=float(values.std(ddof=1)) if finite.all() and len(values)>1 else np.nan,
                minimum=float(values.min()) if finite.all() else np.nan,
                maximum=float(values.max()) if finite.all() else np.nan))
    write_csv(RESULTS/'cohort_summary.csv',cohorts)
    lines=['# Explicit-edge token diagnostic','',
           'This stage uses a newly fitted **scalar shared probe**, separately from the prefix-final multi-output probe.',
           'All base networks are frozen. The same source AFTER probe and AFTER centering are used before and after; BEFORE is a distribution-shift diagnostic, not proof of absent information.',
           'Native inverse target edge j=pi(i) is gathered once. Transferred scores already have source i indexing.',
           'Primary AUROC averages anchors within length and then uses all 21 lengths with fixed 1/21 weights. Missing lengths propagate undefined; no zero imputation.',
           'Pooled-edge AUROC is supplementary. Canonical source closures are all E=0, inverse-target closures all paired E=1, and singletons E=0; their individual AUROCs are structurally undefined.',
           'The both-internal and token-kind-prior controls are frozen supplemental analyses, not the primary endpoint.','',
           'Full machine-readable results: `per_unit.csv`, `cohort_summary.csv`, per-unit `by_length.csv`; score tensors, probes, maps and shuffles are on the external F drive.','',
           '| Unit | Encoding | Point | Source own | Direct | Fit inverse | Best wrong map | Initial target | Fixed target | Target own |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for (u,e,point),g in frame[(frame.support=='eligible')&(frame.stratum=='all')].groupby(['unit','encoding','point'],sort=False):
        vals=g.set_index('method').per_anchor_auc
        wrong=[vals.get('fit_target_identity',np.nan),vals.get('fit_target_complement',np.nan),
               *[vals.get(f'fit_shuffle_{s}',np.nan) for s in SHUFFLE_SEEDS]]
        fmt=lambda v:f'{v:.4f}' if np.isfinite(v) else 'undefined'
        numbers=[vals.get(m,np.nan) for m in ['source_own','direct','fit_target_inverse']]
        numbers += [max(wrong),vals.get('fit_initial_inverse',np.nan),vals.get('fit_fixed_inverse',np.nan),vals.get('target_own',np.nan)]
        lines.append(f'| {u} | {e} | {point} | '+' | '.join(map(fmt,numbers))+' |')
    (RESULTS/'reader_report.md').write_text('\n'.join(lines)+'\n')


def run_unit(unit):
    folder=RESULTS/unit['unit']
    if (folder/'completion.json').exists():
        complete=read(folder/'completion.json')
        assert complete['token_freeze_sha256']==sha(RESULTS/'freeze.json')
        for path,digest in complete['result_sha256'].items():
            assert sha(path)==digest
        print(f'TOKEN REUSE {unit["unit"]}',flush=True)
        return
    started=time.perf_counter()
    data=load_data(unit)
    splits=split_rows(data)
    for encoding in ENCODINGS:
        for condition in CONDITIONS:
            extract_condition(unit,data,encoding,condition)
    records,fits,shuffles=[],[],[]
    with threadpool_limits(limits=3):
        for encoding in ENCODINGS:
            rows,maps,fit_records,shuffle_records=fitting(unit,encoding,splits)
            fits.extend(fit_records); shuffles.extend(shuffle_records)
            records.extend(evaluate(unit,encoding,splits,rows,maps))
    folder.mkdir(parents=True,exist_ok=True)
    for name,rows in [('by_length',records),('summary',summaries(records)),('probe_fit',fits),('shuffle_fit',shuffles)]:
        write_csv(folder/f'{name}.csv',rows)
    write(folder/'completion.json',dict(status='complete',unit=unit['unit'],elapsed_seconds=time.perf_counter()-started,
          data_sha256=unit['data_sha256'],token_freeze_sha256=sha(RESULTS/'freeze.json'),
          result_sha256={str(p):sha(p) for p in folder.glob('*.csv')},
          no_base_training=True,external_features=str(BASE/'features'/unit['unit']),
          external_scores=str(BASE/'scores'/unit['unit']),external_parameters=str(BASE/'parameters'/unit['unit'])))
    print(f'TOKEN UNIT COMPLETE {unit["unit"]} {time.perf_counter()-started:.1f}s',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','check','preflight','all','summarize'])
    parser.add_argument('--units',nargs='*')
    args=parser.parse_args()
    if args.command=='freeze':
        freeze(); print('TOKEN FROZEN'); return
    if args.command=='check':
        print(synthetic_checks()); return
    verify_freeze()
    assert (DIAG/'main_run_completion.json').exists() and (DIAG/'summary/completion.json').exists(), 'Main stage must complete first'
    units=read(DIAG/'cache_audit.json')['units']
    selected=[u for u in units if args.units is None or u['unit'] in args.units]
    assert selected
    if args.command=='summarize':
        summarize_all(units); return
    if not (RESULTS/'preflight.json').exists():
        result=preflight(units[0],load_data(units[0]))
        print(result,flush=True)
    if args.command=='preflight':
        return
    for unit in selected:
        verify_freeze();run_unit(unit);summarize_all(units)
    write(RESULTS/'run_completion.json',dict(status='complete',units=[u['unit'] for u in selected],
        token_freeze_sha256=sha(RESULTS/'freeze.json'),completed_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    main()
