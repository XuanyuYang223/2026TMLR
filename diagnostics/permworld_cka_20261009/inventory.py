"""Read-only source audit and initialization-reference inference; never trains."""
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import tomllib

import numpy as np
import torch

from common import ROOT, WORKSPACE, REPO, sha, write_csv, write_json, weight_hash
from neurips_permutations.training import TrainConfig, _default_model_factory, _seed_everything
from neurips_permutations.cka import ProbeExample, select_probe_examples, probe_identity, extract_landmark_activations
from neurips_permutations.passage import TOKEN_TO_ID, ID_TO_TOKEN

STATUS = 'Reconstructed with the original initialization procedure; original weights not directly verifiable'
PARENT = WORKSPACE/'results/specialist_cka_controls'


def build(config):
    cfg = TrainConfig.from_value(config)
    cfg = replace(cfg, manifest=str(REPO/cfg.manifest))
    _seed_everything(cfg.seed)
    before = sha_bytes(torch.get_rng_state().numpy().tobytes())
    model = _default_model_factory(cfg).eval()
    after = sha_bytes(torch.get_rng_state().numpy().tobytes())
    return model, before, after


def sha_bytes(b):
    import hashlib
    return hashlib.sha256(b).hexdigest()


def samples(d, actions):
    ids = [int(np.flatnonzero((d['lengths']==n)&(d['split']==2))[0]) for n in [10,20,30]]
    examples = [ProbeExample(record_id=i*3+a, n=int(d['lengths'][i]),
                 token_ids=tuple(map(int,d['input'][i,a,:2*int(d['lengths'][i])+4])))
                for i in ids for a in actions]
    assert all(x.token_ids[-1]==TOKEN_TO_ID['<ONE_END>'] for x in examples)
    return ids, examples


def run():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sig = json.loads((PARENT/'protocol.json').read_text())['signature']
    d = dict(np.load(PARENT/'dataset.npz'))
    assert sha(PARENT/'dataset.npz') == 'e353558d7ac211637a9a783461b5eeb198dad6b99e95e2ec5946924f3280e65e'
    source_rows=[]; init_rows=[]; artifacts=[]; cache_checks=[]; config_rows=[]
    cache_by_cp={}; hashes={}; initial_models={}
    # All existing source checkpoints, including overlaps between specialist and k studies.
    jobs=[dict(s, family='specialist', model_id=f"{s['task']}_s{s['seed']}", pool='', k='', replicate='') for s in sig['sources']]
    suffixes=['pilot','r1','r2','r3','r4']
    for ri,suffix in enumerate(suffixes):
        cfgpath=REPO/f'configs/property32_zero_overlap_{suffix}.toml'
        cfg=tomllib.loads(cfgpath.read_text()); config_rows.append({'path':str(cfgpath),'sha256':sha(cfgpath)})
        cdir=REPO/'results/property32-zero-overlap'/('replicates' if ri<3 else 'subset-replicates')/f'r{ri}'/'cka'
        probe=json.loads((cdir/'probe_manifest.json').read_text())
        assert probe['probe_sha256']=='7df2ce6d75c95873061ac540ba6998f886921eaa0585d3ca0f14d6fec6fa7b91'
        for seed in cfg['model_seeds']:
            for k in cfg['task_subset_sizes']:
                for pool in ['a','b']:
                    runid=f'property-{pool}-transformer-tasks{k:02d}-seed{seed}'
                    cp=REPO/cfg['output_dir']/runid/'checkpoint.pt'; marker=cp.with_name('completed.json')
                    m=json.loads(marker.read_text())
                    jobs.append(dict(family='k_series',model_id=f'r{ri}/{runid}', task='',seed=seed,
                        pool=pool,k=k,replicate=f'r{ri}',checkpoint=str(cp),checkpoint_sha256=m['checkpoint_sha256'],
                        marker=str(marker),marker_sha256=sha(marker),activation_cache=str(cdir/'cache'/f'{runid}.pt')))
                    cache_by_cp[str(cp)]=str(cdir/'cache'/f'{runid}.pt')
    # Recover exactly the historical probe, without drawing a new dataset.
    manifest=REPO/'data/permutation-properties-16m-v1/validation_manifest.json'
    examples=select_probe_examples(manifest,count=4096,seed=20260902,shard_index=0)
    probe=probe_identity(examples,dataset_manifest_sha256=sha(manifest),shard_index=0,seed=20260902)
    oldprobe=json.loads((REPO/'results/property32-zero-overlap/replicates/r0/cka/probe_manifest.json').read_text())
    assert probe==oldprobe
    write_json(ROOT/'k_probe_manifest.json',probe)
    tok=np.full((len(examples),64),TOKEN_TO_ID['<PAD>'],np.int64)
    perms=np.zeros((len(examples),30),np.int64)
    for i,x in enumerate(examples):
        tok[i,:len(x.token_ids)]=x.token_ids
        perms[i,:x.n]=[int(ID_TO_TOKEN[t]) for t in x.token_ids[4:4+2*x.n:2]]
    ns=np.array([x.n for x in examples]); np.savez_compressed(ROOT/'k_probe_inputs.npz',input=tok,lengths=ns,permutations=perms)
    write_csv(ROOT/'k_length_input_audit.csv',[dict(length=n,anchors=int((ns==n).sum()),
        unique_permutations=len(set(map(tuple,perms[ns==n,:n])))) for n in range(2,31)])
    config_rows.append({'path':str(REPO/'configs/property_task_geometry.toml'),'sha256':sha(REPO/'configs/property_task_geometry.toml')})
    # Directly inspect checkpoints, reconstruct every distinct model configuration, and compare caches.
    audited={}
    for ix,j in enumerate(jobs):
        cp=j['checkpoint']
        if cp not in audited:
            assert sha(cp)==j['checkpoint_sha256']; assert sha(j['marker'])==j['marker_sha256']
            payload=torch.load(cp,map_location='cpu',weights_only=False)
            cfg=payload['config']; model,before,after=build(cfg)
            ih=weight_hash(model.state_dict()); initial_models.setdefault(j['seed'],(cfg,ih))
            # Loading trained weights must accept this exact architecture and vocabulary.
            trained=build(cfg)[0]; trained.load_state_dict(payload['model'],strict=True)
            marker=json.loads(open(j['marker']).read())
            assert marker['global_step']==20000
            rec=dict(checkpoint=cp,checkpoint_sha256=j['checkpoint_sha256'],initial_weight_hash_rebuilt=ih,
                initial_status=STATUS,original_initial_identity_verified=False,
                rng_before_factory=before,rng_after_factory=after,vocab_size=int(model.config.vocab_size),
                checkpoint_fields=list(payload),config=cfg,data_fingerprints=payload['data_fingerprints'],
                training_steps=marker['global_step'],missing_original_init=True)
            audited[cp]=rec
            if j['family']=='specialist':
                for cond, net in [('matched_init',model),('trained',trained)]:
                    p=PARENT/'features'/f"{cond}_{j['task']}_s{j['seed']}.npz"
                    meta=json.loads(p.with_suffix('.json').read_text())
                    assert meta['archive_sha256']==sha(p); assert meta['dataset_sha256']==sha(PARENT/'dataset.npz')
                    with np.load(p) as a:
                        actions=list(map(int,a['actions'])); ids,ex=samples(d,actions)
                        original=extract_landmark_activations(net,ex,device=device,batch_size=len(ex))['final_norm'].numpy()
                        cached=a['prefix_final_norm'][ids].reshape(-1,256)
                        err=float(np.max(np.abs(original-cached)))
                        assert err<3e-5,(str(p),err)
                    cache_checks.append(dict(path=str(p),sha256=meta['archive_sha256'],kind=cond,
                        prefix_only_sample_max_abs_error=err,original_init_identity_verified=False))
            if cp in cache_by_cp:
                p=cache_by_cp[cp]; a=torch.load(p,map_location='cpu',weights_only=False)
                assert a['checkpoint_sha256']==j['checkpoint_sha256'];assert a['probe_sha256']==probe['probe_sha256']
                inds=[int(np.flatnonzero(ns==n)[0]) for n in [2,10,20,30]]
                original=extract_landmark_activations(trained,[examples[t] for t in inds],device=device,batch_size=4)['final_norm'].numpy()
                err=float(np.max(np.abs(original-a['layers']['final_norm'][inds].numpy())))
                assert err<3e-5,(p,err)
                cache_checks.append(dict(path=p,sha256=sha(p),kind='trained_k',prefix_only_sample_max_abs_error=err,
                    original_init_identity_verified=False))
            del payload,model,trained
        rec=audited[cp]
        source_rows.append({**j,'initial_status':STATUS,'initial_weight_hash_rebuilt':rec['initial_weight_hash_rebuilt'],
            'original_initial_identity_verified':False,'training_steps':rec['training_steps'],'vocab_size':rec['vocab_size']})
        if ix%10==0: print({'audited_models':ix+1,'total':len(jobs)},flush=True)
    # Verify existing initialization activations, extract the one missing reference (seed101) only.
    refs={}
    for seed,(cfg,ih) in sorted(initial_models.items()):
        net=build(cfg)[0]; assert weight_hash(net.state_dict())==ih
        source=REPO/f'results/property32-zero-overlap/replicates/r0/cka/cache/random-transformer-seed{seed}.pt'
        inds=[int(np.flatnonzero(ns==n)[0]) for n in [2,10,20,30]]
        if source.exists():
            a=torch.load(source,map_location='cpu',weights_only=False)
            assert a['probe_sha256']==probe['probe_sha256']
            values=a['layers']['final_norm'].numpy()
            original=extract_landmark_activations(net,[examples[t] for t in inds],device=device,batch_size=4)['final_norm'].numpy()
            err=float(np.max(np.abs(original-values[inds]))); assert err<3e-5
            provenance='reused existing random forward cache; program reproduction sampled'
        else:
            from experiments.longrun_attention import accelerate
            accelerate(net)
            values=extract_landmark_activations(net,examples,device=device,batch_size=128)['final_norm'].numpy()
            # Recheck accelerated full-reference against original, unpatched forward.
            original=extract_landmark_activations(build(cfg)[0],[examples[t] for t in inds],device=device,batch_size=4)['final_norm'].numpy()
            err=float(np.max(np.abs(original-values[inds]))); assert err<3e-5
            provenance='new inference of missing existing-seed101 reference; no training'
        target=ROOT/f'k_initial_reference_seed{seed}.npy';np.save(target,values)
        refs[str(seed)]=dict(path=str(target),sha256=sha(target),initial_weight_hash_rebuilt=ih,
            source=str(source) if source.exists() else None,source_sha256=sha(source) if source.exists() else None,
            prefix_only_sample_max_abs_error=err,initial_status=STATUS,provenance=provenance)
        print({'initial_reference_ready':seed,'provenance':provenance},flush=True)
    candidates=subprocess.run(['rg','--files','--hidden','--no-ignore',str(REPO/'runs')],capture_output=True,text=True,check=True).stdout.splitlines()
    import re
    matches=[p for p in candidates if re.search(r'(step[-_]?0\b|initial[-_]?weights|initialization|init[-_]?checkpoint)',p,re.I)]
    write_csv(ROOT/'model_inventory.csv',source_rows)
    write_json(ROOT/'initialization_audit.json',dict(status_counts={STATUS:len(jobs)},
        logical_models=len(jobs),unique_checkpoints=len(audited),original_initial_weights_found=matches,
        model_configs_and_rng=list(audited.values()),reference_caches=refs,
        initialization_path='training.py _seed_everything immediately followed by _default_model_factory; constructor and self.apply initialization preserved; tied embeddings applied last',
        limitations=['No original step-0 weights or weight hashes found; reconstructed identity is not original identity.',
          'Original training commit and PyTorch environment were not embedded in these checkpoints.',
          'Current/source-history initialization path agrees with stored configs and cached reference features, but cannot establish historical bitwise equality.']))
    write_json(ROOT/'cache_validation.json',cache_checks)
    versions={p:sha(WORKSPACE/'external/neurips/src/neurips_permutations'/p) for p in ['models.py','training.py','passage.py','math_ops.py','cka.py']}
    write_json(ROOT/'artifact_inventory.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        workspace_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=WORKSPACE,text=True).strip(),
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        configs=config_rows,source_hashes=versions,dataset_sha256=sha(PARENT/'dataset.npz'),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),k_probe=probe['probe_sha256'],
        torch_version=torch.__version__,numpy_version=np.__version__,python_version=sys.version,
        reused=['specialist per-length ordinary/input-adjusted CKA, answer strata, gold/output baselines',
          'specialist checkpoint and initialization feature archives', 'Property32 fifty trained activation caches and random-seed17/42 caches'],
        missing=['Original specialist/Property32 step-0 weight artifacts and checksums',
          'Original training commit/environment records', 'Property32 seed101 initial forward cache (completed by inference in this directory)'],
        added=['nine ordered existing-seed pairs','debiased/per-length permutation sensitivity','paired input bootstrap',
          'own-initialization k-series drift','prefix-only cache verification'],training_started=False))


if __name__=='__main__':run()
