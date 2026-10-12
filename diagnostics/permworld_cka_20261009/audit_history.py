"""Compare recorded historical source snapshots, not an inferred training commit."""
import ast
import hashlib
import json
import subprocess
import numpy as np
import torch
from common import ROOT, REPO, WORKSPACE, write_json, sha
from inventory import build
from neurips_permutations.passage import TOKEN_TO_ID


def source(commit,path):
    return subprocess.check_output(['git','show',f'{commit}:{path}'],cwd=REPO,text=True)


def dumpfunc(s,name):
    return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def assignments(s,names):
    values={}
    for n in ast.parse(s).body:
        if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id in names:
            values[n.target.id]=ast.dump(n.value,include_attributes=False)
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id in names:values[target.id]=ast.dump(n.value,include_attributes=False)
    return values


def run():
    audit=json.loads((ROOT/'initialization_audit.json').read_text())
    current_train=(WORKSPACE/'external/neurips/src/neurips_permutations/training.py').read_text()
    current_model=(WORKSPACE/'external/neurips/src/neurips_permutations/models.py').read_text()
    current_passage=(WORKSPACE/'external/neurips/src/neurips_permutations/passage.py').read_text()
    names=['NUMBER_TOKENS','ORIGINAL_FIXED_TOKENS','EXTENDED_FIXED_TOKENS','PROPERTY_FIXED_TOKENS','FIXED_TOKENS','VOCABULARY','TOKEN_TO_ID','ID_TO_TOKEN']
    snapshots=[]
    # These are explicitly recorded CKA analysis commits, not purported training commits.
    commits=['28cee0de294af37975acb3bb7c8104609fc5ae98','7d9f43']
    for commit in commits:
        full=subprocess.check_output(['git','rev-parse',commit],cwd=REPO,text=True).strip()
        t=source(full,'src/neurips_permutations/training.py')
        m=source(full,'src/neurips_permutations/models.py')
        p=source(full,'src/neurips_permutations/passage.py')
        vocabulary_unchanged=assignments(p,names)==assignments(current_passage,names)
        model_unchanged=m==current_model;seeding_unchanged=dumpfunc(t,'_seed_everything')==dumpfunc(current_train,'_seed_everything')
        assert model_unchanged and vocabulary_unchanged and seeding_unchanged
        tree=ast.parse(t);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_default_model_factory')
        env={'__name__':'neurips_permutations.training','TOKEN_TO_ID':TOKEN_TO_ID}
        # Annotations are omitted solely for executing this already audited historical factory.
        node.returns=None
        for arg in node.args.args:arg.annotation=None
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<historical factory>','exec'),env)
        checks=[]
        for seed in [17,42,101]:
            rec=next(r for r in audit['model_configs_and_rng'] if r['config']['seed']==seed)
            current=build(rec['config'])[0]
            from neurips_permutations.training import TrainConfig,_seed_everything
            from common import weight_hash
            _seed_everything(seed);old=env['_default_model_factory'](TrainConfig.from_value(rec['config']))
            same=weight_hash(current.state_dict())==weight_hash(old.state_dict());assert same
            checks.append(dict(seed=seed,current_vs_historical_factory_weights_equal=same))
        # Explicitly inspect the original ordering, retaining its excerpt as audit evidence.
        segment=t[t.index('    _seed_everything(config.seed)',t.index('def train_run(')):]
        segment=segment[:segment.index('    optimizer =')]
        snapshots.append(dict(recorded_analysis_commit=full,not_original_training_commit=True,
            model_source_unchanged=model_unchanged,vocabulary_relevant_assignments_unchanged=vocabulary_unchanged,
            seeding_function_unchanged=seeding_unchanged,initialization_order_excerpt=segment,
            factory_checks=checks,model_source_sha256=hashlib.sha256(m.encode()).hexdigest(),
            training_source_sha256=hashlib.sha256(t.encode()).hexdigest()))
    # Retain the full effective token mapping, rather than recording just its dimension.
    write_json(ROOT/'vocabulary_audit.json',dict(vocab_size=len(TOKEN_TO_ID),tokens=dict(TOKEN_TO_ID),
        source_sha256=sha(WORKSPACE/'external/neurips/src/neurips_permutations/passage.py')))
    write_json(ROOT/'initialization_history.json',dict(snapshots=snapshots,
        interpretation='Recorded analysis-source snapshots agree in effective constructor/vocabulary/seeding behavior. Original training commit and framework version remain unavailable; no original initialization identity is verified.'))
    sig=json.loads((WORKSPACE/'results/specialist_cka_controls/protocol.json').read_text())['signature']
    write_json(ROOT/'relation_dependencies.json',dict(relations=sig['relations'],
        shared_data='All eight relations and all seeds use identical 2688 test orbit anchors; bootstrap indices are coupled.',
        transform_families={'inverse':[r['pair_id'] for r in sig['relations'] if r['input_transform']=='inverse'],
          'complement':[r['pair_id'] for r in sig['relations'] if r['input_transform']=='complement']},
        tasks='Eight pairs use sixteen distinct tasks, but transformations, statistics and input orbits share structure.',
        inferential_policy='No relation-population confidence interval or seed-count inflated relation sample size. Leave-one-relation results are descriptive sensitivity.'))


if __name__=='__main__':run()
