"""Synthetic engineering checks only; no saved scientific arrays or scores."""
import ast
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

D = Path(__file__).resolve().parent
P = D / 'fixed_transform_analysis.py'
ast.parse(P.read_text())
spec = importlib.util.spec_from_file_location('pairing_engineering', P)
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
metric_path = D.parents[1] / 'diagnostics/permworld_scope_inventory_20261009/metrics.py'
assert a.sha(metric_path) == a.METRIC_SHA
spec = importlib.util.spec_from_file_location('old_metric_engineering', metric_path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
checks = []


def check(label, condition):
    assert condition, label
    checks.append({'check': label, 'passed': True})


rng = np.random.default_rng(20261011119)
x, y = rng.normal(size=(64, 12)), rng.normal(size=(64, 9))
keys = np.column_stack([np.repeat(10, 64), np.repeat(np.arange(16), 4)])
for kk, name in [(None, 'raw'), (keys, 'joint_strata')]:
    xc, yc = m.center(x, kk), m.center(y, kk)
    value, reason = a.score(m, xc, yc)
    independent = a.independent_covariance(xc, yc)
    check('synthetic_Gram_vs_covariance_' + name, abs(value-independent) < 1e-12 and reason == '')
check('constant_CKA_undefined', a.score(m, np.ones((12, 4)), np.ones((12, 4)))[0] is None)
check('fixed_weights_no_partial_NaN_reweight', a.fixed_mean([.1, None, .3], [10, 11, 12]) is None)
check('fixed_weights_all_lengths', abs(a.fixed_mean([.1, .2, .3], [10, 11, 12])-.2) < 1e-15)
s = a.SavedSystem.__new__(a.SavedSystem)
s.domain = 'permworld'
s.names, s.actions = ['left', 'designated', 'other'], ['identity', 'complement', 'inverse']
s.ns, s.labels = np.repeat(10, 12).astype(np.int64), np.zeros((12, 3, 3), np.int64)
s.labels[:, 0, 0] = np.repeat([0, 1], 6)
s.labels[:, 0, 1] = 2
s.labels[:, 1, 1] = np.repeat([2, 3], 6)
s.labels[:, 2, 1] = np.tile(np.repeat([2, 3], 3), 2)
s.relations = {'r': {'left': 'left', 'right': 'designated', 'input_transform': 'inverse', 'right_label_offset': 1}}
pair = dict(relation='r', left_task='left', right_task='designated', designated_right_task='designated', transform='inverse')
ids, key, _, audit = s.selection(pair, 'answer_strata', 10)
check('original_three_right_roles_retained', len(ids) == 12 and audit['groups'] == 4 and key.shape == (12, 5))
full = np.column_stack((s.ns, s.labels[:, 0, 0], s.labels[:, :, 1]))
_, gi = np.unique(full, axis=0, return_inverse=True)
_, gj = np.unique(key, axis=0, return_inverse=True)
check('plus1_coordinate_shift_same_partition', np.array_equal(gi[:, None] == gi[None, :], gj[:, None] == gj[None, :]))
check('residual_degrees_of_freedom', audit['residual_df'] == 8)
check('raw_uses_original_anchor_indices', np.array_equal(s.selection(pair, 'raw_within_length', 10)[0], np.arange(12)))
base = dict(domain='permworld', relation='r', architecture='transformer', encoding='one_line', left_task='left',
            right_task='designated', transform='inverse', category='designated', designated_right_task='designated',
            pair_id='toyD', original_right_label_mapping='subtract1', control='answer_strata')
rows = [dict(base, fixed_transform_score_initial=.2, fixed_transform_score_trained=.3, training_score_change=.1),
        dict(base, right_task='other', category='not_designated', pair_id='toyO', fixed_transform_score_initial=.1,
             fixed_transform_score_trained=.25, training_score_change=.15),
        dict(base, right_task='other2', category='not_designated', pair_id='toyO2', fixed_transform_score_initial=None,
             fixed_transform_score_trained=.8, training_score_change=None)]
cats, comparisons, changes = a.category_tables(rows)
empty = [r for r in cats if r['category'] == 'other_known_exact']
check('empty_exact_category_retained', len(empty) == 2 and all(r['total_count'] == 0 and r['finite_mean'] is None for r in empty))
change = next(r for r in changes if r['category'] == 'not_designated')
check('phase_intersection_common_support', change['candidate_total'] == 2 and change['finite_common_pair_count'] == 1
      and change['common_support_right_tasks'] == 'other')
check('Gcat_formula_on_identical_pair_support', abs(change['Gcat']-(-.05)) < 1e-12)
check('no_Torch_import', 'torch' not in sys.modules)
result = dict(status='passed', fixture_only=True, synthetic_inputs_only=True, no_real_cache_read=True,
              no_real_new_CKA=True, no_training=True, no_forward=True, source_sha256=a.sha(P),
              fixture_source_sha256=a.sha(__file__), metric_source_sha256=a.sha(metric_path), checks=checks)
(D / 'engineering_fixture.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('status', 'source_sha256', 'fixture_source_sha256')}, ensure_ascii=False))
