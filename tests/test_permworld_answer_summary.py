"""Checks for strongest-transform controls, pair bootstrap and exact ranks."""
import csv
import tempfile
from pathlib import Path
import unittest

import numpy as np

from experiments.permworld_answer_controls_20261109 import summary as s


def config(units=('archdev001','u1','u2')):
    return dict(tasks=['left','right','other'],lengths=[10,11],
                architectures=['gru'],encodings=['one_line','canonical_cycle'],
                actions=['identity','complement','inverse'],units=[dict(unit=u) for u in units],
                historical_unit='archdev001',relations=[dict(pair_id='pair',left='left',right='right',input_transform='complement')],
                summary_make_plots=False)


def fixture(c):
    rows=[]
    for u in c['units']:
        for enc in c['encodings']:
            for phase in s.PHASES:
                for version in s.VERSIONS:
                    for source in c['tasks']:
                        for candidate in c['tasks']:
                            if source==candidate:continue
                            for action in c['actions']:
                                value=.1
                                if phase=='trained':
                                    value={'identity':.9,'complement':.8,'inverse':.1}[action]
                                    if candidate=='other':value-=.02
                                    if version==s.VERSIONS[-1]:value={'identity':.1,'complement':.15,'inverse':.2}[action]
                                rows.append(dict(unit=u['unit'],phase=phase,architecture='gru',encoding=enc,
                                                 source_task=source,candidate_task=candidate,transform=action,cka_version=version,cka=value))
    return rows


class AnswerSummaryTests(unittest.TestCase):
    def test_strongest_alternative_includes_identity(self):
        c=config(('u1',));rows=fixture(c);ranks=s.rank_candidates(c,rows)
        _,paired=s.transformation_advantages(c,rows,ranks)
        row=next(r for r in paired if r['cka_version']=='pooled_raw' and r['encoding']=='one_line')
        self.assertAlmostEqual(row['trained_transform_advantage'],-.1)
        self.assertAlmostEqual(row['advantage_training_increment'],-.1)
        # This is not the old correct-minus-one-wrong score (.8-.1).
        c['alternative_exact_transforms']={'pair':['identity']}
        _,paired=s.transformation_advantages(c,rows,ranks)
        row=next(r for r in paired if r['cka_version']=='pooled_raw' and r['encoding']=='one_line')
        self.assertAlmostEqual(row['trained_transform_advantage'],.7)
        self.assertEqual(row['eligible_alternative_transforms'],'inverse')

    def test_exact_ranking_ties_and_missing_inventory(self):
        c=config(('u1',));rows=[r for r in fixture(c) if r['phase']=='initial']
        ranks=s.rank_candidates(c,rows)
        self.assertTrue(all(r['rank']==1 and r['ties_with_candidate']==1 and not r['unique_first'] for r in ranks))
        missing=rows[1:];ranks=s.rank_candidates(c,missing)
        group=[r for r in ranks if r['phase']==rows[0]['phase'] and r['encoding']==rows[0]['encoding']
               and r['source_task']==rows[0]['source_task'] and r['transform']==rows[0]['transform']
               and r['cka_version']==rows[0]['cka_version']]
        self.assertTrue(group and all(r['rank'] is None for r in group))

    def test_fixed_lengths_propagate_missing_and_pooled_cannot_be_averaged(self):
        c=config(('u1',));row=fixture(c)[0]
        row.update(cka_version='within_length_debiased',length=10,cka=.4)
        aggregate=s.aggregate_scores(c,[row])[0]
        self.assertIsNone(aggregate['cka'])
        full=s.aggregate_scores(c,[row,dict(row,length=11,cka=.2)])[0]
        self.assertAlmostEqual(full['cka'],.3)
        with self.assertRaisesRegex(ValueError,'cannot be reconstructed'):
            s.aggregate_scores(c,[dict(row,cka_version='pooled_raw')])

    def test_bootstrap_counts_task_pairs_not_seed_rows(self):
        c=config(('u1',));rows=fixture(c);_,paired=s.transformation_advantages(c,rows,s.rank_candidates(c,rows))
        one=s._grid_estimate(c,paired,['u1'],'pooled_raw','trained_transform_advantage')
        duplicate=paired+[dict(r,unit='u2') for r in paired]
        two=s._grid_estimate(c,duplicate,['u1','u2'],'pooled_raw','trained_transform_advantage')
        np.testing.assert_allclose(one['pair_means'],two['pair_means'])
        rng=np.random.default_rng(5);draws=rng.integers(0,1,size=(50,1))
        self.assertEqual(s.bootstrap_pairs(one['pair_means'],draws),s.bootstrap_pairs(two['pair_means'],draws))

    def test_missing_initial_does_not_remove_trained_result(self):
        c=config(('u1',));rows=[r for r in fixture(c) if r['phase']=='trained']
        _,paired=s.transformation_advantages(c,rows,s.rank_candidates(c,rows))
        self.assertTrue(all(r['trained_transform_advantage'] is not None for r in paired))
        self.assertTrue(all(r['initial_transform_advantage'] is None and r['advantage_training_increment'] is None for r in paired))

    def test_learning_nll_uses_named_full_and_conditional_fields_separately(self):
        c=config(('u1','u2'));rows=fixture(c);_,paired=s.transformation_advantages(c,rows,s.rank_candidates(c,rows))
        learning=[]
        for unit in ('u1','u2'):
            for enc in c['encodings']:
                for task in c['tasks']:
                    learning.append(dict(unit=unit,architecture='gru',encoding=enc,task=task,phase='trained',split='test',
                        accuracy_minus_length_mode=.1,full_nll_minus_length_prior_nll=None,
                        conditional_numeric_nll_minus_length_prior_nll=-.5,expected_answer_spearman=.2))
        joined,_=s.learning_associations(c,paired,learning)
        self.assertTrue(all(r['pair_mean_full_nll_gap'] is None for r in joined))
        self.assertTrue(all(r['pair_mean_conditional_nll_gap']==-.5 for r in joined))

    def test_complete_saved_score_summary(self):
        c=config();rows=fixture(c)
        with tempfile.TemporaryDirectory() as tmp:
            result=s.summarize(c,rows,[],Path(tmp))
            self.assertEqual(result['primary_task_pairs'],1)
            self.assertTrue(result['no_training'])
            with (Path(tmp)/'task_pair_bootstrap.csv').open() as f:intervals=list(csv.DictReader(f))
            self.assertEqual({r['bootstrap_unit'] for r in intervals},{'task_pair'})
            self.assertEqual({r['task_pair_count'] for r in intervals},{'1'})
            self.assertIn('not the old', (Path(tmp)/'report.md').read_text())
            self.assertTrue((Path(tmp)/'peaks_valleys_summary.csv').exists())


if __name__=='__main__':unittest.main()
