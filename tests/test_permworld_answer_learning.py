"""Behavioral checks for read-only numeric/full-vocabulary learning diagnostics."""
import csv
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.permworld_answer_controls_20261109 import learning as l


class AnswerLearningTests(unittest.TestCase):
    def test_training_only_prior_smoothing_and_tie_rule(self):
        prior, mode, counts = l.smoothed_length_prior([0, 1, 0, 1], classes=3)
        np.testing.assert_array_equal(counts, [2, 2, 0])
        np.testing.assert_allclose(prior, [2.5/5.5, 2.5/5.5, .5/5.5])
        self.assertEqual(mode, 0)
        self.assertGreater(prior[2], 0)
        with self.assertRaises(ValueError):
            l.smoothed_length_prior([], classes=3)

    def test_nll_direction_and_zero_not_silently_clipped(self):
        baseline = np.array([.8, .2])
        good = np.array([[.99, .01], [.01, .99]])
        bad = np.array([[.01, .99], [.99, .01]])
        y = np.array([0, 1])
        nll = l.categorical_nll(baseline, y)[0]
        self.assertLess(l.categorical_nll(good, y)[0] - nll, 0)
        self.assertGreater(l.categorical_nll(bad, y)[0] - nll, 0)
        value, zeros = l.categorical_nll([[0., 1.], [.5, .5]], y)
        self.assertTrue(math.isinf(value)); self.assertEqual(zeros, 1)

    def test_spearman_ties_and_constantness(self):
        value, reason = l.spearman_expectation([.2, .2, .8, .8], [0, 0, 1, 1])
        self.assertAlmostEqual(value, 1.); self.assertEqual(reason, "")
        self.assertEqual(l.spearman_expectation([.3]*4, [0, 1, 0, 1]), (None, "constant_expected_prediction"))
        self.assertEqual(l.spearman_expectation([0, 1, 2], [1, 1, 1]), (None, "constant_true_answer"))

    def test_full_ce_is_not_conditional_ce_and_optional_missing_stays_missing(self):
        config = dict(unit="fixture", tasks=["t"], architectures=["gru"], encodings=["cycle"], lengths=[10, 11], numeric_classes=3)
        train_y = np.array([0, 0, 1, 1, 0, 0, 1, 1]).reshape(8, 1, 1)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        data = dict(train_lengths=np.repeat([10, 11], 4), train_labels=train_y,
                    test_lengths=np.repeat([10, 11], 4), test_labels=np.repeat(y[:, None, None], 3, axis=1))
        p = np.full((8, 3, 3), .05)
        for i, gold in enumerate(y):p[i, :, gold] = .9
        arrays = dict(prediction=np.repeat(y[:, None], 3, axis=1), output_prob=p,
                      answer_ce=np.full(8, 2.0), numeric_mass=np.full((8, 3), .15))
        get = lambda arch, enc, task, phase, split, field: arrays[field]
        with tempfile.TemporaryDirectory() as directory:
            result = l.evaluate_learning(config, data, get, directory)
            row = result["summary"][0]
            self.assertEqual(row["exact_match_accuracy"], 1.)
            self.assertEqual(row["model_full_vocabulary_nll"], 2.)
            self.assertAlmostEqual(row["model_conditional_numeric_nll"], -math.log(.9))
            self.assertGreater(row["full_nll_minus_length_prior_nll"], 0)
            self.assertLess(row["conditional_numeric_nll_minus_length_prior_nll"], 0)
            self.assertAlmostEqual(row["expected_answer_spearman"], 1.)
            self.assertTrue(row["full_nll_available_all_lengths"])
            # A good conditional numeric prediction must not conceal an
            # invalid full-vocabulary answer-position argmax.
            arrays["prediction"][0, :] = 99
            invalid = l.evaluate_learning(config, data, get, directory)["summary"][0]
            self.assertEqual(invalid["exact_match_accuracy"], .875)
            self.assertEqual(invalid["numeric_argmax_accuracy"], 1.)
            self.assertEqual(invalid["full_vocabulary_prediction_outside_answer_support_fraction"], .125)
            del arrays["answer_ce"]
            result = l.evaluate_learning(config, data, get, directory)
            self.assertIsNone(result["summary"][0]["model_full_vocabulary_nll"])
            self.assertIsNone(result["summary"][0]["full_nll_minus_length_prior_nll"])
            self.assertIsNotNone(result["summary"][0]["model_conditional_numeric_nll"])
            self.assertFalse(result["summary"][0]["full_nll_available_all_lengths"])
            with (Path(directory)/"learning_summary.csv").open() as stream:
                saved = list(csv.DictReader(stream))[0]
            self.assertEqual(saved["model_full_vocabulary_nll"], "")

    def test_undefined_length_spearman_not_reweighted(self):
        config = dict(tasks=["t"], architectures=["gru"], encodings=["line"], lengths=[10, 11], numeric_classes=3)
        train = np.array([0, 1, 0, 1, 0, 1, 0, 1]).reshape(8, 1, 1)
        y = np.array([0, 1, 0, 1, 0, 0, 0, 0])
        data = dict(train_lengths=np.repeat([10, 11], 4), train_labels=train,
                    test_lengths=np.repeat([10, 11], 4), test_labels=np.repeat(y[:, None, None], 3, axis=1))
        p = np.full((8, 3, 3), .1)
        for i, gold in enumerate(y):p[i, :, gold] = .8
        fields = dict(prediction=np.repeat(y[:, None], 3, axis=1), output_prob=p)
        with tempfile.TemporaryDirectory() as directory:
            result = l.evaluate_learning(config, data, lambda *a: fields[a[-1]], directory)
            self.assertIsNone(result["summary"][0]["expected_answer_spearman"])
            self.assertEqual(result["summary"][0]["spearman_undefined_lengths"], 1)
            self.assertEqual(result["summary"][0]["exact_match_accuracy"], 1.)


if __name__ == "__main__":
    unittest.main()
