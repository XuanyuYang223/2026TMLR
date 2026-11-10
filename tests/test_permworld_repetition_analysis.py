"""Behavioral checks for repetition-specific selection, missingness, and ranking."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.permworld_repetition_20261106 import analysis as a


class RepetitionAnalysisTests(unittest.TestCase):
    def test_original_joint_selector_uses_all_right_actions(self):
        # Correct-action labels alone match throughout; another action splits
        # the last two samples into a group too small for the original >=3 rule.
        labels = np.zeros((6, 3, 2), dtype=np.int64)
        labels[4:, 1, 1] = 1
        data = {"test_lengths": np.full(6, 10), "test_labels": labels}
        relation = {"left": "left", "right": "right", "right_label_offset": 0}
        ids, keys, audit = a.selection(data, ["left", "right"], relation, "right", "test", 10, "answer_strata")
        np.testing.assert_array_equal(ids, [0, 1, 2, 3])
        self.assertEqual(keys.shape[1], 5)
        self.assertEqual(audit["residual_df"], 3)

    def test_fixed_length_undefined_propagates(self):
        self.assertIsNone(a.fixed_mean([1., None], 2))
        with self.assertRaises(ValueError):
            a.fixed_mean([1.], 2)
        self.assertEqual(a.fixed_mean([1., 3.], 2), 2.)

    def test_ranking_exact_ties_and_incomplete_candidate_mean(self):
        base = dict(architecture="gru", encoding="one_line", relation="r", control="answer_strata")
        rows = [dict(base, right_task="d", category="designated", fixed_transform_score_initial=.2, fixed_transform_score_trained=.8),
                dict(base, right_task="a", category="not_designated", fixed_transform_score_initial=.2, fixed_transform_score_trained=.8),
                dict(base, right_task="b", category="not_designated", fixed_transform_score_initial=.2, fixed_transform_score_trained=.1)]
        row = a.candidate_contrasts(rows)[0]
        self.assertEqual(row["trained_rank"], 1)
        self.assertEqual(row["trained_ties_with_designated"], 1)
        self.assertFalse(row["trained_unique_first"])
        rows[1]["fixed_transform_score_trained"] = None
        row = a.candidate_contrasts(rows)[0]
        self.assertIsNone(row["trained_designated_minus_other_mean"])
        self.assertIsNone(row["trained_unique_first"])
        self.assertAlmostEqual(row["trained_designated_minus_finite_other_mean"], .7)

    def test_complete_cache_pipeline_and_resume_binding(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "features").mkdir()
            tasks = ["left", "right", "other"]
            config = dict(tasks=tasks, lengths=[10, 11], architectures=["gru"], encodings=["one_line"],
                          relations=[dict(pair_id="r", left="left", right="right", input_transform="inverse", right_label_offset=0)],
                          analysis_metrics=list(a.METRICS))
            data = {}
            for split in ("validation", "test"):
                n = 24
                data[split + "_lengths"] = np.repeat([10, 11], 12)
                data[split + "_labels"] = np.zeros((n, 3, 3), np.int64)
                initial = rng.normal(size=(n, 3, 4)).astype(np.float32)
                for task in tasks:
                    for phase in ("initial", "trained"):
                        x = initial if phase == "initial" else initial + .4 * rng.normal(size=initial.shape)
                        output = np.full((n, 3, 2), .5, np.float32)
                        np.savez(root / "features" / f"gru_one_line_{task}_{phase}_{split}.npz", prefix_final=x,
                                 output_prob=output, prediction=np.zeros((n, 3), np.int64))
            np.savez(root / "data.npz", **data)
            result = a.analyze_unit(config, root, root / "analysis")
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["raw_initial_zero_reference_verified"])
            self.assertEqual(set(result["metrics"]), set(a.METRICS))
            result2 = a.analyze_unit(config, root, root / "analysis")
            self.assertEqual(result2["counts"], result["counts"])
            changed = dict(config, lengths=[10])
            with self.assertRaisesRegex(ValueError, "binding"):
                a.analyze_unit(changed, root, root / "analysis")

    def test_unit_summary_does_not_count_pending_as_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); unit = root / "unit"; unit.mkdir()
            a.write_json(unit / "completion.json", {"status": "completed"})
            a.write_csv(unit / "unit_counts.csv", [dict(comparison="A", control="answer_strata", metric="linear_cka", Bt_positive=26, Delta_positive=18)])
            a.write_csv(unit / "comparison_a_summary.csv", [dict(architecture="gru", encoding="one_line", relation="r", control="answer_strata", branch="hidden", metric="linear_cka", **{k: .1 for k in a.COMPONENTS + a.EFFECTS})])
            a.write_csv(unit / "comparison_b_contrasts.csv", [dict(architecture="gru", encoding="one_line", relation="r", control="answer_strata", initial_designated_minus_other_mean=0., trained_designated_minus_other_mean=.1, Gcat=.1)])
            result = a.summarize_units([dict(unit="u0", status="completed", analysis_dir=str(unit)), dict(unit="u1", status="pending", analysis_dir=str(root / "pending"))], root / "summary")
            self.assertEqual(result["completed_units"], 1)
            rows = a.read_csv(root / "summary/condition_effects_summary.csv")
            self.assertTrue(all(r["sample_sd"] == "" for r in rows))
            self.assertIn("1/2", (root / "summary/results.md").read_text())


if __name__ == "__main__":
    unittest.main()
