"""The historical exploratory unit must not contaminate the new-only cohort."""
from pathlib import Path
import tempfile
import unittest

from experiments.permworld_repetition_20261106 import analysis
from tools.summarize_permworld_repetition_cohorts import summarize_cohorts


class CohortSeparationTests(unittest.TestCase):
    def fixture(self, root, name, value):
        directory = root / name
        analysis.write_json(directory / "completion.json", {"status": "completed"})
        analysis.write_csv(directory / "unit_counts.csv", [dict(comparison="A", control="answer_strata", metric="linear_cka", Bt_positive=value, Delta_positive=value)])
        analysis.write_csv(directory / "comparison_a_summary.csv", [dict(architecture="gru", encoding="one_line", relation="r", control="answer_strata", branch="hidden", metric="linear_cka", **{k: float(value) for k in analysis.COMPONENTS + analysis.EFFECTS})])
        analysis.write_csv(directory / "comparison_b_contrasts.csv", [dict(architecture="gru", encoding="one_line", relation="r", control="answer_strata", initial_designated_minus_other_mean=0., trained_designated_minus_other_mean=float(value), Gcat=float(value))])
        return dict(unit=name, status="completed", analysis_dir=str(directory))

    def test_new_only_mean_excludes_old_and_retains_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self.fixture(root, "old", 30)
            new = [self.fixture(root, "new1", 2)] + [dict(unit=f"new{i}", status="pending", analysis_dir=str(root / f"new{i}")) for i in (2, 3, 4)]
            result = summarize_cohorts(old, new, root / "out")
            self.assertEqual(result["new_four"]["completed_units"], 1)
            self.assertEqual(result["all_five"]["completed_units"], 2)
            new_count = analysis.read_csv(root / "out/new_four/unit_count_summary.csv")[0]
            all_count = analysis.read_csv(root / "out/all_five/unit_count_summary.csv")[0]
            self.assertEqual(float(new_count["mean"]), 2.)
            self.assertEqual(float(all_count["mean"]), 16.)
            self.assertEqual(new_count["sample_sd"], "")
            status = analysis.read_csv(root / "out/new_four/unit_status.csv")
            self.assertEqual(len(status), 4)
            self.assertEqual(sum(r["analysis_completed"] == "False" for r in status), 3)

    def test_empty_new_cohort_has_no_invented_zero_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = self.fixture(root, "old", 26)
            new = [dict(unit=f"new{i}", status="pending", analysis_dir=str(root / f"new{i}")) for i in range(4)]
            result = summarize_cohorts(old, new, root / "out")
            self.assertEqual(result["new_four"]["completed_units"], 0)
            self.assertEqual(analysis.read_csv(root / "out/new_four/unit_count_summary.csv"), [])
            self.assertIn("not estimated", (root / "out/report.md").read_text())
            self.assertEqual(result["all_five"]["completed_units"], 1)


if __name__ == "__main__":
    unittest.main()
