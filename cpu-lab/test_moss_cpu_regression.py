"""Meaningful failure-mode tests for the optimization acceptance gate."""
import unittest
from moss_cpu_regression import evaluate, run_environment, variant_settings


class RegressionGateTest(unittest.TestCase):
    def row(self,variant="baseline",**changes):
        return {"case":"meeting","variant":variant,"outputSha256":"full-words-speakers-timestamps",
                "inputSha256":"fixed-audio","duration":60.,
                "tokens":100,"stop":"eos","returncode":0,"complete":True,
                "wallSeconds":20.,"maxRssKiB":1000,**changes}

    def test_identical_faster_output_passes(self):
        self.assertTrue(evaluate([self.row(),self.row("4",wallSeconds=18.)])["passed"])

    def test_faster_than_production_can_still_regress_validated_cache(self):
        rows = [self.row(),self.row("cache-reference",wallSeconds=10.),self.row("80",wallSeconds=18.)]
        self.assertTrue(evaluate(rows)["passed"])
        cache_rows = [row for row in rows if row["variant"] != "baseline"]
        self.assertFalse(evaluate(cache_rows,baseline="cache-reference")["passed"])

    def test_new_candidate_must_match_validated_cache_output(self):
        rows = [self.row("cache-reference",wallSeconds=10.),self.row("80",wallSeconds=9.,outputSha256="different")]
        self.assertFalse(evaluate(rows,baseline="cache-reference")["passed"])

    def test_text_speaker_or_timestamp_change_rejected(self):
        self.assertFalse(evaluate([self.row(),self.row("4",outputSha256="changed")])["passed"])

    def test_truncated_output_rejected_even_if_fast(self):
        self.assertFalse(evaluate([self.row(),self.row("4",stop="token_limit",complete=False,wallSeconds=1.)])["passed"])

    def test_speed_regression_rejected(self):
        self.assertFalse(evaluate([self.row(),self.row("4",wallSeconds=22.)])["passed"])

    def test_unstable_baseline_rejected(self):
        self.assertFalse(evaluate([self.row(),self.row(outputSha256="other"),self.row("4")])["passed"])

    def test_no_data_rejected(self):
        self.assertFalse(evaluate([])["passed"])

    def test_missing_baseline_rejected(self):
        self.assertFalse(evaluate([self.row("4")])["passed"])

    def test_competing_api_inference_rejected(self):
        self.assertFalse(evaluate([self.row(),self.row("4",concurrentMoss=True)])["passed"])

    def test_changed_input_rejected(self):
        self.assertFalse(evaluate([self.row(),self.row("4",inputSha256="other-audio")])["passed"])

    def test_missing_candidate_rejected(self):
        self.assertFalse(evaluate([self.row()])["passed"])

    def test_thread_sweep_rejects_invalid_worker_counts_and_labels(self):
        for label in ("48@0", "48@-1", "48@4-active", "48@4-passive-extra", "baseline@4"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                variant_settings(label, 16)
        self.assertEqual(variant_settings("48", 16), ("48", 16, False))
        self.assertEqual(variant_settings("48@4-passive", 16), ("48", 4, True))

    def test_default_control_cannot_inherit_passive_wait_or_thread_limit(self):
        inherited = {"OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0",
                     "OMP_DYNAMIC": "TRUE", "OMP_THREAD_LIMIT": "2"}
        default = run_environment(inherited, {"OMP_PROC_BIND": "spread"}, "48", 16, "")
        self.assertNotIn("OMP_WAIT_POLICY", default)
        self.assertNotIn("GOMP_SPINCOUNT", default)
        self.assertNotIn("OMP_THREAD_LIMIT", default)
        self.assertEqual(default["OMP_DYNAMIC"], "FALSE")
        self.assertEqual(default["OMP_NUM_THREADS"], "16")
        passive = run_environment(inherited, {}, "48", 4, "", passive=True)
        self.assertEqual(passive["OMP_WAIT_POLICY"], "PASSIVE")
        self.assertEqual(passive["GOMP_SPINCOUNT"], "0")
        self.assertEqual(passive["MTD_THREADS"], "4")


if __name__ == "__main__":
    unittest.main()
