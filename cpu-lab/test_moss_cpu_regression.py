"""Meaningful failure-mode tests for the optimization acceptance gate."""
import unittest
from moss_cpu_regression import evaluate


class RegressionGateTest(unittest.TestCase):
    def row(self,variant="baseline",**changes):
        return {"case":"meeting","variant":variant,"outputSha256":"full-words-speakers-timestamps",
                "inputSha256":"fixed-audio","duration":60.,
                "tokens":100,"stop":"eos","returncode":0,"complete":True,
                "wallSeconds":20.,"maxRssKiB":1000,**changes}

    def test_identical_faster_output_passes(self):
        self.assertTrue(evaluate([self.row(),self.row("4",wallSeconds=18.)])["passed"])

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


if __name__ == "__main__":
    unittest.main()
