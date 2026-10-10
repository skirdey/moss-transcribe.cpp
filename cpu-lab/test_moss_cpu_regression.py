"""Meaningful failure-mode tests for the optimization acceptance gate."""
import unittest
import tempfile
from pathlib import Path
from moss_cpu_regression import evaluate, run_environment, variant_settings, parse_storage, evaluate_storage, evaluate_candidate_reference, token_fingerprint, evaluate_token_traces, evaluate_shared_activation


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

    def test_reference_cannot_inherit_candidate_phase_budgets(self):
        env = run_environment({"MTD_THREADS_DECODE":"4", "MTD_THREADS_WHISPER":"2"}, {}, "48", 16, "")
        self.assertNotIn("MTD_THREADS_DECODE",env)
        self.assertNotIn("MTD_THREADS_WHISPER",env)

    def test_phase_budget_preserves_startup_cap_and_is_explicit(self):
        env = run_environment({}, {}, "48", 16, "", phase_threads={"decode":8,"logits":8})
        self.assertEqual(env["MTD_THREADS"],"16")
        self.assertEqual(env["MTD_THREADS_DECODE"],"8")
        self.assertEqual(env["MTD_THREADS_LOGITS"],"8")
        self.assertNotIn("MTD_THREADS_PREFILL",env)

    def test_loader_records_are_unambiguous_and_old_binaries_remain_supported(self):
        line="[info] BENCH_MODEL_STORAGE mode=mapped fileBytes=128 tensorBytes=64 copiedBytes=0"
        self.assertEqual(parse_storage(line),{"mode":"mapped","fileBytes":128,"tensorBytes":64,"copiedBytes":0})
        self.assertIsNone(parse_storage("old loader without a storage record"))
        self.assertIsNone(parse_storage(line+"\n"+line))

    def test_requested_mapping_cannot_pass_with_fallback_or_inconsistent_counters(self):
        settings={"baseline":{"opt":0},"mapped":{"opt":4144}}
        valid={"mode":"mapped","fileBytes":128,"tensorBytes":64,"copiedBytes":0}
        row=self.row("mapped",repeat=0,modelStorage=valid)
        self.assertTrue(evaluate_storage([self.row(),row],settings)["passed"])
        for invalid in (None,{**valid,"mode":"copied"},{**valid,"copiedBytes":1},{**valid,"fileBytes":63}):
            self.assertFalse(evaluate_storage([{**row,"modelStorage":invalid}],settings)["passed"])

    def test_beating_older_builds_cannot_hide_same_build_storage_regression(self):
        rows=[self.row(),self.row("cache-reference",wallSeconds=15.),
              self.row("48",wallSeconds=10.),self.row("4144",wallSeconds=12.)]
        self.assertTrue(evaluate(rows)["passed"])
        self.assertTrue(evaluate([r for r in rows if r["variant"]!="baseline"],baseline="cache-reference")["passed"])
        same=evaluate_candidate_reference(rows,"48")
        self.assertFalse(same["passed"])
        self.assertEqual([c["variant"] for c in same["comparisons"]],["4144"])
        rows[-1]=self.row("4144",wallSeconds=9.8)
        self.assertTrue(evaluate_candidate_reference(rows,"48")["passed"])

    def test_private_trace_hash_requires_unique_eos_and_complete_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"private.log"
            prefix="BENCH_GENERATION tokens=3 stop=eos\nBENCH_TOKEN_TRACE eos=99 total=3 offset=0 ids="
            path.write_text(prefix+"[1,2,99]\n")
            first=token_fingerprint(path)
            self.assertEqual(first["count"],3)
            self.assertNotIn("tokens",first)
            path.write_text(prefix+"[1,3,99]\n")
            self.assertNotEqual(first["sha256"],token_fingerprint(path)["sha256"])
            for broken in ("[99,2,99]", "[1,2,98]", "[1,2]", "[1,2,"):
                path.write_text(prefix+broken+"\n")
                with self.assertRaises(ValueError): token_fingerprint(path)

    def test_same_text_and_count_cannot_hide_changed_token_ids(self):
        trace={"sha256":"canonical-token-sequence","count":100}
        rows=[self.row("48",tokenTrace=trace),self.row("8240",tokenTrace=trace)]
        self.assertTrue(evaluate_token_traces(rows,"48")["passed"])
        rows[-1]=self.row("8240",tokenTrace={**trace,"sha256":"different-ids"})
        self.assertTrue(evaluate_candidate_reference(rows,"48")["passed"])
        self.assertFalse(evaluate_token_traces(rows,"48")["passed"])

    def test_missing_trace_tail_and_unstable_token_control_are_rejected(self):
        trace={"sha256":"fixed","count":100}
        for bad in (None,{"error":"truncated"},{**trace,"count":99}):
            rows=[self.row("48",tokenTrace=trace),self.row("8240",tokenTrace=bad)]
            self.assertFalse(evaluate_token_traces(rows,"48")["passed"])
        rows=[self.row("48",tokenTrace=trace),self.row("48",tokenTrace={**trace,"sha256":"other"}),self.row("8240",tokenTrace=trace)]
        self.assertFalse(evaluate_token_traces(rows,"48")["passed"])
        self.assertFalse(evaluate_token_traces([],"48")["passed"])

    def test_trace_scope_keeps_older_binary_support_and_clears_inherited_trace(self):
        trace={"sha256":"fixed","count":100}
        rows=[self.row(),self.row("cache-reference"),self.row("48",tokenTrace=trace),self.row("8240",tokenTrace=trace)]
        self.assertTrue(evaluate_token_traces(rows,"48")["passed"])
        self.assertEqual(run_environment({"MTD_TRACE_TOKENS":"1"},{},"48",16,"")["MTD_TRACE_TOKENS"],"0")
        self.assertEqual(run_environment({}, {}, "48",16,"",trace_tokens=True)["MTD_TRACE_TOKENS"],"1")

    def test_requested_shared_activation_cannot_silently_use_reference(self):
        settings={"48":{"opt":48},"8240":{"opt":8240},"16432":{"opt":16432},"24624":{"opt":24624}}
        profile={"decode":{"sharedQ8Nodes":2},"whisper":{"sharedQ8CastNodes":3}}
        for variant in settings:
            row=self.row(variant,repeat=0,phaseProfile=profile)
            self.assertTrue(evaluate_shared_activation([row],settings)["passed"])
            if variant!="48":
                self.assertFalse(evaluate_shared_activation([{**row,"phaseProfile":None}],settings)["passed"])


if __name__ == "__main__":
    unittest.main()
