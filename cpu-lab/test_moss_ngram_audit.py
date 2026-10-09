import tempfile
from pathlib import Path
import unittest
from moss_ngram_audit import draft, read_trace, replay


class DraftReplayTest(unittest.TestCase):
    def test_future_suffix_does_not_change_the_past_only_draft(self):
        past = [1,2,3,4,1,2]
        self.assertEqual(draft(past), [3,4,1,2])
        self.assertEqual(draft(past, maximum=2), [3,4])

    def test_last_matching_occurrence_and_longest_suffix_win(self):
        self.assertEqual(draft([1,2,3,7,1,2,3,8,1,2,3]), [8,1,2,3])
        self.assertEqual(draft([1,2,9,3,1,2,4,1,2]), [4,1,2])

    def test_no_match_does_not_reduce_calls(self):
        result = replay([1,2,3,4,99], 99)
        self.assertEqual(result['oracleReplayTargetCalls'], 4)
        self.assertEqual(result['callReductionFactor'], 1)
        self.assertEqual(result['targetInputTokenWork'], 4)
        self.assertTrue(result['reachedEos'])

    def test_wrong_draft_adds_work_without_accepting_future_oracle_tokens(self):
        result = replay([1,2,3,1,2,4,99],99,maximum=2)
        self.assertEqual(result['acceptedDraftTokens'],0)
        self.assertEqual(result['oracleReplayTargetCalls'],6)
        self.assertGreater(result['targetInputTokenWork'],6)

    def test_repeated_sequences_reduce_calls_but_do_not_establish_speed(self):
        result = replay([1,2,3]*8+[99],99,maximum=8)
        self.assertGreater(result['callReductionFactor'],1)
        self.assertGreater(result['acceptedDraftTokens'],0)
        self.assertTrue(result['reachedEos'])

    def test_incomplete_or_multiple_eos_sequences_fail(self):
        for values in ([], [1,2], [1,99,2,99]):
            with self.assertRaises(ValueError): replay(values,99)

    def test_trace_requires_matching_actual_generation_and_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'private.log'
            path.write_text('BENCH_GENERATION tokens=3 stop=eos\nBENCH_TOKEN_TRACE eos=99 ids=[1,2,99]\n')
            tokens,eos,sha=read_trace(path)
            self.assertEqual(tokens,[1,2,99]);self.assertEqual(eos,99);self.assertEqual(len(sha),64)
            path.write_text('BENCH_GENERATION tokens=4 stop=eos\nBENCH_TOKEN_TRACE eos=99 ids=[1,2,99]\n')
            with self.assertRaises(ValueError): read_trace(path)


if __name__ == '__main__': unittest.main()
