import json
from pathlib import Path
import tempfile
import unittest

from moss_speculative_audit import audit


class ActualCounterAudit(unittest.TestCase):
    def fixture(self, changes=''):
        # The repeated 1,2 suffix proposes the already known 1,2 continuation;
        # its first verification rejects against EOS. Four calls consume six
        # target inputs, then the cached EOS seed stops without another call.
        return ('BENCH_GENERATION tokens=5 stop=eos\n'
                'BENCH_TOKEN_TRACE eos=9 total=5 offset=0 ids=[1,2,1,2,9]\n'
                'CPU_SPECULATIVE targetCalls=4 draftedCalls=1 proposed=2 accepted=0 '
                'targetInputTokens=6 rewinds=1 draftSeconds=0.000001 '
                'rewindSeconds=0.000002 maximumDraft=4\n'+changes)

    def run_audit(self, contents):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.log'
            path.write_text(contents)
            return audit(path)

    def test_actual_counter_agreement_and_private_output(self):
        result = self.run_audit(self.fixture())
        self.assertTrue(result['counterReplayExact'])
        self.assertEqual(result['proposalRanks']['1'], {'dispatched': 1, 'verified': 1, 'accepted': 0})
        encoded = json.dumps(result)
        self.assertNotIn('ids', encoded)
        self.assertNotIn('[1, 2, 1, 2, 9]', encoded)

    def test_wrong_actual_work_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Actual target work'):
            self.run_audit(self.fixture().replace('targetInputTokens=6', 'targetInputTokens=5'))

    def test_duplicate_actual_record_rejected(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            self.run_audit(self.fixture()+self.fixture().splitlines()[-1])

    def test_missing_actual_field_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.run_audit(self.fixture().replace('rewinds=1 ', ''))

    def test_nonfinite_cost_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid draft'):
            self.run_audit(self.fixture().replace('draftSeconds=0.000001', 'draftSeconds=nan'))

    def test_incomplete_trace_rejected(self):
        with self.assertRaises(ValueError):
            self.run_audit(self.fixture().replace('ids=[1,2,1,2,9]', 'ids=[1,2,1,2]'))


if __name__ == '__main__':
    unittest.main()
