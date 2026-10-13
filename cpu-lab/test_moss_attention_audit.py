import copy
import json
from pathlib import Path
import unittest
from moss_attention_audit import validate

class AttentionAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        report=json.loads(Path(__file__).with_name('actual-audio-attention-v1.json').read_text())
        cls.records=report['stages']+report['attentionReplays']+[report['summary']]

    def corrupt(self,change):
        rows=copy.deepcopy(self.records);change(rows)
        with self.assertRaises(ValueError):validate(rows)

    def test_full_native_record_set(self):
        self.assertEqual(validate(self.records)['summary']['firstDifferingStage'],9)
    def test_missing_stage(self):
        self.corrupt(lambda r:r.pop(0))
    def test_duplicate_stage(self):
        self.corrupt(lambda r:r.__setitem__(0,r[1]))
    def test_missing_replay(self):
        self.corrupt(lambda r:r.pop(2660))
    def test_duplicate_replay(self):
        self.corrupt(lambda r:r.__setitem__(2660,r[2661]))
    def test_unapproved_raw_field(self):
        self.corrupt(lambda r:r[0].update(rawValues=[1,2,3]))
    def test_observer_mutation(self):
        self.corrupt(lambda r:r[-1].update(observerKvBitDifferences=1))
    def test_metadata_failure(self):
        self.corrupt(lambda r:r[-1].update(callbackIdentitiesExact=False))
    def test_replay_control_failure(self):
        self.corrupt(lambda r:r[2660].update(serialSoftmaxReplayBitDifferences=1))
    def test_nonfinite_stage(self):
        self.corrupt(lambda r:r[0].update(maxAbs=float('nan')))
    def test_false_truncation_attribution(self):
        self.corrupt(lambda r:r[2660].update(batchSoftmaxValidReplayVsSerialBitDifferences=1))
    def test_wrong_first_operator(self):
        self.corrupt(lambda r:r[0].update(floatBitDifferences=1))
    def test_changed_weight(self):
        self.corrupt(lambda r:r[-1].update(modelWeightByteChanges=1))
    def test_timing_claim(self):
        self.corrupt(lambda r:r[-1].update(timingPerformed=True))
    def test_unknown_record(self):
        self.corrupt(lambda r:r.append({'record':'privateTokens'}))
    def test_nonsemantic_padding_is_ignored(self):
        rows=copy.deepcopy(self.records)
        rows[-1]['ignoredRawCustomParameterByteDifferences']=3
        self.assertTrue(validate(rows)['summary']['attributionEligible'])
    def test_boolean_is_not_numeric_error(self):
        self.corrupt(lambda r:r[0].update(maxAbs=True))

if __name__=='__main__':unittest.main()
