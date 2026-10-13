import copy
import hashlib
import unittest
from moss_cpu_regression import evaluate, evaluate_candidate_reference, evaluate_token_traces
from moss_speedup_summary import summarize

def fixture():
    variants=['baseline','cache-reference','0','48']
    cases=[];runs=[]
    for name,category,duration,slow,fast in [('recorded-a','recorded-speech',60,10.,5.),('synthetic-en','synthetic-speech',4,1.,1.),('silence','control',60,1.,1.)]:
        digest=hashlib.sha256(name.encode()).hexdigest()
        cases.append({'case':name,'group':name,'category':category,'durationSeconds':duration,'inputSha256':digest,'origin':'existing-fixture'})
        for repeat in range(3):
            for variant in variants:
                wall=slow if variant in ('baseline','0') else fast
                # One extreme outlier proves the summary uses medians, not sums.
                if name=='recorded-a' and repeat==0 and variant=='baseline':wall=100.
                row={'case':name,'variant':variant,'repeat':repeat,'inputSha256':digest,'duration':duration,'outputSha256':digest,
                     'tokens':10,'stop':'eos','returncode':0,'complete':True,'concurrentMoss':False,'wallSeconds':wall,'maxRssKiB':1000}
                if variant in ('0','48'):row['tokenTrace']={'sha256':digest,'count':10}
                runs.append(row)
    report={'runs':runs,'sourceClean':True,'sourceCommit':'a'*40,'sourceTree':'b'*40,'ggmlCommit':'c'*40,'modelSha256':'d'*64,
            'binarySha256':{'baseline':'e'*64,'candidate':'f'*64,'cache-reference':'0'*64},'physicalCoreAffinity':list(range(16)),'threads':16,
            'artifactGate':{'passed':True,'changed':[]},'publication':{'harnessExit':1},'cpu':{'model':'Test CPU','physicalCores':16,'measuredThreads':16}}
    refresh(report)
    protocol={'schema':'test-corpus','cases':cases,'variants':variants,'repeats':3,'caseCount':3,'measuredRuns':36,'threads':16,'discardedWarmups':4,
              'sourceCommit':report['sourceCommit'],'candidateBinarySha256':report['binarySha256']['candidate'],
              'originalBinarySha256':report['binarySha256']['baseline'],'immutableQuiet48Sha256':report['binarySha256']['cache-reference']}
    return report,protocol

def refresh(report):
    report['tokenTraceGate']=evaluate_token_traces(report['runs'],'0')
    report['gate']=evaluate(report['runs'])
    report['cacheReferenceGate']=evaluate([r for r in report['runs'] if r['variant']!='baseline'],baseline='cache-reference')
    report['candidateReferenceGate']=evaluate_candidate_reference(report['runs'],'0')

class SpeedupSummaryTests(unittest.TestCase):
    def setUp(self):self.report,self.protocol=fixture()
    def test_sum_of_case_medians(self):
        s=summarize(self.report,self.protocol)
        self.assertEqual(s['aggregates']['all']['sumCaseMedianWallSeconds']['baseline'],12.)
        self.assertEqual(s['aggregates']['all']['sumCaseMedianWallSeconds']['48'],7.)
        self.assertAlmostEqual(s['aggregates']['all']['originalTo48Speedup'],12/7)
        self.assertEqual(s['aggregates']['recordedSpeech']['originalTo48Speedup'],2.)
    def test_missing_run_rejected(self):
        self.report['runs'].pop()
        with self.assertRaises(ValueError):summarize(self.report,self.protocol)
    def test_duplicate_run_rejected(self):
        self.report['runs'][-1]=copy.deepcopy(self.report['runs'][0])
        with self.assertRaises(ValueError):summarize(self.report,self.protocol)
    def test_input_change_rejected(self):
        self.report['runs'][0]['inputSha256']='changed'
        refresh(self.report)
        with self.assertRaises(ValueError):summarize(self.report,self.protocol)
    def test_dirty_source_rejected(self):
        self.report['sourceClean']=False
        with self.assertRaises(ValueError):summarize(self.report,self.protocol)
    def test_nonfinite_time_rejected(self):
        self.report['runs'][0]['wallSeconds']=float('inf')
        refresh(self.report)
        with self.assertRaises(ValueError):summarize(self.report,self.protocol)
    def test_raw_path_not_exported(self):
        self.protocol['cases'][0]['sourcePath']='/private/recording.wav'
        with self.assertRaises(ValueError):summarize(self.report,self.protocol)
    def test_incomplete_output_retained(self):
        bad=next(r for r in self.report['runs'] if r['variant']=='48')
        bad.update(complete=False,stop='limit')
        refresh(self.report);s=summarize(self.report,self.protocol)
        self.assertEqual(s['measuredRuns'],36);self.assertEqual(s['completeRuns'],35)
        self.assertFalse(s['selected48Gates']['originalTo48']['passed'])
        self.assertFalse(s['tokenTraceGate']['passed'])
    def test_slow_case_retained(self):
        for row in self.report['runs']:
            if row['case']=='recorded-a' and row['variant']=='48':row['wallSeconds']=20.
        refresh(self.report);s=summarize(self.report,self.protocol)
        self.assertEqual(len(s['cases']),3)
        self.assertFalse(s['aggregates']['all']['perCaseOriginalSpeedGatePass'])
        self.assertTrue(s['aggregates']['all']['exactOriginalOutputParity'])
    def test_overlap_retained_as_failure(self):
        self.report['runs'][0]['concurrentMoss']=True
        refresh(self.report);s=summarize(self.report,self.protocol)
        self.assertEqual(s['concurrentMossRuns'],1)
        self.assertFalse(s['selected48Gates']['originalTo48']['passed'])
    def test_stored_gate_must_match_measurements(self):
        self.report['tokenTraceGate']['passed']=False
        with self.assertRaises(ValueError):summarize(self.report,self.protocol)
    def test_stored_exit_must_match_gates(self):
        self.report['publication']['harnessExit']=0
        with self.assertRaises(ValueError):summarize(self.report,self.protocol)

if __name__=='__main__':unittest.main()
