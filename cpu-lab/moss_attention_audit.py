"""MIT. Validate/export numeric attention evidence; never export private values."""
import argparse
import hashlib
import json
import math
from pathlib import Path

STAGES = ('attentionNorm','queryProjection','keyProjection','valueProjection',
          'queryNorm','keyNorm','queryRope','keyRope','scores','probabilities',
          'attentionContext','attentionProjection','attentionResidual','ffnNorm',
          'ffnGate','ffnUp','activation','ffnDown','layerOutput')
STAGE_KEYS = {'record','layer','stage','stageName','query','validLength','elements',
              'floatBitDifferences','maxAbs','nonfiniteElements'}
REPLAY_KEYS = {'record','layer','query','validLength','commonLength',
               'serialSoftmaxReplayBitDifferences','batchSoftmaxCommonReplayBitDifferences',
               'batchSoftmaxValidReplayVsSerialBitDifferences',
               'batchCommonVsValidSoftmaxBitDifferences','batchCommonVsValidSoftmaxMaxAbs',
               'serialContextReplayBitDifferences','batchValidContextReplayBitDifferences'}
ZERO_CONTROLS = ('nullHookVsMetadataObserverHiddenBitDifferences',
                 'nullHookVsProtectedCaptureHiddenBitDifferences','observerKvBitDifferences',
                 'captureKvBitDifferences','observerAndCaptureLogitBitDifferences',
                 'replayControlFailures',
                 'replayOperandChanges','futureProbabilityNonzeroElements',
                 'nonfiniteElements','modelWeightByteChanges')
TRUE_CONTROLS = ('semanticGraphMetadataExact','callbackIdentitiesExact',
                 'prefixAndInputsUnchanged','positionStateExact','attributionEligible')
SUMMARY_KEYS = {'record','audioPrefixTokens','cacheCapacity','generatedCommittedBefore',
                'past','appendTokens','contextOpsBuilt','contextOpsExecuted',
                'firstHiddenDifferenceQuery','firstDifferingLayer','firstDifferingStage',
                'firstDifferingQuery','stageRecords','attentionReplayRecords',
                'quietHiddenBitDifferences','quietLogitBitDifferences','quietKvBitDifferences',
                'modelWeightBytesChecked','timingPerformed','productionPromoted',
                'ignoredRawCustomParameterByteDifferences',
                *ZERO_CONTROLS,*TRUE_CONTROLS}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def nonnegative_number(value):
    return type(value) in (int,float) and math.isfinite(value) and value>=0

def validate(records):
    buckets = {name:[] for name in ('audioStage','audioAttentionReplay','audioAttentionSummary')}
    for record in records:
        require(isinstance(record,dict) and record.get('record') in buckets,'Unknown record')
        buckets[record['record']].append(record)
    summaries=buckets['audioAttentionSummary']
    require(len(summaries)==1,'Require one summary')
    summary=summaries[0]
    require(set(summary)==SUMMARY_KEYS,'Unexpected summary fields')
    require(all(summary[k] is True for k in TRUE_CONTROLS),'Observer/attribution control failed')
    require(all(type(summary[k]) is int and summary[k]==0 for k in ZERO_CONTROLS),'Mutation/replay control failed')
    require(type(summary['ignoredRawCustomParameterByteDifferences']) is int and summary['ignoredRawCustomParameterByteDifferences']>=0,'Invalid ignored byte count')
    require(summary['timingPerformed'] is False and summary['productionPromoted'] is False,'Not a timing or production audit')
    for key,value in {'audioPrefixTokens':859,'cacheCapacity':4971,'generatedCommittedBefore':128,
                      'past':987,'appendTokens':5,'contextOpsBuilt':84,'contextOpsExecuted':84,
                      'stageRecords':2660,'attentionReplayRecords':140,'firstHiddenDifferenceQuery':3,
                      'modelWeightBytesChecked':980917056}.items():
        require(type(summary[key]) is int and summary[key]==value,'Actual fixture/coverage mismatch')
    for key in ('quietHiddenBitDifferences','quietLogitBitDifferences','quietKvBitDifferences'):
        require(type(summary[key]) is int and summary[key]>0,'Require retained state failure')
    stages=buckets['audioStage'];replays=buckets['audioAttentionReplay']
    require(len(stages)==2660 and len(replays)==140,'Incomplete coverage')
    seen=set()
    for row in stages:
        require(set(row)==STAGE_KEYS,'Unexpected stage fields')
        l,s,q=row['layer'],row['stage'],row['query']
        require(all(type(x) is int for x in (l,s,q)) and 0<=l<28 and 0<=s<19 and 0<=q<5,'Stage bounds')
        key=(l,s,q);require(key not in seen,'Duplicate stage');seen.add(key)
        require(row['stageName']==STAGES[s] and row['validLength']==988+q,'Stage identity')
        require(type(row['elements']) is int and row['elements']>0,'Stage element count')
        require(type(row['floatBitDifferences']) is int and 0<=row['floatBitDifferences']<=row['elements'],'Stage differences')
        require(type(row['nonfiniteElements']) is int and row['nonfiniteElements']==0 and nonnegative_number(row['maxAbs']),'Nonfinite stage')
    first=min((r['layer'],r['stage'],r['query']) for r in stages if r['floatBitDifferences'])
    first_keys=('firstDifferingLayer','firstDifferingStage','firstDifferingQuery')
    require(all(type(summary[k]) is int for k in first_keys) and first==(0,9,0) and first==tuple(summary[k] for k in first_keys),'First operator attribution mismatch')
    seen=set()
    for row in replays:
        require(set(row)==REPLAY_KEYS,'Unexpected replay fields')
        l,q=row['layer'],row['query'];require(type(l) is int and type(q) is int and 0<=l<28 and 0<=q<5,'Replay bounds')
        key=(l,q);require(key not in seen,'Duplicate replay');seen.add(key)
        require(row['validLength']==988+q and row['commonLength']==992,'Replay lengths')
        for field in ('serialSoftmaxReplayBitDifferences','batchSoftmaxCommonReplayBitDifferences',
                      'serialContextReplayBitDifferences','batchValidContextReplayBitDifferences'):
            require(type(row[field]) is int and row[field]==0,'Own-result replay failed')
        for field in ('batchSoftmaxValidReplayVsSerialBitDifferences','batchCommonVsValidSoftmaxBitDifferences'):
            require(type(row[field]) is int and row[field]>=0,'Replay differences')
        require(nonnegative_number(row['batchCommonVsValidSoftmaxMaxAbs']),'Nonfinite replay')
        if l==0:
            require(row['batchSoftmaxValidReplayVsSerialBitDifferences']==0,'First-layer truncation does not match serial')
            require((row['batchCommonVsValidSoftmaxBitDifferences']>0)==(q<4),'First-layer length attribution mismatch')
    return {'summary':summary,'stages':stages,'attentionReplays':replays}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raw=args.log.read_bytes()
    records=[json.loads(line) for line in raw.decode().splitlines() if line.startswith('{')]
    result={'schema':'actual-audio-attention-v1','nativeLogSha256':hashlib.sha256(raw).hexdigest(),
            'collectorSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),**validate(records)}
    args.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    main()
