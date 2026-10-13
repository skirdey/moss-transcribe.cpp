"""MIT. Summarize every predeclared CPU case from an anonymous numeric report."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from moss_cpu_regression import evaluate, evaluate_candidate_reference, evaluate_pair_reference, evaluate_token_traces

CASE_FIELDS={'case','origin','group','category','sourceDuration','offsetSeconds',
             'requestedSeconds','durationSeconds','inputSha256'}
GATES=('artifactGate','storageGate','sharedActivationGate','fusedProjectionGate',
       'encoderQ8Gate','encoderQkvGate','gate','cacheReferenceGate',
       'candidateReferenceGate','tokenTraceGate','pairGate')

def require(ok,message):
    if not ok:raise ValueError(message)

def summarize(report,protocol):
    """Keep failures in the result. Incomplete/missing coverage cannot be summarized."""
    cases=protocol['cases'];variants=protocol['variants'];repeats=protocol['repeats']
    require(variants==['baseline','cache-reference','0','48'],'Require the declared four controls')
    require(type(repeats) is int and repeats>=2,'Need repeated measurements')
    require(protocol['caseCount']==len(cases) and len({c['case'] for c in cases})==len(cases),'Case coverage mismatch')
    for case in cases:
        require(set(case)<=CASE_FIELDS,'Unknown case field: no private source paths may be exported')
        require(re.fullmatch(r'[a-z0-9-]+',case['case']) and re.fullmatch(r'[a-z0-9-]+',case['group']),'Use anonymous case/group labels')
        require(case['origin'] in ('existing-fixture','new-excerpt','local-fixture'),'Unknown origin label')
        require(case['category'] in ('recorded-speech','synthetic-speech','control'),'Unknown case category')
    runs=report['runs']
    expected={(c['case'],v,n) for c in cases for v in variants for n in range(repeats)}
    actual=[(r['case'],r['variant'],r['repeat']) for r in runs]
    require(len(actual)==len(set(actual))==protocol['measuredRuns'] and set(actual)==expected,'Missing or duplicate measurements')
    require(report['sourceClean'],'Source was not frozen')
    require(report['physicalCoreAffinity']==list(range(protocol['threads'])) and report['threads']==protocol['threads'],'Thread/affinity mismatch')
    require(report['sourceCommit']==protocol['sourceCommit'],'Source identity mismatch')
    for label,key in [('candidate','candidateBinarySha256'),('baseline','originalBinarySha256'),('cache-reference','immutableQuiet48Sha256')]:
        require(report['binarySha256'][label]==protocol[key],'Binary identity mismatch')
    selected={name:evaluate_pair_reference(runs,reference,'48') for name,reference in
              [('originalTo48','baseline'),('quiet48To48','cache-reference'),('sameBuild0To48','0')]}
    traces=evaluate_token_traces(runs,'0')
    require(traces==report['tokenTraceGate'],'Stored token gate does not match measurements')
    require(evaluate(runs)==report['gate'],'Stored original gate does not match measurements')
    require(evaluate([r for r in runs if r['variant']!='baseline'],baseline='cache-reference')==report['cacheReferenceGate'],'Stored frozen-reference gate does not match measurements')
    require(evaluate_candidate_reference(runs,'0')==report['candidateReferenceGate'],'Stored same-build gate does not match measurements')
    expected_exit=0 if all(report.get(g,{'passed':True})['passed'] for g in GATES) else 1
    require(report['publication']['harnessExit']==expected_exit,'Stored exit does not match the complete gates')
    rows=[]
    for case in cases:
        chosen=[r for r in runs if r['case']==case['case']]
        require(all(r['inputSha256']==case['inputSha256'] and r['duration']==case['durationSeconds'] for r in chosen),'Input identity mismatch')
        row={**case,'variants':{}}
        for variant in variants:
            measurements=[r for r in chosen if r['variant']==variant]
            times=[r['wallSeconds'] for r in measurements]
            require(all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times),'Invalid wall time')
            row['variants'][variant]={'medianWallSeconds':statistics.median(times),'minWallSeconds':min(times),'maxWallSeconds':max(times),
                'allComplete':all(r['complete'] for r in measurements),'overlaps':sum(r['concurrentMoss'] for r in measurements),
                'outputHashStable':len({r['outputSha256'] for r in measurements})==1,'tokenCountStable':len({r['tokens'] for r in measurements})==1}
        for name,gate in selected.items():row[name]=next(x for x in gate['comparisons'] if x['case']==case['case'])
        rows.append(row)
    scopes={}
    for name,categories in [('all',None),('recordedSpeech',{'recorded-speech'}),('syntheticSpeech',{'synthetic-speech'}),('controls',{'control'})]:
        chosen=[r for r in rows if categories is None or r['category'] in categories]
        total={v:sum(r['variants'][v]['medianWallSeconds'] for r in chosen) for v in variants}
        scopes[name]={'caseCount':len(chosen),'inputSeconds':sum(r['durationSeconds'] for r in chosen),'sumCaseMedianWallSeconds':total,
            'originalTo48Speedup':total['baseline']/total['48'] if chosen else None,
            'originalTo48LessWallPercent':100*(1-total['48']/total['baseline']) if chosen else None,
            'sameBuild0To48Speedup':total['0']/total['48'] if chosen else None,
            'quiet48To48Speedup':total['cache-reference']/total['48'] if chosen else None,
            'exactOriginalOutputParity':all(r['originalTo48']['exactOutputParity'] for r in chosen),
            'perCaseOriginalSpeedGatePass':all(r['originalTo48']['speedPass'] for r in chosen)}
    return {'schema':'cpu-speedup-summary-v2','corpus':protocol['schema'],'cases':rows,'aggregates':scopes,
        'measuredRuns':len(runs),'discardedWarmups':protocol['discardedWarmups'],'repeats':repeats,
        'summarizedOriginalAnd48Runs':len(cases)*2*repeats,'fullTokenTraceRuns':len(cases)*2*repeats,
        'completeRuns':sum(r['complete'] for r in runs),'concurrentMossRuns':sum(r['concurrentMoss'] for r in runs),
        'selected48Gates':selected,'artifactGate':report['artifactGate'],'tokenTraceGate':traces,
        'sourceCommit':report['sourceCommit'],'sourceTree':report['sourceTree'],'ggmlCommit':report['ggmlCommit'],
        'modelSha256':report['modelSha256'],'binarySha256':report['binarySha256'],
        'generalHarnessExit':report['publication']['harnessExit'],
        'generalGateFailures':{g:report[g]['failures'] for g in ('gate','cacheReferenceGate','candidateReferenceGate')},
        'aggregationMethod':'Median of each case and variant, then divide the sum of original medians by the sum of opt48 medians. No case excluded.',
        'productionDeployed':False,'tenfoldGoalAchieved':False,
        'publicHardware':report['cpu'],
        'publication':{'personalMachineNamesAndPathsRemoved':True,'hostActivityRemoved':True},
        'limitations':['Fresh CLI processes with warm filesystem cache; wall includes model load.',
            f'Shared host and {repeats} repeats; no confidence interval or absolute WER/DER measurement.',
            'Distinct PCM inputs are not independent recordings; excerpts and room fixtures share source audio.',
            'Synthetic clips test output retention, rather than natural-conversation accuracy.',
            'Original/frozen builds expose output/count/EOS checks; full token IDs are compared only between same-build opt0 and opt48.',
            'The general frozen-reference gate also compares the slower opt0 control; its failures are retained separately.']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=json.loads(args.report.read_text());protocol=json.loads(args.protocol.read_text())
    require(report['publication']['pathLabelsAnonymized'] and report['publication']['hostActivityRemoved'],'Use the anonymous public report')
    summary=summarize(report,protocol)
    summary['sourceReportSha256']=hashlib.sha256(args.report.read_bytes()).hexdigest()
    summary['protocolSha256']=hashlib.sha256(args.protocol.read_bytes()).hexdigest()
    summary['privateCaptureReportSha256']=report['publication']['privateCaptureSha256']
    args.output.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
