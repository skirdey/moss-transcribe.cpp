#!/usr/bin/env python3
"""MIT. Summarize existing numeric phase evidence; do not rerun inference."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def audit(path, variant):
    data = path.read_bytes()
    report = json.loads(data)
    selected = [r for r in report['runs'] if r['variant'] == variant]
    # The harness marks rc=1 as complete only for empty stdout at EOS.
    if not selected or any(not r['complete'] or r['returncode'] not in (0, 1)
                           or r['stop'] != 'eos' or r['concurrentMoss'] for r in selected):
        raise ValueError('Missing, incomplete, failed or concurrent evidence')
    records = []
    for case in sorted({r['case'] for r in selected}):
        runs = [r for r in selected if r['case'] == case]
        if any(not r.get('phaseProfile') for r in runs):
            raise ValueError('Missing phase instrumentation')
        median = lambda values: statistics.median(values)
        phases = {}
        for name in ('mel', 'whisper', 'adaptor', 'prefill', 'decode', 'logits'):
            phases[name] = {
                field: median([r['phaseProfile'][name][field] for r in runs])
                for field in ('totalSeconds', 'buildSeconds', 'allocateSeconds', 'computeSeconds')
            }
        records.append({
            'case': case, 'samples': len(runs),
            'durationSeconds': runs[0]['duration'],
            'wallSeconds': median([r['wallSeconds'] for r in runs]),
            'loadSeconds': median([r['timing']['load'] for r in runs]),
            'inferenceSeconds': median([r['timing']['inference'] for r in runs]),
            'hostLoadRange': [min(r[key][0] for r in runs for key in ('hostLoadBefore', 'hostLoadAfter')),
                              max(r[key][0] for r in runs for key in ('hostLoadBefore', 'hostLoadAfter'))],
            'phases': phases,
        })
    return {
        'sourceReportSha256': hashlib.sha256(data).hexdigest(),
        'sourceCommit': report['sourceCommit'], 'sourceTree': report['sourceTree'],
        'variant': variant, 'samples': len(selected), 'records': records,
        'protocol': 'Medians of existing paired fresh-process report, two samples per case; phase timings can overlap and medians are not additive.',
        'quietBaseline': False, 'newInferenceRun': False,
        'tenfoldAchieved': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--variant', default='48')
    args = parser.parse_args()
    print(json.dumps(audit(args.report, args.variant), indent=2))
