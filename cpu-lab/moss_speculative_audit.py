"""Check actual CPU speculative counters against private, complete greedy traces.

Only counts and hashes are emitted. Input logs contain recoverable transcripts
and must remain private. This audit establishes proposal/work accounting; latency
and numerical equivalence require separate native and full-input measurements.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from moss_ngram_audit import draft, read_trace, replay


def expected_work(tokens, eos, maximum):
    result = replay(tokens, eos, maximum)
    cursor = rewinds = 0
    ranks = {str(i): {'dispatched': 0, 'verified': 0, 'accepted': 0}
             for i in range(1, maximum+1)}
    positions = {}
    while tokens[cursor] != eos:
        proposal = draft(tokens[:cursor+1], maximum)
        bucket = str(cursor//128*128)
        entry = positions.setdefault(bucket, {'targetCalls': 0, 'proposed': 0,
                                              'accepted': 0, 'targetInputTokens': 0})
        entry['targetCalls'] += 1
        entry['proposed'] += len(proposal)
        entry['targetInputTokens'] += 1+len(proposal)
        for rank in range(1, len(proposal)+1):
            ranks[str(rank)]['dispatched'] += 1
        matched = 0
        terminal = False
        for rank, (candidate, target) in enumerate(zip(proposal, tokens[cursor+1:]), 1):
            ranks[str(rank)]['verified'] += 1
            if candidate != target:
                break
            ranks[str(rank)]['accepted'] += 1
            matched += 1
            if target == eos:
                terminal = True
                break
        entry['accepted'] += matched
        rewinds += matched != len(proposal) or terminal
        cursor += 1+matched
        if terminal:
            break
    counters = {'targetCalls': result['oracleReplayTargetCalls'],
                'draftedCalls': result['callsWithDraft'], 'proposed': result['proposedTokens'],
                'accepted': result['acceptedDraftTokens'],
                'targetInputTokens': result['targetInputTokenWork'], 'rewinds': rewinds}
    return counters, result, ranks, positions


def audit(path):
    tokens, eos, sha = read_trace(path)
    contents = Path(path).read_text()
    records = re.findall(r'CPU_SPECULATIVE ([^\r\n]*)', contents)
    if len(records) != 1:
        raise ValueError('Need exactly one actual speculative counter record')
    fields = re.findall(r'(\w+)=([^ ]+)', records[0])
    values = dict(fields)
    integer_names = ('targetCalls', 'draftedCalls', 'proposed', 'accepted',
                     'targetInputTokens', 'rewinds', 'maximumDraft')
    if len(values) != len(fields) or set(values) != set(integer_names) | {'draftSeconds', 'rewindSeconds'}:
        raise ValueError('Incomplete or duplicate actual counter fields')
    if any(not re.fullmatch(r'\d+', values[k]) for k in integer_names):
        raise ValueError('Invalid integer counter')
    actual = {k: int(values[k]) for k in integer_names}
    if actual.pop('maximumDraft') != 4:
        raise ValueError('This audit requires the frozen four-token draft policy')
    costs = {k: float(values[k]) for k in ('draftSeconds', 'rewindSeconds')}
    if any(not math.isfinite(v) or v < 0 for v in costs.values()):
        raise ValueError('Invalid draft or rewind cost')
    expected, replayed, ranks, positions = expected_work(tokens, eos, 4)
    if actual != expected:
        raise ValueError('Actual target work differs from private greedy-trace replay')
    return {'case': Path(path).stem, 'privateLogSha256': sha, 'maximumDraft': 4,
            'actualCounters': actual, 'counterReplayExact': True, 'costs': costs,
            'replay': replayed, 'proposalRanks': ranks, 'seedPositionBuckets': positions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', nargs='+', type=Path)
    args = parser.parse_args()
    rows = [audit(path) for path in args.logs]
    print(json.dumps({'schema': 'strict-speculative-work-v1',
                      'protocol': 'Actual runtime counters must equal past-only replay of complete private EOS traces. Proposal rank counts distinguish dispatched, verified and accepted work. Position buckets use the committed seed offset; no token IDs are emitted. Includes rejected target inputs. Not a latency or universal parity claim.',
                      'auditScriptSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'traceReaderSha256': hashlib.sha256(Path(__file__).with_name('moss_ngram_audit.py').read_bytes()).hexdigest(),
                      'runs': rows}, indent=2))


if __name__ == '__main__':
    main()
