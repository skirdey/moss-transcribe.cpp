"""Past-only ngram draft replay on private *actual greedy* MOSS token traces.

This is an oracle acceptance/work audit, not batched target inference. It neither
proves numerical equivalence nor measures speed. Output contains aggregate counts
and hashes only; never publish the input traces, which reconstruct transcripts.
Each target call starts with a known greedy seed token and proposes up to N
following tokens from the already committed prefix plus that seed. A real target
must verify those proposals and rollback KV state after the first mismatch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def draft(prefix, maximum=8, minimum_ngram=2, maximum_ngram=16):
    if maximum < 1 or minimum_ngram < 1 or maximum_ngram < minimum_ngram:
        raise ValueError('Invalid draft parameters')
    for size in range(min(maximum_ngram, len(prefix)-1), minimum_ngram-1, -1):
        suffix = prefix[-size:]
        # Latest matching occurrence with at least one already known continuation.
        for start in range(len(prefix)-size-1, -1, -1):
            if prefix[start:start+size] == suffix:
                return prefix[start+size:start+size+maximum]
    return []


def replay(tokens, eos, maximum=8, minimum_ngram=2, maximum_ngram=16):
    if maximum < 1 or minimum_ngram < 1 or maximum_ngram < minimum_ngram:
        raise ValueError('Invalid draft parameters')
    if not tokens or tokens[-1] != eos or eos in tokens[:-1]:
        raise ValueError('Need one complete greedy sequence with terminal EOS')
    cursor = calls = proposed = accepted = input_work = drafted_calls = 0
    histogram = {}
    while cursor < len(tokens):
        # This seed is the current cached target argmax, available before a call.
        seed = tokens[cursor]
        if seed == eos:
            break  # Greedy generation stops without evaluating the EOS token.
        prefix = tokens[:cursor+1]
        proposal = draft(prefix, maximum, minimum_ngram, maximum_ngram)
        matched = 0
        for candidate, target in zip(proposal, tokens[cursor+1:]):
            if candidate != target:
                break
            matched += 1
            if target == eos:
                break
        calls += 1
        drafted_calls += bool(proposal)
        proposed += len(proposal)
        accepted += matched
        input_work += 1+len(proposal)
        committed = 1+matched
        histogram[str(committed)] = histogram.get(str(committed), 0)+1
        cursor += committed
    sequential = len(tokens)-1
    return {'tokensIncludingEos':len(tokens), 'sequentialTargetCalls':sequential,
            'oracleReplayTargetCalls':calls, 'callsWithDraft':drafted_calls,
            'proposedTokens':proposed, 'acceptedDraftTokens':accepted,
            'proposalAcceptanceFraction':accepted/proposed if proposed else 0,
            'targetInputTokenWork':input_work,
            'targetInputWorkRatio':input_work/sequential if sequential else 0,
            'callReductionFactor':sequential/calls if calls else 0,
            'committedTokensPerCallHistogram':histogram,
            'reachedEos':cursor >= len(tokens) or tokens[cursor] == eos}


def read_trace(path):
    contents = Path(path).read_bytes()
    chunks = re.findall(rb'BENCH_TOKEN_TRACE eos=(\d+) total=(\d+) offset=(\d+) ids=(\[[0-9, -]*\])', contents)
    legacy = re.findall(rb'BENCH_TOKEN_TRACE eos=(\d+) ids=(\[[0-9, -]*\])', contents)
    generation = re.findall(rb'BENCH_GENERATION tokens=(\d+) stop=(\w+)', contents)
    record_count = len(re.findall(rb'BENCH_TOKEN_TRACE ', contents))
    if len(generation) != 1 or record_count != len(chunks)+len(legacy):
        raise ValueError('Truncated trace or missing/duplicate generation record')
    if chunks:
        if legacy: raise ValueError('Mixed trace formats')
        eos = int(chunks[0][0]); total = int(chunks[0][1]); tokens = []
        for chunk_eos, chunk_total, offset, values in chunks:
            decoded = json.loads(values)
            if int(chunk_eos)!=eos or int(chunk_total)!=total or int(offset)!=len(tokens) or not decoded:
                raise ValueError('Missing, reordered, duplicate or inconsistent trace chunks')
            tokens.extend(decoded)
        if len(tokens)!=total: raise ValueError('Incomplete trace tail')
    else:
        if len(legacy)!=1: raise ValueError('Need one complete token trace')
        eos = int(legacy[0][0]); tokens = json.loads(legacy[0][1])
    if not isinstance(tokens,list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('Token IDs must be nonnegative integers')
    if generation[0][1] != b'eos' or len(tokens) != int(generation[0][0]):
        raise ValueError('Trace differs from complete generation record')
    return tokens, eos, hashlib.sha256(contents).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+', type=Path)
    p.add_argument('--maximum', nargs='+', type=int, default=[2,4,8])
    args = p.parse_args()
    rows = []
    for path in args.logs:
        tokens, eos, sha = read_trace(path)
        for maximum in args.maximum:
            rows.append({'case':path.stem,'privateTraceSha256':sha,'maximumDraftTokens':maximum,
                         'minimumNgram':2,'maximumNgram':16,**replay(tokens,eos,maximum)})
    print(json.dumps({'protocol':'Past-only oracle greedy-trace replay. No target execution, latency or quality claim; input work includes rejected drafts. Known seed plus longest/latest ngram, no future tokens available to draft.',
                      'auditScriptSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'runs':rows},indent=2))


if __name__ == '__main__':
    main()
