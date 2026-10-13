# CPU past-only speculation: rejected full-input candidate

The default-off opt786480 prototype preserves all controlled native fixtures but
fails the complete-audio pilot. Both German repeats change the full output and
token trace; speech latency increases10–41% over frozen opt48. This is research
source and negative evidence. Production continues to use its existing binary.
The original10x full-input goal remains unachieved. No corpus score is claimed
for a candidate that already fails exact output and latency gates.

## Implementation and scope

Bit524288 enables a CPU-only generation controller when causal context262144 and
transposed cache16 are also selected. With the established48 settings, opt is
786480. Other settings retain the original serial generator. The drafter uses
only committed output plus the known seed: longest/latest repeated ngrams of
length2–16, at most four proposed tokens. No trace, future token or extra model
is available to the runtime drafter.

The target appends the seed plus proposals as actual T1–5. Every accepted token
must equal the target's first maximum after the same last100-token repetition
penalty, clamped to1–1.5. Logits are computed lazily for the necessary rows.
Rejected cache positions are excluded by a checked logical rewind. EOS,
token-limit and the existing empty-marker guard are checked after each emitted
token, with the terminating token excluded from consumed cache. Default hooks
are null; the lab uses them to inspect target states and poison discarded slots.
Matching the batched target does not itself establish equivalence to serial
floating-point arithmetic on every real audio prefix.

## Frozen provenance and native controls

Runtime source35036546f5aade49479728369f6cdfe56b3855fe has tree
818f4743420e70d7c5407d6758bebc39b579273e. Public source8382623c2014f85881f0680a0d8f10ce414b20b0
has the same tree. GGML is eced84c86f8b012c752c016f7fe789adea168e1e; native
GCC13.3 Release runs on a XeonGold5416S with16 physical cores and all phase
budgets16. Model SHA and binary/library/source hashes are in the reports.

[Complete native evidence](strict-speculative-native-v1.json) records22 default
CTest and40 existing Python checks,384 native context oracle controls,
3840 invalid-layout rejections, and all96 unchanged original baseline records.
The opt48 baseline still reproduces the original eight batch-state failures;
it is not replaced by a more permissive reference.

The extended causal-context gate passes all120 prefix/mode/T1–5 cases, including
every hidden float, full vocabulary logit, active/prefix KV byte, rewind,
changed-token decode and poisoned suffix. Actual context dispatch is28 per
multi-query append and0 for T1. The separate generator gate passes114/114 cases:
prefix31/128/513/1024, penalties1/1.1/1.5, forced accepted lengths0–4,
limits1/2/5/10, first-token virtual EOS and six virtual EOS stops inside a batch.
It compares every appended row against actual serial decode, then checks every
commit and poisons the entire discarded capacity. Six fixtures without a unique
virtual EOS token are skipped by the declared fixture policy; they are not
missing runs. All980917056 loaded weight bytes remain unchanged.

These are finite controlled fixtures, not a proof for real encoded audio or
all cache capacities. The first controller lost its SSH observation after the
length gate actually completed with exit0. The completed120 cases were inspected
and retained; only the previously unstarted generation gate resumed. Initial
partial and resumed observation logs are hashed and their coverage is stated.

## Complete full-input pilot

[All48 measured runs](strict-speculative-pilot-v1.json) compare production0,
immutable quiet opt48, same-build48 and actual786480. There are six cases,
two repeats, alternating variant order and one discarded full warmup per
variant. Fresh processes include model loading and all draft, rejected-target,
verification, logits, rewind and transcription costs; OS file cache is warm.
Automatic ingestion is paused during measurements and restored afterwards.
Half-second process monitoring detects no other MOSS process in any measured run;
other host jobs remain running. No reboot or disk-cache eviction is performed.

| Case | Frozen48 median seconds | Speculative median seconds | Latency increase | Exact full output |
|---|---:|---:|---:|---|
| English60s |13.8179|17.5540|27.04%|yes|
| German60s |11.6384|12.9425|11.20%|**no**|
| Public60s |11.9642|13.2178|10.48%|yes|
| English120s |29.9251|42.1101|40.72%|yes|
| Silence60s |7.4405|7.4804|0.54%|yes|
| Empty real60s |7.3554|7.3806|0.34%|yes|

All measured processes finish with EOS. Same-build48 matches the frozen reference
and complete token hashes. Both German speculative repeats first differ at
zero-based token position217:458 serial tokens versus436 speculative tokens,
including EOS. Candidate output/trace gates and the fixed five-percent latency
gates reject it. Model, source, binaries and libraries remain unchanged.
Production comparisons are retained separately; gains over slower production
cannot satisfy the faster immutable48 reference.

## Actual proposal costs

[The work audit](strict-speculative-work-v1.json) parses one complete numeric
CPU_SPECULATIVE record from each of12 candidate logs and requires its call,
proposal, acceptance, input-work and rewind counters to match independent
past-only replay of that same completed private trace. Proposal ranks distinguish
all dispatched work from rows reached by verification and accepted tokens;
seed-position buckets show how this changes across generation. No IDs or text
are published. Six parser/failure controls pass alongside40 existing Python
checks. The German work audit checks the candidate's changed trace; it does not
turn that failed output into an equivalent reference.

On matching English/public traces the target call reduction is about19–27%,
but target input work is1.60–2.13 times serial. Accepted/proposed fractions are
about0.19–0.24. Draft lookup costs are only milliseconds; excess target work is
material. Fewer calls alone therefore do not establish a latency improvement.

## Reproduction and next gate

```sh
ctest --test-dir build -LE model --output-on-failure
cd cpu-lab
python3 -m unittest test_moss_cpu_regression test_moss_ngram_audit test_moss_speculative_audit
# Model-backed diagnostics are explicit and stay outside automatic CTest:
../build/tests/moss_target_batch_gate MODEL --causal-context --speculative-lengths
../build/tests/moss_speculative_gate MODEL
../build/tests/moss_speculative_audio_gate MODEL PRIVATE_AUDIO
# Private complete runtime logs are inputs; numeric JSON is the output:
python3 moss_speculative_audit.py PRIVATE_CANDIDATE_LOGS
```

[The real-audio diagnostic](strict-speculative-audio-v2.json) uses frozen source
34a3ba96abc831b1b7d7c5bcfe31622ff9322370/public77c6dd5d6d5a1570840fdd87d8da49365ea98ffb,
identical tree13e73083423af5c8eede841a55d0d152fd1ae93d and unchanged runtime
source from the pilot. A fresh complete build passes22 native checks in25.60s
and46 Python checks. The actual audio prefix has859 tokens; both decoders have
capacity4971. After112 exact commits, append113 has T5 and first differs at row3,
query position990: all1024 hidden elements,151936 full logits and53248 active KV
values differ. Max absolute hidden/logit errors are0.570608/0.256892, with no
nonfinite values. The row's first maximum still matches, illustrating why argmax
alone is insufficient. All980917056 model weight bytes remain unchanged. This
locates a state failure but does not yet attribute it to a particular operator.
The durable remote probe completes with declared diagnostic exit1 and both PIDs
are absent; automatic ingestion is restored and the production API is healthy.

The real-audio diagnostic stops at the first hidden/full-logit/cache mismatch;
its expensive serial comparisons cannot be used as latency measurements.
A repair must reproduce the failing real prefix and capacity, preserve all
controlled fixture evidence, and pass complete audio tokens/output/EOS and the
fixed corpus and latency gates before any promotion. Even a correct repaired
drafter needs a measured batch-cost fallback or a better acceptance profile.
Encoder work remains necessary for the10x full-input target.

The primary [Lossless but Not Free study, submitted July19,2026](https://arxiv.org/abs/2607.17283)
was checked October11. Its abstract reports a best1.61x wall gain across five
consumer Apple-silicon configurations, with three slowing down; serial
verification and draft/target cost balance explain failures. Its greedy and
statistical distribution tests support measuring real verification costs, not
assuming CPU MOSS gains or exact raw state. No implementation was imported.
All new code and evidence remain MIT; audio, transcripts, token IDs, model values
and credentials remain private.
