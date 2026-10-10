# Attention cache, matmul and copy diagnosis

The [original stage trace](target-batch-trace.md) placed the first observed batch
discrepancy at attention context. This extension separates actual cache stores,
the value/probability matmul, its permutation/contiguous copy and the pinned
CPU dot routine. It changes observation, not decoder arithmetic. The strict
batch candidate and the tenfold full-input goal remain unachieved.

## Protocol and capture controls

Keep the original four independent states, immutable model, complete cloned
cache, semantic graph/callback checks and quiet-versus-capture hidden/KV checks.
Retain the raw value matmul output before its permutation. Copy active K/V
storage in canonical layouts, verify every new cache slot against its captured
K-RoPE or V-projection producer, and compare raw matmul with the selected context
copy. Replay `ggml_vec_dot_f32` from the same pinned CPU shared library at both
the common batch length and each query's valid causal length. Require full-length
replays to reproduce native raw outputs; per-query replays are diagnostic.
Only numeric counts/errors are written, never vectors or token IDs.

All 96 original null-hook case records and their summary must match exactly.
The expanded observation must also reproduce all 26 old trace case records,
13,832 stage records and the summary. Every fixture has 28 operation records,
giving 728 rows. Copy, store and full-length replay errors invalidate expanded
attribution even when final hidden/KV bits are preserved. Separate nonmodel
controls exercise distinct tensor-layout markers, grouped-query head mapping,
unsupported layouts and changed output-retention flags.

## Preserved failed observation

V1 passed 19 native CTests (25.60 s) and 40 Python tests, and reproduced the full
96-case matrix and all old trace records exactly. Its native trace returned 0
under the old final-result controls. The new internal checks found 168 failing
layer records across all six single-token controls: intermediate storage had
been overwritten. This run is **ineligible for expanded attribution**. Its
[complete report](context-ops-native-v1.json) and [728 operation records](context-operations-v1.jsonl)
are retained with exact source, binary, model and log hashes.

The pinned GGML allocator reuses reservations based on node/source sizes; those
checks do not notice changed output flags. Marking an intermediate as an output
after a quiet graph of the same shape can therefore reuse a plan that recycles
that intermediate. Matching final results cannot establish its lifetime.

V2 adds an explicit fresh-reservation option to `compute_graph_with_inputs`,
defaulting to false. Only the private nonnull decoder audit hook requests it;
ordinary prefill and single-token decode retain their allocation behavior.
The nonmodel control computes a known intermediate and final result in fresh
contexts with equal graph shapes. The stale plan gives 64 wrong intermediate
elements; forced reservation gives zero, while both final results remain exact.
The trace now rejects internal copy/store/replay failures before attribution.

## Repaired native evidence and next kernel

V2 passed 20 native CTests (25.71 s) and 40 Python tests. The full 96-case
null-hook matrix matches the original case records and summary exactly. The
26-case trace returns 0, reproduces all 13,832 old stage records and has zero
internal failures in 728 operation rows. Every raw-context copy, new cache
store and native/full-length dot replay is exact, including the six previously
invalid single-token controls. Replay operands, prefix, inputs, positions and
980,917,056 loaded weight bytes remain unchanged; nonfinite counts are zero.

In layer 0 of every fixture, corresponding valid K/V inputs and attention
probabilities match. The full-length dot replay reproduces actual native
batch matmul, while the valid-length replay reproduces actual serial matmul.
All 13 fixtures with first-context drift have exactly the same bit-difference
count between full-length and valid-length replays as between native serial
and batch outputs. The other 13 fixtures have zero in both comparisons.
This identifies dot-length-dependent arithmetic at the first divergence in
these fixtures. It does not establish a repaired full decoder or later-layer
state parity.

The [native GCC 13.3 disassembly](context-dot-disassembly-v2.txt) shows separate
multiply/add instructions for vectorized eight/four-element portions of the
tail and scalar fused multiply-add for the final one to three elements.
For example, changing length 33 to 36 moves a valid term between these paths.
Even finite zero probabilities at future positions can therefore change the
rounding applied to a valid product. No fast-math assumption is needed for this
specific emitted code; native compile flags are retained in the report.

The next candidate should call the same pinned dot at each query's valid
length, distributing independent rows across the existing CPU workers. It must
remain disabled by default and pass the complete hidden/logit/KV/rewind/poison
matrix before any timing or speculative integration. No candidate kernel is
implemented or promoted by this observation change.

The [repaired report](context-ops-native-v2.json) has SHA-256
`fb3bdbcf37d70fe60dc54b5754396f147fbf958e44d1d6e9e6003c26f716c727`;
the [complete operation rows](context-operations-v2.jsonl) have SHA-256
`b60190956a0dd36c89f5b7fd31696ee2748910a115d26f7f17f7fabdffbc02fa`.
Source and public exact revisions, binary/library/model/log hashes, compiler
flags and dynamic-library resolution are recorded. The 362 active one-second
observations found no overlapping MOSS process; host load ranged 5.74–10.82.
Shorter overlaps are not excluded and other host jobs remained running.
Automatic processing and the authenticated production API were restored.

A later [causal-context kernel and full-state gate](causal-context.md) implement
the proposed valid-length dots and pass all 96 state/logit/rollback fixtures.
Its separate paired append/full-logit timings retain every sample and a slower
long-prefix T2 fixture; they do not measure full audio latency.

## Reproduce

Build the pinned source and GGML revision with Release CPU flags as in the
original trace documentation. Run default checks, then the full matrix and
expanded trace on the same Q8 checkpoint:

```sh
ctest --test-dir build -LE model --output-on-failure
build/tests/moss_target_batch_trace --audit-allocation
build/tests/moss_target_batch_trace --audit-context-layout
taskset -c 0-15 build/tests/moss_target_batch_gate /path/to/moss-transcribe-q8_0.gguf
taskset -c 0-15 build/tests/moss_target_batch_trace /path/to/moss-transcribe-q8_0.gguf --context-operations
```

The unchanged strict matrix returns 1 for its eight rejected cases. Trace exit 0
with zero operation-control failures establishes eligible observation, not
batch parity, a measured speedup, a formal proof or an audio accuracy result.
