# Stateful CPU target-batch audit

This model-backed research executable tests the existing private T>1 decoder
path against repeated `decode_one` calls at a nonzero cached prefix. It defines
a lab-only friend accessor; it adds no production append/rewind API, changes no
decoder math, and does not implement a speculative generator.

Native hp-fury result: **strict parity FAIL; timing blocked**. All96 cases
completed with finite outputs and passing fixture/state/weight guards;88 were
exact and8 differed. Greedy choices and EOS decisions matched in all456
comparisons, illustrating why those checks alone cannot establish losslessness.

The strict acceptance rule includes every hidden float, every full-vocabulary
logits row and every active per-layer K/V float bit. Matching greedy token choices
alone is insufficient. The executable returns 0 only for complete exact parity,
1 for completed arithmetic/state mismatch, and 2 for an exception or incomplete
fixture. It never measures performance. A failed gate forbids downstream timing
or promotion of that candidate; passing it would still require a matched timing
experiment and actual audio/full-input correctness tests.

## Reproduce

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMT_BUILD_TESTS=ON
cmake --build build -j 16
ctest --test-dir build -LE model --output-on-failure
(cd cpu-lab && python3 -m unittest test_moss_cpu_regression test_moss_ngram_audit)
taskset -c 0-15 build/tests/moss_target_batch_gate /path/to/moss-transcribe-q8_0.gguf
```

Choose one CPU per physical core using the host topology before setting affinity.
The executable forces the CPU backend, opt48, startup/phase thread budgets16,
and `OMP_NUM_THREADS=16`/`OMP_DYNAMIC=FALSE`. Its model input is not distributed
in this repository. The probe is deliberately outside default CTest.

## Protected work

One model is loaded and its small F16 tensors promoted using the ordinary loader.
The probe copies every loaded tensor's bytes afterward, then checks that complete
snapshot after the arithmetic suite. Each case has two independent decoder/KV
owners sharing read-only model weights. All computations run serially through
the process-wide backend; independent KV ownership does not enable concurrent
backend calls.

The 96 cases cross two prefix modes with twelve prefix lengths and four append
sizes. Prefix mode0 uses deterministic uniform F32 embeddings in approximately
[-0.2,0.2); mode1 uses actual dequantized model token embeddings from deterministic
IDs. Prefix lengths are31/32/33,127/128/129,511/512/513,1023/1024/1025, and
append sizes are1/2/4/8. Appended embeddings always come from the actual model.
Both states start with identical prefix hidden states and active cache bits.
Their cache capacity is1088 positions; the transposed V layout is canonicalized
as position/head/feature before comparison.

After append, the probe checks every hidden/logit/KV bit, first-maximum token
choice and EOS decision, finite outputs, final positions, and prefix/input
immutability. It then rewinds to `prefix+floor(T/2)` and verifies that negative
or forward rewinds leave state unchanged. In the batch owner it poisons the
**entire discarded suffix** of both caches with quiet NaNs and checks byte-exact
readback and preservation of the accepted prefix. Both owners decode a different
actual token and compare complete hidden/logit/active-cache outputs again;
previously accepted cache entries must remain unchanged. This also tests rewind
to the original prefix in T1 controls. Inactive bytes are copied as integer bits
and are not used in arithmetic by the cache auditor.

Arithmetic mismatches are retained through the complete matrix. Fixture failures
terminate immediately. Numeric rows contain counts/errors/state flags only;
no token IDs, embeddings, logits, KV values, weight bytes, private audio,
transcripts or credentials are printed.

## Frozen native evidence

The [complete numeric report](target-batch-native-v1.json) retains all96 rows,
summary counts, compiler flags, source/library/binary/log hashes, the failure
matrix, and terminal restoration state. Frozen source is
`ade6b8b999c278a99404053553e2a6cb74f8d29d`, tree
`7388780e9bbed80b2198f2a4d19d574e054e2dee`; public source commit
`4c5e4d21a034367b2a2835380d051a1e548d6ea2` has the identical tree.
GGML is pinned to`eced84c86f8b012c752c016f7fe789adea168e1e`.
The isolated native root is`/home/stan/hw-moss-target-batch-v1/source`, left clean.
GCC13.3 Release builds the actual Xeon path; GGML enables native ISA, OpenMP,
CPU repacking and llamafile. Loaded dynamic libraries resolve inside this root.

Default checks pass:17 non-model CTests in25.47s and40 Python regression/audit
checks. The standalone model probe deliberately exits1 for its observed mismatch;
this is retained evidence of the existing path failing the strict gate, not a
passing batch candidate or a production change.

| Append size | Exact cases /24 | Failed cases |
|---|---:|---|
| T1 | 24 | None |
| T2 | 24 | None |
| T4 | 22 | Mode0 prefix32; mode1 prefix1024 |
| T8 | 18 | Mode0 prefix32/512/1023; mode1 prefix512/1024/1025 |

Append comparisons cover368,640 hidden floats and54,696,960 logits. They find
37,888 hidden,5,621,615 logit and1,087,486 active-KV bit differences; the maximum
hidden absolute difference is2.4924144744873047. Rollback/changed-token checks
also find1,264,507 total hidden/logit/KV bit differences. Every original prefix
and accepted-cache immutability check passes. Thus affected accepted batch states
can remain different when a changed token is decoded; suffix isolation does not
repair already different accepted entries.

All980,917,056 loaded tensor bytes remain unchanged. Inputs and prefix states
are preserved, invalid rewinds leave state unchanged, poisoned suffix readback
passes, and every append/changed output is finite. One-second observation records
251 active probe snapshots with no competing MOSS process; one-minute host load
ranges6.91–11.63. Other host work was left running, and shorter overlaps are not
excluded. No timing samples, full-input latency or audio correctness are measured.
Automatic processing is restored to1; the authenticated API identity remains
the existing production binary, and the owned watcher is stopped.

## Interpretation and next work

Synthetic and token-embedding fixtures are useful state/bit proofs, but they do
not establish audio transcript parity or a speculative decoding latency gain.
The T>1 path has a causal mask and a common KV extent; opt48's T1 attention uses
its specialized CPU softmax. GGML matrix dispatch and reduction lengths can also
differ with query count. These are hypotheses to isolate, not an attribution
of the observed mismatch without intermediate-operator comparisons.

Next isolate the first differing layer/operator and repeat the failing fixtures
before changing reductions. T2 is exact in this finite matrix, but that observation
does not justify silently dropping the failed T4/T8 cases or promoting arbitrary
batched inputs. A separately declared T2-only candidate would need its own frozen
gate, matched append plus all-logits timing, and eventually complete draft/audio/
full-input tests. Preserve this failed source/report while exploring a new root.

[Lossless but Not Free (19 July2026)](https://arxiv.org/abs/2607.17283) measures
consumer-hardware speculative verification and draft costs: its best configuration
reaches1.61× wall-clock speedup, while three of five configurations slow down.
It validates distribution and greedy sequence agreement. Those criteria do not
replace this decoder's complete state/float-bit gate. Our inference is to measure
the actual append plus all-logits work only after its correctness boundary passes.

[SpecStream (27 September2026)](https://arxiv.org/abs/2609.33184) separates committed
history from local candidate rollback and shares streamed KV chunks across queries.
Its GPU offloading throughput results and task-quality checks are not evidence for
exact CPU state or single-input latency here. Our inference is to retain explicit
committed/rejected cache ownership; online softmax or changed reduction trees
would need an independent bit proof.

The original tenfold objective still uses the immutable quiet opt48 full-input
baseline including model load. Operator timing, warmed service latency, fewer
target calls and batch throughput cannot substitute for that denominator.
