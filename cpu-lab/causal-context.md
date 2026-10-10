# Exact causal context and paired batch timing

Research bit **262144**, disabled by default, implements per-query causal
context dots for CPU batch append. It fixes all eight previously rejected
fixtures: the complete 96-case model-state matrix now matches opt-48 serial
decode exactly. Paired append plus every full-vocabulary logit is faster in
11 of 12 timing fixtures and slower in one. Full audio latency, draft acceptance,
corpus accuracy and the tenfold goal remain unmeasured by this experiment.

## Runtime and safeguards

The [protected operation diagnosis](context-operations.md) identified
dot-length-dependent rounding at the first context discrepancy. The new
`src/cpu_context.*` callback calls the same pinned `ggml_vec_dot_f32` at
`past + query + 1`, reproducing the serial query's logical length. Independent
head/query/feature outputs use existing GGML workers. It allocates no callback
scratch, creates no nested worker team, borrows no userdata and changes no
weights. Grouped-query heads select the corresponding KV head.

Qwen dispatch requires the research bit, transposed V cache, a causal mask,
nonempty cached prefix, more than one query and batch size one. The helper
checks CPU backend, F32 types, compatible head counts, row/head storage,
contiguous probabilities and integer/storage bounds. Unsupported layouts return
null and retain ordinary matmul. Prefill, single-token decode, ordinary opt 48
and production API behavior retain their existing paths.

The model gate enables opt **262192** (262144 + 48) only for batch append.
Load, prefix prefill, serial reference and rollback use opt 48. It requires
28 context operations actually built and executed for every multi-query
fixture, and zero for every single-query control.

## Native exactness

GCC 13.3 Release passed 21 nonmodel CTests (26.20 s) and 40 Python tests.
The new native oracle compares against actual single-query `GGML_MUL_MAT`,
with fresh allocation plans and actual one/sixteen-worker configurations.
Its 384 cases cover sixteen prefix/vector boundaries, T2/4/8, two GQA shapes,
finite-zero or NaN future probabilities and poisoned inactive V capacity.
Every output bit matches, every operand byte stays unchanged, and all 3,840
unsupported-layout controls reject.

The unmodified opt-48 gate reproduces all 96 original case records and summary
exactly, including its strict exit 1 and eight rejected cases. The candidate
returns 0 with **96/96 exact**. Every hidden float, every vocabulary logit,
active/prefix KV byte, position, greedy/EOS decision and changed-token result
after rewind/suffix poison matches. Nonfinite counts are zero, and all
980,917,056 loaded weight bytes remain unchanged. These finite fixtures permit
timing; they do not prove universal arithmetic or audio/corpus equivalence.

The [complete native report](causal-context-native-v1.json) has SHA-256
`57236cbe9e6a3d0144f0be0ea4c0ce540f5f6224be43435c62b9325ce6c7c22a`.
It includes all 384 oracle rows, 96 baseline rows, 96 candidate rows, actual
dispatch counts, compiler flags and source/model/artifact/log hashes.
Frozen source is `3256f99b2cea1f1ebcb712d484908d7b719c1d02`, with exact public
source `0ca97a31f222d6dc3d8a415717c68a5b36310235` and tree
`c5867dd45f516febba4832060b94872e704071ee`.

## Paired timing and limits

Timing started only after the complete state gate passed. A standalone
benchmark links against the unchanged validated runtime, whose library hashes
match before and after timing. All runtime source bytes match the gated source.
The benchmark uses real token embeddings, complete owned cache snapshots,
inactive-capacity poison and a full restore/readback before each pair. One
warmup pair per fixture is discarded; five measured pairs alternate order.
All 60 measured and 12 warmup pairs preserve complete hidden/logit/active-KV
bits, prefix/input bytes and position state; loaded weight bytes stay unchanged.

The timed region includes actual append graph allocation/dispatch, context
work, extraction and every vocabulary logit row, including vector copies.
It excludes model loading, prefix setup, state snapshot/reset, state checking,
drafting, rollback and audio processing. Times below are medians of five pairs;
speedup is serial median divided by batch median. It measures known input rows,
not the cost or acceptance rate of proposing those rows.

| Prefix | Append tokens | Serial + logits, ms | Batch + logits, ms | Speedup |
|---:|---:|---:|---:|---:|
| 32 | 2 | 16.457 | 10.593 | 1.554x |
| 32 | 4 | 30.902 | 15.327 | 2.016x |
| 32 | 8 | 62.672 | 25.552 | 2.453x |
| 128 | 2 | 15.522 | 10.773 | 1.441x |
| 128 | 4 | 31.174 | 15.681 | 1.988x |
| 128 | 8 | 62.022 | 26.455 | 2.344x |
| 512 | 2 | 17.571 | 14.972 | 1.174x |
| 512 | 4 | 34.506 | 20.978 | 1.645x |
| 512 | 8 | 68.670 | 34.292 | 2.003x |
| 1024 | 2 | 18.703 | 19.426 | **0.963x** |
| 1024 | 4 | 37.761 | 26.609 | 1.419x |
| 1024 | 8 | 74.909 | 40.435 | 1.853x |

The [complete timing report](causal-context-timing-v1.json) has SHA-256
`23ace911371e12008fe06a497b65d071be3aa7150e4bc60c376ddfc259481628`.
It retains every sample, warmup, range, median, compiler argv and runtime hash.
Timing source is `638b35dad82d44a427642e7f4711f4da53ab7ef9`; exact public source
is `d0821323b5380c92001a4ef5f7f43230e1dd4317`, tree
`2ab23b4c8f9c1c3b214ea8e593584318a3ac8a7d`. A separate fresh CMake build with
the benchmark's internal include path passes all 21 nonmodel tests (25.97 s);
source `3d342f2a584c907b351fb4d5443fa1bba45183aa`, exact public
`53f0533eb5a50c3ba2620fe9ad17e0a5a81396e7`, tree
`8121172d721dbcd7c4bd724c1db101d90e0a9535`. Earlier roots remain preserved.

All model runs use physical-core affinity 0–15 and sixteen workers per phase.
One-second observations found no overlapping MOSS in 511 state-gate and
77 timing snapshots. Host load ranged 6.59–11.77 and 0.16–4.73 respectively;
shorter overlaps are not excluded and other host jobs remained running.
Automatic processing and the authenticated production API were restored.

## Reproduce and next acceptance gate

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMT_BUILD_TESTS=ON
cmake --build build -j 16
ctest --test-dir build -LE model --output-on-failure
# Baseline retains its eight known strict failures, exit 1:
taskset -c 0-15 build/tests/moss_target_batch_gate /path/to/model.gguf
# Require complete 96/96 exact and actual dispatch counts before timing:
taskset -c 0-15 build/tests/moss_target_batch_gate /path/to/model.gguf --causal-context
taskset -c 0-15 build/tests/moss_causal_context_bench /path/to/model.gguf
```

Next integrate a strict greedy-verified draft, including all proposal,
verification, rejection/rollback and logits costs. Measure actual acceptance
by token position and complete inputs, with exact transcript/token/speaker/
timestamp/EOS and silence controls, then the fixed corpus accuracy gate.
Small batches at long prefixes need a measured fallback policy. The original
full-input opt-48 reference and encoder work remain necessary for the 10x goal.
No production promotion or full-input speed claim follows from these results.
