# Exact inline half conversion for dense VNNI

This standalone probe pairs inline F16C scale conversion with the prior serial
and parallel VNNI kernels, pinned dot and ordinary prequantized GGML graph. It
preserves original Q8 weights, half scales, wrapped -128 sign behavior, eight
four-code FMA chains and the pinned horizontal-add tree. No MOSS model route or
production change is added. Two encoder-shaped matrices improve; the contraction
still regresses, so full-model gain remains unproven.

## Both frozen attempts

V1 compiled local `b79bbb9dbb6faaa64e572a4bb494ce9c36ea6cf4` equals public
`0d7ebd38f45da51e28a2f94432e30e283a7970c9`, tree
`f3193554afd3441b072b5dc9a0dd90d6ae505fbe`. Native GCC compilation, 15 CTests
(22.33 seconds) and 34 Python checks pass. The first exhaustive converter audit
then finds four raw-bit differences and exits 1 before matrix arithmetic or
timing. For half +0 with downward rounding, pinned software conversion produces
-0; F16C alone produces +0. The mismatch occurs in all four FTZ/DAZ settings.
Original MXCSR is restored. Exact failed source and
[`vnni-inline-failure-v1.json`](vnni-inline-failure-v1.json) retain the finding,
diagnostic records and source/artifact/log/controller hashes. The failed
attempt is retained without editing or replaying its timing.

V2 adds the precise compatibility case: inline conversion returns -0 for half
+0 under downward rounding. MXCSR is read once per tile. It preserves all other
F16C conversions and arithmetic order. Compiled local
`815c46616909dfd7bf7c0930b68225b3b6fda7d4` equals public
`4c1a2b25eb1d57d8efdd7a7164570e2c22f69c4c`, tree
`3e3cd7128a177c40d07069c5fd8b93f44827d5aa`. Its separate native source is clean,
pinned to ggml eced84, and compiled with GCC 13.3 on Xeon Gold 5416S.

Each run tests all 65,536 half payloads under sixteen rounding/FTZ/DAZ modes,
including zeros, subnormals, infinities and NaNs, restoring the exact original
MXCSR afterward. Arithmetic and benchmark invocations together pass
**2,097,152 conversion comparisons** with zero differences. V2 passes fifteen
CTests in 22.34 seconds and 34 Python checks. All **78 arithmetic** and **111
protected benchmark** shape records match both pinned-dot and ordinary-graph
output float bits, original source/weight/input/graph bytes, reconstructed panel
codes/scales, inactive columns and ABI padding. Benchmark records comprise
**30 normal timed** and **81 arithmetic-only edge** controls. Edge zero durations
are not timings. The three M1500 encoder shapes use normal synthetic input only.

All cases and source/compiler/artifact/library/log hashes are retained in
[`vnni-inline-native-v2.json`](vnni-inline-native-v2.json). Build Release with
MT_BUILD_TESTS=ON and run `build/tests/moss_vnni_q8_inline`, then the same command
with `--benchmark`. This standalone target is outside default CTest and returns
77 for unsupported ISA or incompatible pinned dot traits.

## Paired total observations

Five variants rotate/reverse order over six rounds; discard the first and report
the median of five. Candidate totals include allocation, initialized packing,
scale conversion, computation, scatter and cleanup. F32-to-Q8 input conversion,
model loading, audio, attention and full generation are excluded. Worker budget
and static contiguous weight-row ownership match. Separate stage diagnostics
include an extra timestamp barrier absent from primary totals, so their stage
medians are not additive and are not the same samples as total medians.

| Workers | K / N / M | Ordinary GGML µs | Prior parallel µs | Inline parallel µs | GGML / inline |
|---:|---|---:|---:|---:|---:|
| 1 | 1024 / 1024 / 64 | 1298.547 | 1113.658 | 963.194 | 1.3482× |
| 16 | 1024 / 128 / 16 | 17.285 | 18.429 | 16.820 | 1.0277× |
| 16 | 3072 / 128 / 16 | 25.173 | 28.770 | 26.807 | 0.9391× |
| 16 | 1024 / 1024 / 64 | 141.031 | 144.812 | 115.737 | 1.2185× |
| 16 | 1024 / 4096 / 64 | 972.862 | 813.490 | 736.435 | 1.3210× |
| 16 | 1024 / 1024 / 1500 | 2913.227 | 3533.851 | 2739.117 | 1.0636× |
| 16 | 1024 / 4096 / 1500 | 11439.567 | 12962.720 | 9386.736 | 1.2187× |
| 16 | 4096 / 1024 / 1500 | 11098.733 | 13748.899 | 13279.041 | 0.8358× |

Inline conversion improves 23 of 30 paired totals versus the original parallel
kernel; seven are slower. Twenty-one are faster than ordinary GGML; nine are
slower. Every actual-width one-worker control improves against ordinary GGML.
The encoder expansion is 1.2187× faster, but contraction is 19.6% slower. These
shared-host operator observations establish neither a useful complete-model
gain nor the original quiet opt48 full-input tenfold target.

Exact-binary disassembly retains two converter PLT calls and a 2368-byte frame
in the prior tile. The inline tile has zero converter calls and a 1152-byte frame.
Its inner block loop (0x3980–0x3d81) keeps accumulators in registers, with zero
vector stack accesses; remaining stack accesses are outside that loop. The
remaining call is the stack-protection failure path. This direct compiler
evidence supports the hypothesis; it does not isolate a causal latency share.

Both controllers restore automatic processing to 1 and verify authenticated
health plus unchanged production binary. V2 exits zero; V1 exits one after its
failed gate. The owned watchers are stopped after terminal controllers.
Ten one-second V2 probe-active snapshots observe no concurrent MOSS at shared
load 7.48–8.17. Shorter overlaps cannot be excluded; other jobs remain active.
Own source/evidence is MIT, dependency licenses retained. Private audio, text,
raw tokens, weights and credentials are absent from these artifacts.

## Next concrete experiment and primary-source review

Current weight-row-first scheduling rescans the entire activation panel buffer
for each weight tile. Test input-tile-first traversal within each worker's same
contiguous weight-row partition to reuse a panel and the worker's weight subset
in cache. Retain the old schedule as a paired control with the same exactness
and allocation/packing boundaries. The [cache traversal probe](vnni-cache.md) now records this experiment;
its results have their own paired timing and exactness boundaries.
A viable schedule then needs actual F32 conversion/graph callback, layer/cache,
complete token/output/EOS and matched full-input latency gates.

[Intel's compiler article](https://www.intel.com/content/www/us/en/developer/articles/technical/whats-new-in-llvm-for-4th-gen-intel-xeon-processor.html)
shows F16C conversion instructions replacing software calls. That is compiler
context, not proof of this GGML converter's bit parity; our exhaustive audit
provides the latter for the tested host/modes.

[Ada-MK, May 12 2026](https://arxiv.org/html/2605.11581) uses offline dependency
and resource scheduling, with a hybrid prefill/decode engine. Its evaluations
use GPTQ-W4A16 Qwen on an NVIDIA L20 and generation throughput across batch
sizes. Static scheduling and phase-specific controls are transferable ideas;
its numeric format, GPU implementation and throughput gains do not establish
lossless Q8 CPU or tenfold full-input MOSS latency.
