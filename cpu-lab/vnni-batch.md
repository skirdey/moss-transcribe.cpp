# Exact dense VNNI matrix batches

This is a standalone owned-vector experiment with no MOSS model route or
production change. It replaces the [AMX batch probe](amx-batch.md)'s repeated
tile zero/store/reload work with four-code reductions in vector registers.
The native arithmetic gate passes. Timing is mixed: some matrix shapes improve,
but the encoder-shaped control is slower, so there is no full-model promotion.

## Arithmetic and protected controls

[`moss_vnni_q8_batch.cpp`](moss_vnni_q8_batch.cpp) retains original row-major Q8
weight codes and half scales. Two weight rows consume a dense sixteen-input
panel. Each VNNI instruction reduces four codes, with eight separate float FMA
chains and the pinned horizontal-add tree. Unsigned absolute weight bytes and
masked wrapped activation negation preserve PSIGNB behavior, including literal
-128. No saturated partial sums, new quantization, or checkpoint format are
introduced. Output uses the actual GGML column-major layout.

The fixture retains external GGML operand storage across every graph execution.
Two input updates, raw F32 source bytes, Q8 weight/input bytes, graph operand
bytes, reconstructed panel codes/scales and finite raw output float bits are
checked. Both the pinned vector dot and ordinary GGML graph must match before
timing. K32/96 tails, partial weight/input tiles, K1024/3072, 64-row matrices,
and K1024/N1024/M1500 are covered at one and sixteen workers.

Timing has three controls: pinned dot loop, ordinary prequantized GGML graph,
and dense VNNI. Six rotating/reversed rounds discard the first, then report the
median of five rounds. VNNI panel allocation, serial construction, half-scale
setup, vector sign/reduction and output scatter are included on every call.
F32-to-Q8 activation conversion, audio, attention, loading and generation are
excluded. All variants have the same worker budget; VNNI and the standalone dot
loop have the same contiguous weight-row ownership. OpenMP is unpinned on a
shared host. These are operator observations, not quiet-host causal estimates.

Build Release with MT_BUILD_TESTS=ON, then run
`build/tests/moss_vnni_q8_batch` and
`build/tests/moss_vnni_q8_batch --benchmark`. The target is outside default
CTest. Missing AVX512 VNNI/BW/VL/F16C/OpenMP or incompatible pinned dot traits
return 77; a portable skip-path compile does not validate the vector body.

## Frozen attempts and native result

V1 local `b7a9f7ed0d08d4c8a32b32335f6100d39dc67cd9` equals public
`7b8be65463c07a7d3ad1a2d50ae614746143af85`, tree
`eaad3a502cdf914767a5486ca2ac8cc6eb043513`. The initial CMake target incorrectly
linked `moss`, omitting transitive GGML include directories. The native build
stopped with `ggml.h: No such file or directory`, before arithmetic or timing.
[`vnni-batch-build-failure-v1.json`](vnni-batch-build-failure-v1.json) preserves
the exact source/log/controller hashes and failure. The
[reverse link patch](vnni-batch-v1-reverse-link.patch) reconstructs the failed
configuration from the corrected source. The frozen attempt is retained.

V2 corrects only the link to the existing `moss-transcribe` library. Vector
source is unchanged. Compiled local
`15c64e07878bfc9a696b37f82c750eeb7c1e7ca1` equals public
`a745d7c099d8dbbf681852ba8068fe3a5bc37570`, tree
`924c7f6da947e4ce9bc85ead4a8bce2136d269da`. The separate frozen source is clean
and pinned to ggml eced84. Native GCC 13.3 vector code compiles on Xeon Gold
5416S. Fifteen native CTests pass in 22.30 seconds and 34 Python checks pass.

All **74 arithmetic** and **109 protected benchmark** records match pinned-dot
and ordinary-graph float bits and panel codes/scales. The benchmark contains
**28 normal timed controls and 81 arithmetic-only edge controls**. Zero edge
durations are not timings. All records and source/compiler/binary/library/log
hashes are retained in [`vnni-batch-native-v2.json`](vnni-batch-native-v2.json).

| Workers | K / N / input rows | Ordinary GGML µs | Dense VNNI µs | GGML / VNNI |
|---:|---|---:|---:|---:|
| 1 | 1024 / 128 / 16 | 40.581 | 33.010 | 1.2294× |
| 1 | 3072 / 128 / 16 | 120.023 | 96.725 | 1.2409× |
| 1 | 1024 / 1024 / 64 | 1301.409 | 1131.182 | 1.1505× |
| 8 | 1024 / 1024 / 64 | 187.250 | 191.460 | 0.9780× |
| 16 | 1024 / 1024 / 64 | 250.827 | 277.808 | 0.9029× |
| 16 | 1024 / 4096 / 64 | 973.514 | 844.678 | 1.1525× |
| 16 | 1024 / 1024 / 1500 | 2891.531 | 3660.378 | 0.7900× |

Sixteen of 28 normal controls have a higher ratio; twelve are slower. Every
actual-width control improves at one worker, but most multiworker actual-width
controls regress. The 16-worker encoder-shaped matrix is **26.6% slower** than
ordinary GGML. Small-shape graph dispatch gains and a 1.15x large projection
observation do not establish useful full-model latency. No model routing or
full-input pilot is added for this version.

The successful controller exits zero and restores automatic processing to 1;
the failed build also restores it. Authenticated API health and original
production identity are verified. One-second monitoring observes three
probe-active snapshots without concurrent MOSS, at load 4.68–5.09. Shorter
overlaps cannot be excluded and other host jobs remain active. The frozen
AMX/VNNI attempts were separate shared-host runs; their absolute times are not
a controlled direct comparison.

Report SHA256:
`ca21ffe231c35ed8d12662dd879158af92cf78de5cc1d4935d9a2dcbf0ccac6d`.
Own source/evidence is MIT and dependency licenses remain intact. No private
audio, full text, raw token IDs, model weights or credentials are published.
The immutable quiet opt48 baseline and 10x objective remain unchanged, active
and unachieved.

## What the baseline already does and next experiment

Source inspection of pinned `ggml-cpu.c` shows contiguous prequantized operands
are offered to `llamafile_sgemm`; converted F32 operands are offered after
conversion. Its `tinyBLAS_Q0_AVX` includes 4×4 blocked schedules, four-row
half-scale conversion, PSIGNB/updot and the original vector accumulator layout.
This source inspection is not a runtime route observation. Ordinary GGML is a
stronger control than an independently parallelized dot loop.

The next concrete experiment is to instrument panel construction separately,
then test single-team parallel packing for large input-row counts and retain
the exact arithmetic. Measure packing, execution and total together; any
allocation changes must keep padding initialized and preserve operand guards.
Do not repeat this unchanged timing trial. A successful version still needs
actual F32 conversion and graph callback controls, protected layer/cache checks,
complete output/token/EOS parity and matched full-input latency gates.
