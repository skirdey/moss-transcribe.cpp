# Exact F32-input VNNI graph integration

This standalone experiment runs quantization, packing and exact VNNI traversal
inside a public GGML custom operation. It reuses existing callback workers,
with no nested OpenMP team. All three large encoder-shaped matrices improve
against ordinary F32-input GGML, while small projections regress. No actual
MOSS model routing or production change is added.

## Frozen native evidence

Compiled local `e56c407ef1044fa7dfc86a63d89754547f04a9b6` equals public
`6d9234c79d71fb018f06a8eade887121c9300e34`, tree
`b22c11b3423067af20397d51595361e902e780fc`. A separate clean source pins ggml
`eced84c86f8b012c752c016f7fe789adea168e1e`. GCC13.3 on Xeon Gold5416S compiles
the actual AVX512 VNNI/F16C/OpenMP body.15 CTests pass in22.36s and34 Python
checks pass. All78 arithmetic and111 protected benchmark shape records are
exact, including30 normal timed and81 arithmetic-only edge controls. Both
invocations pass2,097,152 exhaustive half-conversion comparisons across sixteen
rounding/FTZ/DAZ modes, restoring original MXCSR.

Each shape compares pinned vector dot, ordinary prequantized Q8 GGML, ordinary
F32 GGML, weight-first custom graph and cache-first custom graph. Complete
float bits agree with ordinary F32 GGML, including two input updates. F32 source,
Q8 weights, prequantized control and external graph operands remain immutable.
Actual per-callback quantized bytes match the pinned quantizer; initialized
panels, tail columns and ABI padding match. Every actual worker visits once,
worker budgets match1/8/16 as requested, and both traversals assign identical
weight tiles. Outputs are poisoned before parity so missing writes fail.

Callback scratch is allocated by worker zero, published through an atomic
barrier, quantized by rows, packed by panels, then consumed using the same
2-weight-row/16-input-row tile and original eight FMA chains. Further barriers
publish completed rows/panels and wait for consumers before cleanup. The
context lives through serialized graph executions; concurrent execution of
the same graph/context is unsupported. No private GGML threadpool access is
needed. Repeated callbacks check actual worker count and completed cleanup.

Build Release with MT_BUILD_TESTS=ON. Run `build/tests/moss_vnni_q8_f32`, then
the same executable with `--benchmark`. This target is outside default CTest
and returns77 on unsupported ISA or incompatible pinned dot traits. Every case,
six retained timing samples, source/compiler/library/artifact/log hashes and
controller/watcher checksums are in
[`vnni-f32-native-v1.json`](vnni-f32-native-v1.json).

## Paired primary timing

Three F32 graph variants rotate/reverse over seven rounds, discarding the first.
Each median averages the middle two of six retained sorted samples. Candidate
total includes graph callback overhead, per-callback scratch allocation, F32
quantization, initialized packing, worker barriers, exact tile computation and
cleanup. Ordinary GGML uses its warmed backend work buffer. The Q8 graph and
pinned dot are arithmetic-only controls. Callback audit is disabled in timed
calls, then repeated after timing. Graph construction/arena allocation, model loading,
audio, attention and full generation are excluded. No prior-turn timing is used
as the paired denominator.

| Workers | K / N / M | Ordinary F32 GGML µs | Weight-first custom µs | Cache-first custom µs | GGML / cache | Weight-first / cache |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 1024 / 1024 / 64 | 1310.753 | 978.698 | 970.183 | 1.3510× | 1.0088× |
| 16 | 1024 / 128 / 16 | 19.152 | 30.944 | 30.315 | 0.6318× | 1.0207× |
| 16 | 3072 / 128 / 16 | 31.006 | 42.375 | 42.333 | 0.7324× | 1.0010× |
| 16 | 1024 / 1024 / 64 | 252.864 | 211.279 | 204.195 | 1.2383× | 1.0347× |
| 16 | 1024 / 4096 / 64 | 525.665 | 401.281 | 396.606 | 1.3254× | 1.0118× |
| 16 | 1024 / 1024 / 1500 | 3273.397 | 2564.786 | 2321.024 | 1.4103× | 1.1050× |
| 16 | 1024 / 4096 / 1500 | 11606.915 | 9595.423 | 8608.396 | 1.3483× | 1.1147× |
| 16 | 4096 / 1024 / 1500 | 11350.751 | 13807.458 | 8813.771 | 1.2878× | 1.5666× |

Cache-first is faster than ordinary F32 GGML in15/30 cases and slower in15.
It improves26/30 versus weight-first custom traversal; four are slower. All
three M1500 encoder-shaped matrices beat ordinary F32 GGML by1.2878–1.4103×.
For each of these three, the largest cache sample is below the smallest
ordinary sample. The two small16-row projections at16 workers are36.5–58.3%
slower than ordinary GGML. These mixed results support selective large encoder
routing for the next experiment, not universal replacement. No complete-model
gain or causally isolated scheduling speedup is established.

The controller exits0, restores automatic processing to1 and verifies
authenticated API health plus unchanged production identity. The owned watcher
is stopped afterward. Eleven one-second probe-active snapshots observe zero
concurrent MOSS processes, at one-minute shared load8.20–8.82. Shorter overlaps
cannot be excluded; unrelated jobs remain active. Source/evidence is MIT with
dependency licenses retained; private audio, text, raw tokens, weights and
credentials are absent.

## Next complete-model gate

Port the validated callback into a separate opt-in encoder helper with explicit
per-encode ownership of every callback context. Keep ordinary GGML as fallback
for incompatible backend, ISA, type, dimensions and layout. Scope the first
route to large dense Q8 encoder linear operations; preserve biases, attention
and reductions. Compare actual encoder output bits on real weights, then every
complete output/token/EOS on matched full inputs before measuring full-input
latency including model loading. Any route must identify itself in retained
metadata. The quiet opt48 full-input10× goal remains active and unachieved.
