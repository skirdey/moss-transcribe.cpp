# Exact multi-row AMX Q8 experiment

This standalone experiment tests matrix workloads after the slower
[single-input sparse probe](amx-lanes.md). It adds no MOSS route, optimization
flag, model format, or production change. Native arithmetic passes, but the
measured version is rejected for model integration: all actual-width normal
controls are slower than ordinary GGML.

## Why matrix batches

[`amx-batch-phases-v1.json`](amx-batch-phases-v1.json) summarizes twelve existing
same-build opt48 samples from the paired fusion pilot, with two samples per
case. [`moss_phase_audit.py`](moss_phase_audit.py) reproduces this summary without
running inference. Its source report SHA256 is
`917b76d1659f5717b5aa7d4b37de9ec6a76eb13952293a738c52e5ca39a380b7`.

| Input | Wall including load | Encoder | Prefill | Decode | LM head |
|---|---:|---:|---:|---:|---:|
| English 60 s | 17.7541 s | 6.0821 s | 1.9899 s | 6.4809 s | 1.0347 s |
| German 60 s | 15.1417 s | 6.0680 s | 1.9510 s | 4.2973 s | 0.7046 s |
| Public speech 60 s | 15.7348 s | 6.1155 s | 1.9849 s | 4.7086 s | 0.7635 s |
| English 120 s | 32.9562 s | 10.5996 s | 4.5765 s | 13.8691 s | 1.8106 s |
| Real empty speech 60 s | 8.5372 s | 5.4400 s | 1.7324 s | 0 s | 0.0016 s |

These are medians on a shared host, not the immutable quiet baseline. Phase
subtimers can be nested and medians are not additive. The encoder and prefill
remain substantial work even if decode dispatch were eliminated. The original
quiet opt48 wall times and 10x targets remain unchanged and unachieved.

## Layout and arithmetic

[`moss_amx_q8_batch.cpp`](moss_amx_q8_batch.cpp) uses two original weight rows
against sixteen input rows per tile. It packs only activations into dense
four-code groups. Weight codes and half-scale bytes stay in ordinary Q8 storage;
there is no duplicated or compressed checkpoint. Eight separate four-code
integer sums retain the pinned floating FMA chains and horizontal-add tree.
Literal activation -128 receives the pinned PSIGNB wrapped-negation correction.
The output is scattered into GGML's actual column-major matrix layout.

The candidate timing includes panel allocation/construction, half-scale/mask
setup, tile configuration/release and output scatter. Q8 input quantization,
model loading, attention, audio and generation are excluded. The ordinary GGML
graph control also consumes prequantized Q8 activations. This is an operator
comparison, not a prediction of full-input latency.

The fixture owns every operand. External GGML buffers retain their storage
across all graph executions. Two input updates and normal, zero, signed-code
extreme and finite half-scale distributions are checked. Raw F32 source, Q8
weight and activation bytes remain immutable; packed activation codes/scales
are reconstructed and compared. Both the pinned vector dot and an ordinary
GGML graph must match raw output float bits before any timing is accepted.

The shape set includes K32/96 tails, partial weight/input tiles, K1024/3072,
64-row matrices, and the K1024/N1024/M1500 encoder shape. Native arithmetic uses
1/16 workers. Timing uses 1/8/16 workers, with six rotating/reversed variant
rounds, first round discarded, then median of five rounds. Variants have the
same worker budget; the standalone dot and AMX loops use the same contiguous
weight-row ownership. OpenMP timings on this shared host are unpinned.

Build with Release and MT_BUILD_TESTS=ON, then run
`build/tests/moss_amx_q8_batch` and
`build/tests/moss_amx_q8_batch --benchmark`. Unsupported ISA, Linux tile
permission or pinned dot traits return 77. Portable skip-path compilation is
not AMX validation. The target stays outside default CTest.

## Frozen native evidence and decision

[`amx-batch-native-v1.json`](amx-batch-native-v1.json) retains every numeric
record, source/compiler/binary/library/log hashes and limitations. Compiled
local commit `197a6a05d30587d4f9b7d4e02db5626061fbf17c` is exactly public source
commit `77db1c003f4ac0df8b85ba5e546723a74f3babf1`, tree
`c5836a38d780abadc8ae6f0b5759e737e1c11e53`. The separate frozen source is clean;
later documentation/evidence do not change its runtime or test source. The
native GCC 13.3 AMX body compiled on Xeon Gold 5416S with pinned ggml eced84.

Fifteen native CTests pass in 22.63 seconds and 34 Python checks pass. All 74
arithmetic records and 109 protected benchmark records have zero output-float,
ordinary-graph and activation-panel byte differences. The 109 benchmark records
contain **28 normal timed controls and 81 arithmetic-only edge controls**;
zero edge durations are not performance measurements. Timing excludes the
first round and includes candidate packing on every call.

| Workers | K / N / input rows | Dot loop µs | Ordinary GGML µs | AMX µs | GGML / AMX |
|---:|---|---:|---:|---:|---:|
| 1 | 1024 / 1024 / 64 | 2172.170 | 1302.990 | 4880.142 | 0.2670× |
| 8 | 1024 / 1024 / 64 | 304.839 | 195.352 | 779.099 | 0.2507× |
| 16 | 1024 / 1024 / 64 | 213.047 | 139.599 | 573.520 | 0.2434× |
| 16 | 3072 / 128 / 16 | 24.327 | 24.937 | 100.791 | 0.2474× |
| 16 | 1024 / 4096 / 64 | 828.778 | 513.481 | 1997.706 | 0.2570× |
| 16 | 1024 / 1024 / 1500 | 5136.015 | 2896.133 | 12955.911 | 0.2235× |

Twenty-one of 28 normal timing controls are slower than ordinary GGML. The seven
higher ratios occur only at small K32/96 shapes with graph worker-dispatch
overhead; the corresponding AMX times still exceed the standalone dot loop.
Dense input panels and avoiding the earlier sparse gather did not make this
four-code tile schedule faster at actual widths. The encoder-shaped matrix is
about **4.47× slower** at 16 workers. Retain the failure and do not add a MOSS route
or full-input pilot for this unchanged version.

The private controller exits zero and restores automatic processing to 1.
Authenticated API health and production binary identity are verified. A
one-second process watcher observes three probe-active snapshots without
concurrent MOSS and load 8.77–9.45. This sampling cannot exclude shorter
overlaps; other host jobs remain active. These are shared-host observations,
not quiet-host causal estimates. No full-model latency was measured here.

Report SHA256:
`28aa0f440ab26479f715c2ca294ffa1744d192d2b10aac6bd1b89bec6254abe5`.
No private audio, transcript, raw token IDs, model weights or credentials are
published. New source and evidence are MIT; existing dependency licenses stay
in place. The 10x goal remains active and unachieved.

## Primary research follow-up

[HiNa-MoE, October 4, 2026](https://arxiv.org/html/2610.05123) retains general
weight layouts and transforms intermediates to feed AMX. Its microkernel uses
BF16 on MoE workloads; its results do not establish exact Q8 arithmetic on this
dense model. The new four-code INT8 probe tests the activation-layout principle
while retaining the pinned reduction order. It does not copy the BF16 kernel
or apply its multi-socket speed ratios to this single-NUMA host.

[ZipServ, March 18, 2026](https://arxiv.org/html/2603.17435) reconstructs compressed
BF16 exponent data into GPU tensor-core fragments. That hardware-aware layout
principle is relevant to avoiding whole-layer expansion. Its BF16 exponent
redundancy does not imply useful compression for the Q8 codes occupying 97.83%
of this model's tensor bytes; the actual checkpoint audit already measured
7.6408-bit code entropy. It provides no exact CPU/Q8 or 10x evidence here.

Any successful operator still needs F32 conversion and actual graph integration,
protected layer/cache checks, complete output/token/EOS parity, and all matched
full-input latency controls before a production decision.

The next concrete alternative is dense four-code AVX-512/VNNI batch execution:
keep the same owned operands, panel and eight floating chains, but compute sums
in vector registers to avoid repeated AMX zero/store/reload operations. That is
an unimplemented hypothesis, not a speed claim. Preserve this frozen negative
attempt and require both ordinary-graph and pinned-dot parity before measuring
any new version.
