# Single-team VNNI panel packing

This standalone experiment retains the serial dense VNNI algorithm as a paired
control and adds worker-owned panel initialization and packing in the same
OpenMP team as matrix execution. No model route or production change is added.
The exactness gate passes, but all three encoder-shaped matrices remain slower
than the ordinary prequantized GGML graph.

## Source and arithmetic

[`moss_vnni_q8_parallel.cpp`](moss_vnni_q8_parallel.cpp) retains original Q8
weights, sixteen-column panels, two weight rows, wrapped -128 sign behavior,
eight separate four-code FMA chains and the pinned horizontal-add tree. The
candidate allocates panels without a serial zero-fill; each worker initializes
all panel bytes, including inactive columns and ABI padding, then writes codes
and half scales. The packing loop's implicit barrier precedes computation.
Worker count and static weight-row ownership match the serial control.

The fixture checks two input updates, original weight/input/source/graph operand
bytes, reconstructed panel codes/scales, all candidate tail columns/padding and
finite raw output float bits against both the pinned dot and ordinary graph.
K32/96 tails and extreme codes/scales are covered at one and sixteen workers.
Encoder-shaped K/N=1024/1024,1024/4096,4096/1024 with M1500 are also checked at
both budgets. Those three shapes use normal synthetic inputs only; they are not
complete audio/model tests.

Frozen compiled local commit `5c47bd452fd1669fafad6465df43c77996a811f5` equals
public `803f905c05f66397443d2f81a9d82634f3be84f0`, tree
`a625d033e2bb0d4676ee8ddbf9f251184528d4ea`. The separate clean native source is
pinned to ggml eced84. GCC 13.3 compiles the actual AVX512 VNNI/F16C/OpenMP body.
Fifteen CTests pass in 22.41 seconds and 34 Python checks pass. All **78 arithmetic**
and **111 protected benchmark** records pass both output controls and panel
checks with zero differences. The benchmark includes **30 normal timed** and
**81 arithmetic-only edge** records. Edge zero durations are not timings.
All records/source/compiler/artifact/library/log hashes appear in
[`vnni-parallel-native-v1.json`](vnni-parallel-native-v1.json).

Build Release with MT_BUILD_TESTS=ON, then run
`build/tests/moss_vnni_q8_parallel` and
`build/tests/moss_vnni_q8_parallel --benchmark`. The target is outside default
CTest and returns 77 on unsupported ISA or incompatible pinned dot traits.

## Paired total timings and separate stage diagnostics

Four variants rotate/reverse order over six rounds: pinned dot loop, ordinary
prequantized GGML graph, serial VNNI, and single-team VNNI. The first round is
discarded and five-round medians retained. Every candidate call includes panel
allocation, initialization, packing, scale setup, computation, scatter and
cleanup. F32-to-Q8 conversion, audio, attention, model loading and full generation
are excluded. OpenMP is unpinned on a shared host.

| Workers | K / N / M | Ordinary GGML µs | Serial VNNI µs | Parallel VNNI µs | GGML / parallel |
|---:|---|---:|---:|---:|---:|
| 16 | 1024 / 128 / 16 | 19.201 | 20.445 | 18.436 | 1.0415× |
| 16 | 3072 / 128 / 16 | 24.146 | 40.382 | 28.558 | 0.8455× |
| 16 | 1024 / 1024 / 64 | 139.546 | 162.963 | 164.006 | 0.8509× |
| 16 | 1024 / 4096 / 64 | 671.743 | 659.780 | 646.731 | 1.0387× |
| 16 | 1024 / 1024 / 1500 | 2930.198 | 3627.954 | 3362.843 | 0.8713× |
| 16 | 1024 / 4096 / 1500 | 11496.809 | 13862.718 | 13464.625 | 0.8539× |
| 16 | 4096 / 1024 / 1500 | 11192.415 | 15143.456 | 13969.539 | 0.8012× |

Parallel packing improves 12 of 30 normal controls versus serial; sixteen have
a higher ratio than ordinary GGML and fourteen are slower. The three encoder
controls improve 7.9%, 3.0%, and 8.4% in speed ratio versus serial but remain
14.8%, 17.1%, and 24.8% slower than ordinary GGML. No full-input pilot or
promotion follows from this version.

Separate six-round diagnostic passes measure allocation, packing, execution and
cleanup. Parallel diagnostics add a single timestamp barrier absent from primary
total timing; their packing stage includes team creation and execution includes
join. Stage medians are not additive and do not describe the same sample as total
medians. For K/N1024 M1500, serial allocation/packing medians are 73.529/231.230 µs;
parallel allocation/packing-and-team medians are 0.387/22.014 µs. Execution remains
the larger cost (serial diagnostic 3238.089 µs; parallel 3592.341 µs). This supports
investigating the compute loop, not a causal claim from summed medians.

The controller exits zero, restores automatic processing to 1, and verifies the
authenticated API and unchanged production identity. Nine one-second probe-active
snapshots observe no concurrent MOSS, at shared-host load 1.79–1.86. Shorter
overlaps cannot be excluded; other jobs remain active. The owned watcher is
terminated after the controller completes. Own source/evidence is MIT; dependency
licenses remain intact. No private audio, text, tokens, weights or credentials
are published. The original quiet opt48 full-input 10× goal remains active and
unachieved.

## Compute-loop evidence and next experiment

Disassembly of this exact binary's `tile<2>` shows two calls to
`ggml_fp16_to_fp32@plt` inside the Q8 block loop and accumulator stack stores
around them, with a 2368-byte stack frame. This is direct compiler evidence;
it does not prove the calls' share of latency. The next version will test an
inline F16C conversion only after exhaustive half-code parity against the pinned
converter, and retain the current serial/parallel controls. Do not alter or
rerun this frozen attempt merely for observation.

A [September 14 INT8 portability study](https://arxiv.org/html/2609.16085) holds
model/scales fixed and compares per-input predictions across kernels. Its
vision-model prediction checks and observed cross-kernel differences reinforce
checking individual outputs. They do not prove exact GGML Q8 arithmetic,
lossless checkpoint compression or a 10× MOSS gain. Any successful operator
still needs actual F32 conversion/graph callback, layer/cache, complete token/EOS
and matched full-input latency gates.
