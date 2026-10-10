# Sparse AMX Q8 arithmetic probe: exact, slower

This standalone owned-vector probe preserves the pinned x86 Q8 dot's eight
four-code integer groups, block-order FMA chains and final horizontal-add tree.
It loads ordinary row-major Q8 weight codes directly. Sparse activation columns
produce eight independent integer sums rather than a single 32-code sum. No
packed checkpoint weight copy is created. There is no MOSS model route or flag.

Direct signed AMX products differ from the pinned VNNI PSIGNB path when an
activation code is -128 and its weight is negative: PSIGNB wraps negation of
-128. The probe corrects those integer partials before float conversion. It
preserves the original half-scale products and float accumulation order.
This is exactness against the pinned Q8 CPU routine, not the original F32 model.

## Frozen supported-hardware evidence

Compiled source `e149c0e72d063745f15a42d875fdd64421aeedac` is exactly public
`38075936cfccfba2ea19df0233084d3de4d310c3`, tree
`6d7b34ac84078ce68025fc8c55764615cc8126cd`. The
[complete provenance and numeric report](amx-lanes-native-v1.json) records source,
compiler, native binary/library/test/controller/log hashes and all measurements.
GCC compiled and executed the native AMX body on hp-fury's Xeon Gold 5416S.
Fifteen native CTests passed in 22.43 seconds; 34 Python gate/parser checks passed.

All 96 native arithmetic cases and 144 protected timing cases have zero output
float-bit differences. The grids cover 1/16 workers (timing adds 8), K32/96/1024/
3072, N17/1024/3072 and normal/zero/extreme/half-scale-edge distributions. Literal
codes -128/-127/0/127 and finite half-scale extremes include signed zeros,
subnormals, smallest normals and maximum positive/negative finite scales.
Every case uses two input updates. Raw Q8 activation and weight bytes are
checked for immutability before and after timing; outputs must remain finite.
F32 converter source uses a separate value-equality guard. Operand snapshots
are validation overhead, distinct from a packed inference checkpoint copy.

The executable returns 77 when the Linux/native AMX plus pinned AVX512 VNNI,
F16C, OpenMP and Q8 trait requirements are absent. It requests Linux tile-state
permission for its own process. It does not time a portable fallback.
The macOS Clang syntax check exercised only the guarded skip path; the GCC
supported-hardware result is the evidence for the actual arithmetic body.

## Protected warm timing

Six alternating-order rounds discard the first and take the median of five.
Normal distributions alone are timed: five calls per round at K>=1024, twenty
for smaller widths. Candidate timings include activation panel allocation and
construction, AMX configuration/release and the -128 checks/corrections. Input
quantization, model loading, audio, attention and full generation are excluded.
These are unpinned OpenMP worker counts on a shared loaded host. Both candidates
use the same owned operands and worker budget. The standalone probe shares its
panels across workers; future graph-private panels require separate measurement.

Representative actual-width timings, microseconds per operation:

| Workers | K | N | Pinned SIMD | Sparse AMX | SIMD/AMX ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1024 | 1024 | 37.675 | 116.196 | 0.3242x |
| 1 | 1024 | 3072 | 129.402 | 330.675 | 0.3913x |
| 1 | 3072 | 3072 | 403.525 | 894.444 | 0.4511x |
| 8 | 1024 | 1024 | 11.446 | 31.569 | 0.3626x |
| 8 | 1024 | 3072 | 16.686 | 51.897 | 0.3215x |
| 8 | 3072 | 3072 | 43.944 | 123.611 | 0.3555x |
| 16 | 1024 | 1024 | 15.795 | 37.572 | 0.4204x |
| 16 | 1024 | 3072 | 14.644 | 41.051 | 0.3567x |
| 16 | 3072 | 3072 | 47.876 | 143.849 | 0.3328x |

At 16 workers these widths are about 2.4–3.0 times slower with AMX. Across all
36 normal shape/worker controls, 35 are slower (ratios 0.2311–1.0129). The sole
higher ratio is a tiny K96/N17, 16-worker case: 5.008 to 4.945 microseconds,
1.0129x. All cases remain in the report; this small difference is not evidence
of an actual-width or full-model improvement.

**Decision: reject model integration of this version.** Exact arithmetic alone
does not overcome its sparse tile utilization, gather/scale and configuration
work. That bottleneck explanation is an inference from the implementation and
measurements, not a hardware-counter attribution. No full-input trial is run
for this slower standalone candidate, and no production promotion is made.
The controller exited 0, restored automatic processing to 1 and verified the
original authenticated API binary/health. The quiet opt48 baseline and tenfold
objective remain unchanged, active and unachieved.

## Reproduction and next work

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMT_BUILD_TESTS=ON
cmake --build build --target moss_amx_q8_lanes -j 16
./build/tests/moss_amx_q8_lanes
./build/tests/moss_amx_q8_lanes --benchmark
```

Use the pinned GGML submodule and supported Linux hardware. A skip is not a
parity pass. Retain the original frozen result; do not repeat this unchanged
probe to select favorable measurements. Before another AMX version, isolate
panel/scale/gather costs or investigate a denser exact-order layout with a
specific change. Any actual-model route still needs protected layer/cache,
complete-token/output/EOS and production/frozen/same-build latency gates.

The [Intel AMX intrinsics documentation](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)
supports the tile API used here; the
[Linux XSTATE documentation](https://www.kernel.org/doc/html/next/arch/x86/xstate.html)
specifies process-local tile permission. Neither establishes numerical parity
or performance for MOSS; the owned-vector test and complete-model gates supply
separate evidence. Own source, benchmark and report remain MIT licensed.
