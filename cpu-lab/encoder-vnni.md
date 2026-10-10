# Selective real-weight Whisper encoder VNNI route

An explicit research flag routes the six dense encoder linear operations per
layer through the exact F32-input VNNI callback. Full-input speech medians
observe 1.0220–1.0318× against the same-build opt 48 control. All complete output,
token, EOS, route, artifact and latency gates pass. Production remains on its
validated opt0 binary; this is modest shared-host evidence, not the 10× goal.

## Scope and ownership

Research bit 65536 combines with opt 48 as 65584. The factory requires CPU,
16 actual workers, contiguous F32 input and Q8_0 weights, and K/N 1024/1024,
1024/4096 or 4096/1024 at M1500. It also requires the pinned AVX512 VNNI/BW/VL/
F16C compilation and compatible Q8 traits. Null/type/shape/layout/ISA/thread
mismatches return null so the existing linear function is used. Bias additions,
attention, convolution, norms, activations, adaptor and decoder remain unchanged.

Each encode owns distinct heap contexts for its custom nodes until graph
execution and output reads finish. Existing GGML workers quantize complete
rows, pack initialized panels, run input-panel-first tiles, synchronize, and
release scratch. Four-code integer sums, the eight floating FMA chains and
final add tree match the pinned implementation. Worker zero publishes scratch
through the reusable atomic barrier; cleanup follows the final consumer barrier.
Allocation failure or an unexpected callback worker count produces invalid
outputs and marks the encoder failed. The existing serialized backend lifetime
still applies. Profile counters are collected after the join on the caller.

## Frozen correctness and provenance

Local `61f6836a51c6f5bea05fc7b45d5208f6dce56fff` equals public
`2a2b1cca0ad1ce10ecdecf1234c8825eb3d23c0f`, tree
`f9735254f35ec4df89be8847988d4e20548adab2`; ggml
`eced84c86f8b012c752c016f7fe789adea168e1e`. A separate clean root builds the
actual native body with GCC 13.3 on hp-fury. 16 native non-model CTests pass in
24.57s, including 12 protected synthetic shape/distribution records, two input
updates, complete float-bit comparisons, immutable external operands and
scratch cleanup. Null/type/shape/layout/8-worker fallbacks pass. 36 Python
regression/audit tests reject missing executions, wrong workers and bad phases.
Apple Clang also syntax-checks the unsupported-ISA branch and its callers;
local runtime testing was unavailable because CMake is absent.

The real-weight encoder gate checks every chunk from six audio inputs. All 14
chunks and 21,504,000 output floats match raw bits. Each chunk executes all 144
nodes at 16 workers, with zero failures and stable mel input. Encoder-only,
same-loaded-model observations are 1.0547–1.1512×, median 1.0668×. They always
run the reference first and candidate second, so they are not counterbalanced
performance evidence. Their purpose is intermediate-state correctness.

Source, compiler, flags, test/binary/library/log hashes, all synthetic and real
encoder records, phase counters and the complete 48-run report are retained in
[`encoder-vnni-native-v1.json`](encoder-vnni-native-v1.json), SHA256
`c5c9a1249547fd3ec82c96728aee65c6be31436ed16f7381b51bd380a0435922`.

## Full-input gate

Six cases, four variants and two alternating-order rounds use fresh processes,
16 physical-core workers and discarded full-inference warmups. Full wall time
includes model load. Controls are production opt 0, the frozen opt 48 build,
same-build opt 48, and the selective 65584 candidate. Across all 48 runs, per-case output hashes,
token counts, completion/EOS and input identities match; all 24 same-build
complete generated-token hashes agree, including unique terminal EOS. All 12 candidate
runs exercise 288 encoder nodes for 60 s and 576 for 120 s, exactly once at 16 workers.
Artifact integrity, route and all three fixed five-percent median-regression
gates pass. Silence retains its 13-token EOS result; real-empty retains its one-token EOS
and empty return code. No token-limit shortcut
or partial transcript is accepted.

| Input | Same-build48 s | Candidate65584 s | Same-build / candidate | Frozen48 / candidate |
|---|---:|---:|---:|---:|
| meeting-60s | 13.8231 | 13.4405 | 1.0285× | 1.0261× |
| dinner-60s | 11.6948 | 11.3626 | 1.0292× | 1.0271× |
| vox-vmaiq-60s | 12.0380 | 11.6672 | 1.0318× | 1.0318× |
| meeting-120s | 30.1480 | 29.4980 | 1.0220× | 1.0238× |
| silence-60s | 7.5291 | 7.1541 | 1.0524× | 1.0526× |
| empty-real-60s | 7.4082 | 7.0041 | 1.0577× | 1.0574× |

These are two-sample shared-host medians, not statistical confidence or a
quiet-host causal claim. One-minute load spans 12.96–16.19 in the full-input
runs; no additional MOSS process is sampled by the 0.5 s harness monitor.
During the separate encoder gate, 59 one-second active snapshots see no MOSS
process, at load 6.89–12.78. Shorter overlaps cannot be excluded; other jobs were
left running. The controller exits0, restores automatic processing to 1, and
verifies authenticated API health plus unchanged production SHA. Its owned
watcher is stopped afterward. Private audio, full text, raw token IDs, model
weights and credentials are excluded. Repository MIT/dependency licenses apply.

Build with MT_BUILD_TESTS=ON, run CTest excluding model fixtures, then invoke
`build/tests/moss_encoder_vnni_gate MODEL AUDIO...` with CPU and 16-thread/profile
settings. Run `moss_cpu_regression.py` with the production and frozen references,
`--candidate-reference 48 --variants baseline cache-reference 48 65584`,
`--repeats 2 --pin-physical --profile --trace-tokens` and all six cases in the
retained report. Never reuse a result directory or omit the same-build control.

## Remaining target

The immutable quiet opt 48 full-input targets remain about 1.16–1.39s for 60 s
speech and 3.1526 s for 120 s. The current candidate takes 11.36–13.44s and 29.498 s
respectively. The tenfold goal remains active and unachieved.

A possible next fusion shares one F32 conversion/panel pack across Q/K/V,
writes three contiguous [t,d] slices in one operation, and uses zero-copy views
for the existing consumers. That proposal is unimplemented. It needs a new
frozen source, exact intermediate and complete-token/EOS gates, actual execution
metadata and matched full-input latency. A small encoder-only improvement cannot
establish the original whole-pipeline goal.
