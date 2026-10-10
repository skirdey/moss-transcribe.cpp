# Tenfold CPU performance research

Started 2026-10-09. The 10x objective is **active and unachieved**. The primary
comparison is the fastest validated Q8 build (opt 48, binary
`bb8bd09737c1063657f4bb0dde0c114bc63cde1a528518950c19007295387825`),
with the original production binary measured separately. Paired full-input wall
time, including model initialization, is the primary latency metric. Batch
throughput and warm persistent-server latency will be reported separately.

The previous 63-run confirmation is the starting evidence, not a new result.
English 60 s took 13.8929 s, German 60 s 11.6387 s, and the three public speech
clips took 11.9394–12.8915 s. A 10x improvement means roughly 1.16–1.39 s for
those short speech inputs. The 120 s stress input took 31.5263 s in one run;
its target is 3.1526 s. Silence/empty-speech behavior remains a regression gate.

## Current model and bottlenecks

The pinned GGUF is 986,881,024 bytes. The read-only audit found 343 Q8_0 tensors
(959,668,224 bytes) and 341 F32 tensors (21,248,832 bytes). Q8 accounts for 97.83%
of tensor bytes. Qwen3 projection weights occupy about 468 MB, the tied
embedding/LM head 165 MB, and the audio encoder about 342 MB. Model weights and
private audio/transcripts are not included in this repository.

In the validated English 60 s run, median phase timing was: prefill 1.4612 s,
decoder 5.4506 s, LM head 0.8952 s, and embedding 0.0105 s. The remaining 6.0754 s
includes audio preparation/encoding, model loading and other work. This is a
residual, not a separately instrumented encoder measurement. Making only the
decoder phase infinitely fast would cap the overall gain at about 1.65x. Both
audio processing and generation therefore need substantial improvement.

`moss_lossless_audit.py` samples each tensor's beginning, middle and end. Its
weighted Q8 code entropy was 7.6408 bits, scale entropy 7.6008 bits per 32-weight
block, and independent-symbol estimate 7.8783 bits/weight versus the stored 8.5.
That estimate describes a specific coding model, not a universal compression
bound. Sampled zlib1 and xz1 size ratios were 0.9127 and 0.9197. Median warm
sample decode throughput was about 136 and 13 MB/s respectively on this run.
These are sample codec measurements, not full-model compression or inference
speed. Every codec sample passed an exact round trip.

## Papers reviewed and applicable ideas

Only primary papers, author technical reports and official documentation are
used below. Reported gains retain their original hardware and baseline context.

| Source | What it establishes | Implication for this CPU/Q8 workload |
|---|---|---|
| [DFloat11, NeurIPS 2025](https://arxiv.org/html/2504.11651v3) | Entropy-coded BF16 weights, approximately 30% smaller, exact reconstruction; online GPU decompression has a nonzero cost. Large gains compare with memory-constrained CPU offload. | BF16 exponent coding does not directly apply to most of this Q8 model. Its fused reconstruction principle is worth testing; its offload gains are not CPU speed predictions. |
| [Unweight, Cloudflare technical report, April 2026](https://research.cloudflare.com/papers/unweight-2026.pdf) | BF16 palette/exponent compression reconstructed immediately before Hopper matrix instructions, with per-projection autotuning and pipelining. The report is explicitly ongoing research. | Reconstruct into the consumer's packed tile layout rather than expand a whole layer into RAM. Hopper kernels cannot execute on this host. |
| [Approaching Shannon Bound, ISCA 2026](https://arxiv.org/html/2606.15789) | Tile-addressable ANS reconstruction integrated with GPU GEMM, across several numeric formats. Reported serving gains depend on model format and memory-enabled batch sizes. | Measure entropy on the actual checkpoint, then account for decode cost. The paper's large footprint reductions in skewed low-bit formats do not establish a 10x latency gain for this Q8 checkpoint. |
| [ForgeMegakernel, September 2026](https://arxiv.org/html/2609.12379) | Per-model persistent H100 decode with instruction streams, dependency counters, buffer pools and intermediate-state checks. Serving gains are smaller than isolated-kernel gains. | Transfer the scheduling and validation methodology to CPU workers. GPU implementation is not directly usable. Final-output checks alone do not adequately diagnose fused arithmetic errors. |
| [MPK / Mirage, December 2025](https://arxiv.org/html/2512.22219) | GPU task graphs, event dependencies and cross-task memory pipelining within a persistent kernel. | Explore CPU task scheduling and fewer global barriers; distinguish dispatch savings from memory-bandwidth savings. |
| [Lossless but Not Free, July 2026](https://arxiv.org/abs/2607.17283) | A consumer-hardware speculative-decoding study with both wins and slowdowns; batched target verification and draft cost determine benefit. | Test greedy-verified multi-token drafting only after proving batch target execution is faster and matches sequential states/tokens. Draft speed alone is insufficient. |
| [FairyFuse, April 2026](https://arxiv.org/html/2604.20913) | CPU ternary kernels fuse eight widely-linear GEMVs and reuse activation registers. Its 29.6x kernel gain compares with FP32; end-to-end speedup over llama.cpp Q4_K_M is 1.24x on a Xeon 8558P. | Requires a ternary-trained checkpoint and changes weights/architecture. It is not lossless repacking of this MOSS checkpoint; activation reuse and avoiding repeated parallel regions remain useful scheduling ideas. |
| [CPU–GPU MoE design, OSDI 2026](https://arxiv.org/html/2606.10493) | CPU row tiling, concurrent gate/up work, fine-grained dependency barriers and fused activation conversions. FP8 post-scaling reports numerical differences from its reference. | Scheduling and conversion reuse are relevant to dense Qwen blocks; its FP8/BF16 arithmetic and dual-socket/GPU speed figures are not exact Q8 CPU evidence. Preserve the reference reduction tree before fusing scale application. |
| [Intel AMX tuning guide](https://www.intel.com/content/www/us/en/developer/articles/technical/tuning-guide-for-ai-on-the-4th-generation.html) | Matrix acceleration for INT8/BF16 and ISA-aware framework dispatch. | The host exposes AMX/VNNI, but MOSS must allocate the appropriate packed weights to enter ggml's optimized path. Preserve existing Q8 codes/scales; no new activation precision change is assumed safe. |

Some papers use "lossless" for equal task accuracy or matching distributions.
That differs from exact weight bytes, exact arithmetic, and identical generated
tokens. Each experiment must state which property is actually checked.

## Experiments and next milestones

1. **Packed Q8 matrices: rejected candidate.** The ordinary loader allocates
   all weights in the default CPU buffer. The archived opt-bit-128 prototype
   adds packed copies of eligible projection matrices using ggml's AMX/VNNI
   buffer. The tied embedding stays ordinary for direct row lookup. Input Q8
   codes/scales are unchanged, but native probes show float-bit differences.
   Full-model output also changed on German and public speech. The candidate
   was removed from active loader code. Opt 176 (128 + validated 48) is only for
   reproducing the archived experiment; it is not a valid active option.

   A separate exact-order AVX-512 probe computes 16 output rows together with
   eight independent accumulator chains per row and the pinned dot's final
   horizontal-add tree. Packing reconstructs every original Q8 byte. It tests
   signed-byte extremes, zeros, normal random inputs, and a partial output tile.
   The graph-level version now also matches activation conversion and passes
   36 exactness cases. Its first model integration failed latency and is archived
   below; the reusable graph helper has no active model routing.
2. **CPU megakernel scheduling.** Profile operator work and barriers, then
   specialize the decode graph without changing dependencies. Try compatible
   Q/K/V and gate/up projection scheduling, stable scratch/workspace reuse,
   persistent worker teams, and dependency-based handoffs. The OSDI CPU study
   provides a concrete gate/up scheduling example; its FP8 post-scaling math
   is not numerically identical and will not be copied into this exact gate. Keep intermediate
   state checks and varied shapes; the previous fused-attention failure remains
   a warning against inferring full-model gains from warm microbenchmarks.
3. **Audio encoder and prefill.** Benchmark packed GEMM and independently
   instrument mel, encoder, adaptor and prefill. Investigate processing the
   independent audio chunks in a compatible batch with exact boundary handling.
   Encoder reduction is necessary for the end-to-end 10x latency objective.
4. **Verified multi-token generation.** Prototype a cheap structural/ngram draft
   or compatible smaller draft. The target must verify all accepted greedy
   tokens, with correct causal masks, positions, KV rollback and EOS handling.
   Token agreement, intermediate-state drift, acceptance rate and total time
   must all be measured. Do not count reduced target calls as measured speed.
5. **Lossless compressed tiles.** Investigate exact Q8 scale/code packing and
   SIMD-friendly tile-addressable codecs if actual entropy supports a useful
   bandwidth reduction. Compare compression/decompression traffic against plain
   packed reads. Exact round-trip bytes alone do not establish faster inference.
6. **Persistent API and batching.** Reuse loaded weights/workspaces and batch
   independent audio jobs where useful. Report cold single-input, warm
   single-input, and sustained throughput separately. Do not reinterpret batch
   throughput as the requested single-input latency gain.

## Acceptance protocol

Pin the existing model/data and fixed opt-48 reference. Alternate variant order,
keep physical-core affinity and thread counts matched, record host load, exclude
concurrent MOSS inference, and verify artifact hashes. The harness now accepts
`--reference-opt 48 --reference-root /home/stan/hw-moss-softmax-v1`.

Retain the existing full-output, word/timestamp/speaker, token/EOS and genuine
empty-speech gates. Score the human-reference subset and expand accuracy evidence
as optimizations become more invasive. Record float-bit differences, exact weight
reconstruction, memory, failures, acceptance rates and intermediate-state checks.
A small regression suite is not proof of universal numerical equivalence.

Keep the production API available and restore automatic processing after every
temporary pause. Publish source, reproducible reports and failed experiments
under MIT. Merge each PR once its stated checks pass; keep unvalidated research
off the default/production path. The goal remains active until the actual tenfold
end-to-end improvement is achieved and verified.


## Measured research results (2026-10-09)

The AMX/VNNI native probe (`packed-kernel-v1.jsonl`) used identical Q8 weight
bytes and finite outputs, but all 12 shapes changed float bits. At 16 threads,
its matrix probes were 1.46–4.20x faster, with maximum absolute error up to
0.000106812. This was sufficient reason to require the full-output pilot.

The paired pilot (`packed-pilot-v1.json`) completed 18 measured runs: three
60 s inputs, three variants, two alternating rounds, after three discarded
warmups. Artifact hashes passed and no concurrent MOSS process was detected.
German output changed from 458 to 435 tokens in both rounds; public speech
changed from 490 to 476. The English output hash matched. Every run reached EOS.
The opt-48 reference retained exact output parity with production on all cases.

| Input | Opt 48 median wall | Packed 176 median wall | Exact output | Decision |
|---|---:|---:|---|---|
| English 60 s | 41.137 s | 36.456 s | Yes, both repeats | Overall candidate rejected |
| German 60 s | 39.860 s | 29.972 s | No, both repeats | Reject |
| Public speech 60 s | 42.306 s | 31.700 s | No, both repeats | Reject |

Other users' 20 TTS CPU workers were active and recorded host load rose from
about 24 to 36. The production English samples varied from 26.64 to 107.48 s.
These highly contended timings are not a stable isolated speed estimate, and
changed token counts make the failing cases unsuitable for equivalent-work
speed claims. Peak packed-process RSS was 2,568,752 KiB versus opt 48's
1,701,948 KiB. The loader patch is archived; active loader source is unchanged.
Human DER was not rescored after exact output already rejected the candidate.

The new exact-order dot probe (`exact-kernel-v1.jsonl`) passed **54/54** cases
with **zero float-bit differences** and exact byte reconstruction. It covers
1/16 threads, K=32/1024/3072, N=17/1024/3072, and normal/zero/signed-byte-extreme
inputs. For larger normal-random matrices (K,N >= 1024), 16-thread speedups were
1.33–1.55x, and one-thread speedups 1.38–1.63x. Small/tail cases can be slower;
all measurements are retained. The partially filled N=17 tile has padding;
full tiles have exactly the ordinary Q8 storage size.

The fresh native CMake build passed all seven model-independent CTests,
including the exact-order probe. The 12 regression-gate unit tests and three
synthetic GGUF audit tests pass. The archived loader patch applies cleanly to
its pinned base. Production API health, enabled service and resumed automatic
processing were verified after the pilot.

These are warm direct dot measurements on prequantized input. They exclude
input quantization, graph dispatch and packing from the timed dot loop. Packing
cost is recorded separately; inference integration would retain extra copies
or replace storage, which must be measured. Shared-host contention also applies
to these microbenchmarks. They do not establish an end-to-end speedup or model
output equivalence. The graph and full-model follow-up below test those missing
costs and reject the first integration on latency.

### Exact Q8 graph integration: output passes, latency fails

`src/cpu_q8.*` implements immutable 16-row packing and an exact-order custom
graph dot. It uses the existing ggml workers and the pinned CPU activation
converter. The archived model integration shares conversion across Q/K/V and
gate/up; prefill, embedding lookup and unsupported shapes use ordinary weights.
CPU-only loader ownership keeps callback pointers valid. Ordinary and packed
copies coexist. Bit 256 selects Qwen projection copies, bit 512 the LM head;
the pilot used 816 (48 + 256 + 512). These model routes are removed from active
source. The reusable helper and synthetic graph test remain for further research.

Both 36-case graph runs passed exact activation bytes and raw output floats,
including shared/separate conversion, signed-byte extremes, zeros, tails,
K=32/1024/3072 and N up to 151936. The first run was unpinned with simultaneous
production MOSS and other users' TTS work; its erratic timings are retained in
`exact-graph-cast-v1.jsonl`. The pinned run waited for API idle with auto work
paused; other TTS jobs remained active. Its reported normal 16-thread graph
ratios were 1.18–1.43x, excluding packing and graph construction. **Those warm
graph ratios are withdrawn.** A full operand audit detected mutation of the
original graph's synthetic weight tensor after its first computation, before
any timing. Its activation input remained unchanged; the first, input-only
audit had missed the weight mutation. The new retention/post-timing guard and
negative audit are documented in [shared activation](shared-activation.md).
Initial graph parity, separately owned-vector dot timings and the fresh-process
full pilot below are distinct evidence. Real model weights are loader-owned.
Unsupported ISA builds skip with code 77.

The full pilot (`exact-decode-pilot-v1.json`) measured 18 fresh processes after
three discarded warmups: three 60-second cases, three variants and two rounds
with reversed order. Every output hash, token count and EOS stop matched;
artifact hashes passed and no concurrent MOSS process was detected.

| Input | Opt 48 median wall | Exact packed 816 median wall | Exact output | Latency gate |
|---|---:|---:|---|---|
| English 60 s | 17.970 s | 19.774 s | Yes, both rounds | Fail: 10.04% slower |
| German 60 s | 15.793 s | 17.279 s | Yes, both rounds | Fail: 9.41% slower |
| Public speech 60 s | 18.059 s | 18.162 s | Yes, both rounds | Pass: 0.58% slower |

Peak RSS was 2319904 KiB versus 1702252 KiB for opt 48, an increase of
603.18 MiB. Median model-load phase across all six speech runs was 1.781 s
versus 1.017 s. The host was still shared (recorded one-minute loads about
22.5–31.0), and two rounds do not establish a stable isolated speed estimate.
Some decoder/logit phase timings improved while full wall time regressed;
the full-input metric remains the gate. Packing, extra storage, graph building,
dispatch and conversion scheduling need further work. The exact contribution
of each cost has not been isolated.

The latency failure stops promotion and the expanded quality/long-input suite.
Human DER was not rescored for this rejected candidate; tested raw-text parity
is reported separately from corpus-level accuracy. The native candidate build
passed seven other model-independent CTests plus both graph runs; 12 harness
tests and three audit tests pass. After removing model routes, a separate fresh
native build passed all eight model-independent CTests. Apple Clang C++17
syntax checks passed, but
local CMake was unavailable, so no macOS runtime result is claimed. API health,
active service and automatic processing restored to 1 were verified after the
terminal pilot failure. The 10x objective remains active and unachieved.

Next experiments should separate node/worker scheduling from arithmetic, try
compatible projection fusion and exact batched dots, and instrument encoder,
mel and adaptor costs directly. Adding packed copies or accepting a warm dot
gain alone is insufficient.

## Direct frontend and encoder isolation

The [new phase report](mel-phases.md) records a 36-run, two-round full-model
pilot and actual mel/Whisper/adaptor costs. All exact parallel DFT outputs,
tokens and EOS matched; all 12 native frontend cases were float-bit identical.
Its 5–18x frontend samples did not establish an end-to-end gain: English and
German medians failed the 5% gate. The same-opt control itself was 20.63% slower
on English, so this loaded run does not isolate a causal regression cost.
FFT changed English/German text, and encoder-only AMX changed German/public
speech outputs. All three model routes are archived against `4f40141`; helper
transforms and opt-in numeric profiling remain. A fresh probes-only build
passed nine native CTests. The intermediate build's three complete output/EOS
smokes passed but are not a latency acceptance. No broader/long or human DER
suite was run after failure, and production is not promoted.

Control inference medians were 31.739–44.494 s under shared load. Mel occupied
about 1.9–2.9%, graph construction/allocation 0.65–0.94%; decode and Whisper
encoding remain the major measured phases. Backend compute includes worker
scheduling and barriers, not only arithmetic. New work should target those
compute paths, tune thread budgets and prototype compatible projection fusion
or exact multirow kernels. A single-row arithmetic proof is insufficient for
the encoder's multirow SGEMM path. The 10x goal remains active and unachieved.

## Reproduce the probes

Use a native Release build with `-DMT_BUILD_TESTS=ON`, as documented in the root
README. On AVX-512 VNNI/BW/F16C with OpenMP, `moss_exact_q8_bench` compares the
pinned library's Q8 dot against the exact-order candidate. Other hosts return 77.
The AMX probe also returns 77 if no packed buffer is available; on this host its
expected status is 1 because numerical differences reject it.

```bash
OMP_NUM_THREADS=16 OMP_PROC_BIND=spread \
  OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}' \
  ./build/tests/moss_exact_q8_bench > exact-kernel.jsonl
./build/tests/moss_packed_q8_bench > packed-kernel.jsonl
python3 cpu-lab/moss_lossless_audit.py /path/to/moss-transcribe-q8_0.gguf \
  --output lossless-audit.json
python3 -m unittest discover -s cpu-lab -p test_moss_lossless_audit.py
python3 -m unittest discover -s cpu-lab -p test_moss_cpu_regression.py
ctest --test-dir build -LE model --output-on-failure
```

The rejected loader patch applies to a fresh fork checkout at `2dbe59e`:

```bash
git checkout 2dbe59e
git apply /path/to/packed-q8-candidate-v1.patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
# Create a frozen root containing source/ and build/, then run the public harness:
python3 /path/to/moss_cpu_regression.py --root /path/to/frozen-root \
  --reference-root /path/to/validated-opt48-root --reference-opt 48 \
  --variants baseline cache-reference 176 --cases meeting-60s dinner-60s vox-vmaiq-60s \
  --repeats 2 --pin-physical --name packed-pilot
```

Supply baseline, model and audio paths using the harness options when reproducing
elsewhere. Private fixtures are represented by hashes only; the public VoxConverse
fixture can be prepared using the root README's pinned dataset instructions.
Never mix a changed source/binary into an existing run directory.

The exact graph test is registered as `test_cpu_q8` in native builds:

```bash
OMP_NUM_THREADS=16 OMP_PROC_BIND=spread \
  OMP_PLACES='{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}' \
  ./build/tests/test_cpu_q8 > exact-graph.jsonl
```

To reproduce the rejected integration, start from a **fresh** checkout at
`338aa8d` and apply `exact-decode-cast-v1.patch`, which includes the helper,
loader/decoder routes and graph test. Use opt 816 and the paired harness with
`--reference-opt 48 --variants baseline cache-reference 816 --repeats 2`.
The expected native pilot status is 1 (latency rejection), although shared-host
timings can vary. See `exact-graph-provenance-v1.json` for actual compiled source,
library, binary and graph-report hashes; its base commit alone does not describe
the dirty experimental tree. The failed build remains frozen for reproducibility.

## CPU worker budgets and actual draft replay, 2026-10-09

The [thread scheduling report](thread-scheduling.md) preserves the complete
30-run 4/8/16-thread pilot, the separate 18-run mixed-phase pilot, bounded
passive-wait failure, initial logger truncation and repaired private trace
audit. All 48 timed outputs and EOS matched the frozen reference; artifact
integrity passed. Global eight threads observed 1.21–1.36x speedups, while four
failed latency by 26.68–43.69%. The mixed candidate kept encoder/prefill at 16
and decode/logits at eight; it passed all gates at 1.10–1.14x speedup.

Both suites ran with other TTS jobs active and high shared load. Same-opt
controls varied materially; the mixed suite lacks a same-build all-16/global-8
control. These results cannot establish a quiet-host gain, prove mixed beats
global eight, or replace the original quiet opt48 reference for the 10x goal.
Production remains on its validated binary and policy. Automatic processing
was restored and authenticated API health was OK after each controller.

Bounded CPU phase scopes default to the startup thread budget and restore it
after nested scopes or exceptions. The fresh native build passed ten
model-independent CTests and 25 Python tests. Private tracing is off by
default; repaired chunks passed all three output/EOS smokes and full trace
parser checks. Public source mappings, exact replay-repair patch, test-log
hashes and numeric reports accompany the implementation.

Actual past-only eight-token ngram replay reduced target calls by only
1.23–1.34x while requiring 2.31–3.19x the sequential target-token work.
Four-token drafts saved almost as many calls for less rejected work. This is
an oracle acceptance/work estimate, with no batched target, numerical verifier,
KV rollback or latency implementation. It does not justify prioritizing this
ngram draft toward a 10x full-input improvement.

The tenfold objective remains active and unachieved. Next investigate exact
multirow encoder compute and loader copy costs; continue measuring against
the frozen fastest opt48 reference. Encoder, prefill and loading must improve
alongside decoding. Track persistent/batch throughput separately, preserve
full text/speaker/timestamp/EOS gates, and avoid rerunning rejected kernels
without a concrete implementation change.

## September/October 2026 matrix-engine research

[BF16 component-product emulation, September 4, 2026](https://arxiv.org/html/2609.04663)
splits each FP32 operand into three BF16 residual components and evaluates six
selected products. Prepacked panels and a tile-resident reuse schedule limit
conversion and intermediate traffic. It targets FP32-level error relative to
oneMKL SGEMM and explicitly does not promise bitwise identity. Its measurements
use a dual-socket Xeon 8462Y+ with 64 physical workers. This is a candidate for
an arithmetic probe, not evidence for MOSS speed or exactness on hp-fury.

[HiNa-MoE, October 4, 2026](https://arxiv.org/html/2610.05123) investigates CPU
matrix-engine tiling, fused layout transformation and MoE scheduling. Its
reported gains use BF16 MoE workloads. MOSS is dense, Q8 and single-NUMA here;
the reusable ideas are packing and scheduling, not its numerical format or
published speed ratios.

The pinned `ggml_compute_forward_mul_mat` in
`third_party/ggml/src/ggml-cpu/ggml-cpu.c` first tries llamafile SGEMM and otherwise
converts F32 activations to the weight type's dot format. Its Q8_0 SGEMM case
also rejects an F32 right-hand operand: both ordinary Q8 routes use Q8_0
activations. Replacing this with a plain FP32/BF16 product would change both
activation conversion and reduction. A new probe must identify the executed
multirow path, preserve its inputs, and report raw float differences before
full-output gating. Neither paper establishes that equivalence.

A separate next experiment is to share Q8 activation conversion across Q/K/V
and gate/up while retaining ordinary weight storage and the reference
`ggml_mul_mat`. The archived opt816 trial combined shared conversion with
custom packed weights and dots; it did not isolate conversion sharing. A
conversion-only probe can avoid its extra packing/storage cost. It must still
compare the actual raw Q8 activation bytes and float outputs at one and multiple
input rows: an already-Q8 operand reaches SGEMM at a different dispatch point.
Conversion, scheduler barriers and full loading/inference remain timed. This
is a proposed experiment, with no measured gain or integration yet.

## Completed CPU model-mapping trial, 2026-10-09

The [mapping report](model-mapping.md) records exact identity of all 684
checkpoint tensors (980,917,056 payload bytes) and all 40 full-input outputs,
tokens and EOS. Mapping passed artifact/storage gates, but failed latency:
120-second speech was 22.78% slower and silence 8.34% slower than copied
storage in the same binary. Silence also failed both older-reference gates;
the frozen runner returned 1. Peak long-input RSS remained about 1.99 GiB.
The copied/mapped loader diagnostics were 0.8593/0.0332 seconds in a shared
process with warm file cache; lazy page faults were paid during the subsequent
byte comparison. That ratio is not a full-input improvement.

The entire prototype and tests remain public as an exact patch and historic
tree. Ordinary inference restores the original loader and native configuration.
The final build passed ten native CTests and five full-output/EOS smokes;
the new same-build benchmark gate passed all 28 Python tests on hp-fury.
Its failure test rejects candidates that beat older binaries while losing
to the same-build control. Future paired trials should use
`--candidate-reference` in addition to the frozen fastest opt48 comparison.

Two repetitions under variable shared load establish rejection of this trial,
not a statistical confidence interval or a new quiet-host speedup. Production
remains on its validated binary; automatic processing and API health were
restored. The tenfold objective is active and unachieved. Next isolate shared
activation conversion with reference matrix math, without duplicated packed
weights, and prove single/multirow activation bytes and float results first.

## Shared activation conversion and operand lifetime audit, 2026-10-09

The [shared activation report](shared-activation.md) isolates conversion reuse
with ordinary weights and reference matrix arithmetic. All 72 synthetic cases
matched activation bytes and output float bits at 1/8/16 workers, including
actual Qwen projection widths, strided rows and a 1500-row encoder input.
Two input updates, immutable operands/artifacts and no concurrent MOSS passed.
The 16-thread Qwen decode group observed 1.109x. Custom shared conversion was
effectively neutral on the 1500-row encoder (1.0011x); shared cast observed
about 1.20x there. All controls/results are retained, including slower cases.

The new fixture initially failed because graph-allocated synthetic operands
were reclaimed before repeated execution. It now retains them and byte-checks
all weights/inputs after each update and timing. The older exact-Q8 graph
microbenchmark lacked the same guard. Its input-only audit missed corruption;
the complete audit detected **weight mutation in the original graph before
timing**, with the activation unchanged and no added temporary. Its historical
warm graph ratios are withdrawn. Initial graph output comparisons and the
separate owned-vector dot/fresh-process full-model pilots remain their distinct
evidence; actual model weights have external loader-owned storage.

The final guarded build passed twelve native CTests in 24.77 seconds and 28
Python checks on hp-fury. Frozen failed fixtures, exact repair patches, test-log
and source hashes, raw numeric reports and public equivalent trees are retained.
No actual model routing or production promotion was added. These warm graph
results exclude loading, audio, attention and generation; they establish
neither a full-input gain nor corpus accuracy or a tenfold result.

Next test shape-aware shared cast/block conversion in the actual Q/K/V and
gate/up paths, retaining ordinary weights and reference matrix operations.
Use matched production, frozen fastest opt48 and same-build opt48 controls,
including long speech/silence and full text/speaker/timestamp/token/EOS gates.
The original quiet reference and tenfold objective remain unchanged and active.

## Actual-model shared conversion, 2026-10-09

The [full report](shared-model.md) integrates shape-aware block/cast reuse
into actual Qwen and Whisper projections without weight copies or new matrix
arithmetic. Research bits8192/16384 are off by default. Native validation
passed all 13 CTests and 33 Python gate/parser tests on hp-fury; the actual
Qwen layer/cache fixture has zero float-bit differences across decode and
prefill at 1/16 workers. The failed CPY/DUP fixture attempt is retained with
an exact source reverse patch and build/test provenance.

The [72-run paired pilot](shared-model-pilot-v2.json) includes production,
frozen fastest opt48, same-build opt48, decoder reuse, encoder reuse and both.
All output hashes, counts and EOS match; all 48 numeric runs match complete
same-build token hashes. Artifact and route gates pass. The original runner
exited1: combined reuse failed German latency against frozen opt48 (+5.98%)
and same-build opt48 (+5.31%). Real-empty speech failed production for every
opt48 control/candidate (+15.87% to +20.20%). No candidate passes all controls
or is promoted. Shared host load spans 23.59-32.44, and unchanged phases
also vary; the apparent isolated encoder improvement cannot establish causality.

Production and automatic processing were restored and health verified. The
quiet opt48 baseline and tenfold objective are unchanged, active and unachieved.
The report adds three primary papers with explicit hardware/numerical limits.
Next test one exact-dot CPU operation per decode projection group with
ordinary weights and private per-worker conversion; preserve prefill fallback,
actual layer/cache bits, complete token sequences and all three latency gates.

## Exact-dot projection fusion, 2026-10-09

The [projection fusion report](fused-projection.md) combines eligible one-row
Qwen Q/K/V and gate/up operations while retaining ordinary Q8 weights and the
pinned converter/dot routines. Private per-worker activation conversion avoids
a custom barrier or shared mutable scratch. Research bit 32768 is off by default;
prefill and unsupported shapes fall back. Fifteen native CTests and 34 Python
checks pass. Protected projection and actual layer/cache tests preserve float
bits and operands across repeated input updates. At 16 workers, valid protected
graph probes observe 1.208x Q/K/V and 1.183x gate/up; slower 8-worker cases remain.

The [48-run full-input pilot](fused-projection-pilot-v1.json) passes complete
outputs, EOS, artifacts, observed routes, complete same-build token hashes and
all three latency controls. Speech medians are only 0.2–2.7% faster than same-build
opt48. Two repeats and shared host load 14.36–30.01 establish neither statistical
confidence nor quiet-host causality. Silence is 0.29% slower than frozen opt48,
within the fixed five-percent rejection gate. Production remains unchanged,
automatic processing was restored and API identity/health verified. The quiet
opt48 reference and tenfold objective remain unchanged, active and unachieved.

Next arithmetic probe: sparse AMX columns preserve eight separate four-code
integer sums, original float FMA chains and the final horizontal-add tree.
Require supported-hardware exact arithmetic and protected timing before any
actual-model integration. Full-model gates remain mandatory for a new route.

## Sparse AMX exact-order arithmetic, 2026-10-09

The [AMX probe report](amx-lanes.md) tests separate four-code integer sums,
original half-scale products and pinned FMA/horizontal-add order without a
packed checkpoint copy. The native GCC body runs on hp-fury. All 96 arithmetic
and 144 protected timing records preserve float bits and raw Q8 operands,
including literal -128 PSIGNB correction and finite half-scale extremes. Fifteen
native CTests and 34 Python checks pass; unsupported machines return 77 without
portable fallback timing. It is an owned-vector probe, not a model route.

Thirty-five of 36 normal shape/worker timing controls are slower. Representative
16-worker widths are 2.4–3.0x slower than pinned SIMD; the sole higher ratio is
K96/N17 at 16 workers, 5.008 to 4.945 microseconds (1.0129x). All measurements,
including this tiny case, source/test/binary/library/log hashes and limits are
retained. Panels/configuration are included; input quantization, loading and
full generation are excluded. Shared OpenMP timings are unpinned. Reject this
version for model integration; arithmetic exactness does not establish speed.

Production/automatic processing were restored and API identity/health verified.
The original quiet opt48 reference and tenfold objective remain unchanged,
active and unachieved. Next isolate layout/scale/gather overhead before proposing
a concrete new arithmetic version. Any actual-model integration requires all
protected layer/cache, complete output/token/EOS and matched latency gates.

## Dense AMX matrix batches, 2026-10-09

The [multi-row report](amx-batch.md) extends the exact four-code method to two
original weight rows and sixteen input rows using dense activation panels.
An ordinary prequantized GGML graph is an additional protected control; all
operands have stable owned storage and raw-byte immutability checks. Native
GCC AMX compilation, 15 CTests and 34 Python checks pass. All 74 arithmetic
and 109 protected benchmark records match output float bits and packed input
bytes. Of those 109 records, 28 normal controls are timed and 81 edge controls
are arithmetic-only. Packing/configuration/output scatter are timed; activation
quantization, audio, model loading and generation are excluded.

Twenty-one of 28 normal controls are slower than ordinary GGML. All actual-width
controls are slower; the seven higher ratios occur only in small K32/96 cases
with graph worker-dispatch overhead and remain slower than the standalone dot
loop. At 16 workers the K1024/N1024/M1500 encoder-shaped matrix takes 12.956 ms
versus 2.896 ms for ordinary GGML, about 4.47x slower. Reject this version for
model routing and retain the numeric evidence. No full-input gain is claimed.

A reproducible audit of twelve existing same-build opt48 phase samples finds
about 6.1 s in the encoder and 2.0 s in prefill for the 60 s speech cases. These
two-sample shared-host medians do not replace the quiet opt48 baseline. The
report also records a deeper HiNa-MoE activation-layout review and the BF16/GPU
compression limits of ZipServ, with primary links. Neither paper establishes
exact Q8 or10x gains here. Automatic processing and API identity/health are
restored. The goal remains active/unachieved. Next test dense exact VNNI batches
in vector registers to remove tile zero/store/reload cost, then require actual
layer/cache/token/output/EOS and matched full-input gates for any viable route.

## Dense VNNI matrix batches, 2026-10-09

The [register-based batch probe](vnni-batch.md) preserves original Q8 weight
bytes, eight four-code floating chains and PSIGNB wrapped-negation behavior
without AMX tile zero/store/reload work. Protected pinned-dot and ordinary GGML
graph controls retain all operands. V1 had a wrong CMake link target and stopped
before arithmetic; its source, exact reverse patch and failure hashes are
retained. V2 fixes only that link. Native GCC vector code, 15 CTests and 34
Python checks pass. All 74 arithmetic and 109 protected benchmark records
match raw output float bits and dense activation panel codes/scales. The
benchmark has 28 normal timed and 81 arithmetic-only edge controls.

Sixteen of 28 normal controls are faster and twelve slower than ordinary GGML.
All actual-width controls improve at one worker, while most multiworker cases
regress. K1024/N4096/M64 observes 1.1525x at 16 workers, but the encoder-shaped
K1024/N1024/M1500 takes 3.660 ms versus 2.892 ms (26.6% slower). Candidate
allocation/serial packing/sign/reduction/scatter are included; input conversion
and full inference are excluded. No actual model route or full-input gain is
claimed. Automatic processing and authenticated API identity/health are restored.

Pinned GGML source already contains blocked Q8 SGEMM schedules. The next
experiment isolates dense panel construction and tries single-team parallel
packing at large row counts, preserving initialized padding and all byte/float
guards. Then test actual F32 graph conversion/callback and full-model gates for
any successful version. No unchanged trial replay. The quiet opt48 baseline
and 10x target remain active and unachieved.

## Single-team VNNI panel packing, 2026-10-09

The [parallel packing probe](vnni-parallel.md) keeps serial packing and ordinary
prequantized GGML as paired controls. Original Q8 codes, half scales, wrapped
negation and pinned eight-chain FMA order remain intact. Candidate workers fully
initialize panels/tails/padding, then compute after the packing barrier in one
team. Native15 CTests/34 Python checks pass; all78 arithmetic/111 protected
benchmark records are exact (30 normal timed,81 arithmetic-only edges). Twelve
of30 timed cases improve versus serial,16 versus ordinary GGML. Encoder-shaped
K/N1024/1024,1024/4096,4096/1024 M1500 improve3–8% in speed ratio versus serial
but remain14.8–24.8% slower than ordinary GGML. No actual-model route/promotion.

Separate stage diagnostics reduce allocation/packing overhead but execution
remains larger. Diagnostic barriers/timers are absent from paired primary totals
and stage medians are not additive. Exact-binary disassembly shows two
`ggml_fp16_to_fp32@plt` calls per Q8 block and accumulator stack spills around
them. Next test inline F16C only after exhaustive65536 half-bit-pattern parity,
then preserve all paired controls before F32 graph/layer/full-input gates. No
unchanged timing replay or causal spill-latency claim. All records and frozen
source/artifact/compiler/log hashes remain public; production identity unchanged
and automatic processing restored. The quiet opt48 target remains unachieved.

[Is INT8 Portable?, September14 2026](https://arxiv.org/html/2609.16085) compares
per-input predictions while holding quantized artifacts/scales fixed. Its
vision-model cross-kernel observations reinforce individual-output validation;
they do not prove exact GGML arithmetic, lossless Q8 compression or tenfold MOSS
latency. The next kernel must keep exact controls irrespective of average quality.

## Exact inline half conversion, 2026-10-09

The [inline F16C probe](vnni-inline.md) preserves the serial and original
parallel VNNI controls. Initial V1 native compilation/15 CTests/34 Python checks
pass, then exhaustive half auditing rejects four signed-zero differences under
downward rounding before matrix timing. Its exact public source/failure is
retained. V2 adds that precise half+0 compatibility case and reads MXCSR once per
tile. Every65536 halfpayload under16rounding/FTZ/DAZmodes matches (2097152
comparisons across both invocations), and originalMXCSRrestores. All78arithmetic
and111protected shape records are exact;15 CTests22.34s and34Python pass.

Twenty-three of30paired totals improve versus originalparallel;21 versusordinary
GGML. Encoder K/N1024/1024 M1500 observes1.0636× and1024/4096 observes1.2187×
overGGML, while4096/1024 remains19.6%slower. Inputquantization/model/audiofull
generation excluded. Exactbinaryinlineblockloop retainsaccumulatorsinregisters
with0convertercalls/0vectorstackaccesses; outside-loopstackstores andstackguard
remain. No causal latency attribution orfull-inputgainclaim. No model route.
Next test input-tile-first traversal within eachworker'ssameweightpartition,
retainingoldpairedschedule; then requireF32graph/layer/fulltoken/EOS/latencygates.
Allcases/source/artifact/loghashes retained. Production unchanged andautomatic1.

[Intel compiler guidance](https://www.intel.com/content/www/us/en/developer/articles/technical/whats-new-in-llvm-for-4th-gen-intel-xeon-processor.html)
illustrates F16C conversion lowering; it does not prove our converter parity.
[Ada-MK, May12 2026](https://arxiv.org/html/2605.11581) uses static offline
dependency/resource scheduling with phase-specific inference. Its GPTQ-W4A16
Qwen/L20GPU throughput tests are not exact Q8 CPU or tenfold latency evidence.
Transfer static scheduling and measured phase controls, preserving our arithmetic.

## Cache traversal and recent primary research, 2026-10-09

The standalone cache probe changes only traversal within each worker's weight
partition, preserving the previous tile arithmetic and panel construction. It
compares six variants, retains all six post-warmup samples, and checks identical
worker ownership as well as float bits, operands and panel bytes. Native results
are recorded in [the cache report](vnni-cache.md); model routing is a later gate.

[Space Filling Curves, April 7 2026 revision](https://arxiv.org/html/2601.16294v2)
uses Hilbert traversal of output tiles and optional K partitioning/replicated
outputs to reduce communication. Evaluation uses BF16 on several many-core CPUs;
the Llama-3-8B case measures prefill at batches8–64 and input lengths1024–4096.
Its up-to5.5× individual GEMM gain and up-to1.85× prefill gain are not our exact
Q8 CPU or full-input latency result. Inference: traversal without K splitting
can improve locality while preserving each output's accumulation order; partial
K reductions need a separate exactness proof. Our simple loop interchange does
not implement the paper's Hilbert algorithm.

[WACT, August 25 2026](https://www.mdpi.com/2079-9292/15/17/3811)
uses frozen per-tensor codec policies and strict RAW fallback after counting
metadata and alignment. Modes include bit-width packing, delta/Rice, significance
maps and zero runs. Five CNNs yield1.801× occupied-packet compression with exact
INT8 reconstruction; software evaluates the full policy, hardware implements
only single-tile Rice paths. This is compression-interface evidence, not measured
Xeon latency. Quantize/dequantize accuracy differs from original FP32 accuracy,
so codec losslessness does not mean lossless quantization. Transfer the complete
byte accounting, frozen policies and reinjection checks, preserving existing
MOSS Q8 codes and scales; measure decoder cost before claiming any latency win.

Cache native outcome: all189 shape records and2097152 half conversions are
exact, including identical worker ownership.15 CTests22.65s/34Python pass.
All threeM1500 matrices are1.30–1.35× faster thanordinary prequantizedGGML;
the prior contraction regression is removed (1.6092× versuspriorinline).
23/30cases beatGGML;21/30beatpriorinline. Retained six-sample distributions
include losses and outliers. The next F32 graph gate must include quantization
and callback overhead and compare against ordinary F32 GGML's selected route.
No actual-model route, full-input measurement or tenfold achievement follows.

## F32-input custom graph integration, 2026-10-09

The [F32 graph probe](vnni-f32.md) uses existing callback workers and atomic
barriers, preserving all tile arithmetic. Candidate totals include quantization,
scratch allocation, packing, coordination and cleanup. All78 arithmetic and
111 protected records,2097152 half conversions,15 CTests22.36s and34 Python
checks pass. Ordinary F32 GGML and Q8/pinned controls match exact output bits;
callback quantized bytes, initialized panels, actual workers and ownership match.
AllM1500 matrices beatF32GGML1.2878–1.4103×, while15/30timed cases regress.
Next port selective large encoder routing with per-encode context ownership,
real-weight encoder bit parity and complete-token/output/EOS/full-input gates.
No model routing or tenfold achievement is claimed.

[The Integer Alibi, August18 2026 revision](https://arxiv.org/html/2608.13756)
uses shared INT8 operands to isolate GPU kernel scale/rounding divergence.
Its exact INT32 argument requires no overflow and the actual instruction path;
it explicitly cautions about saturating intermediate CPU instructions. End-to-
end historical runs lack runtime kernel manifests, unlike teacher-forced replay.
Power-of-two scales are a diagnostic probe with no validated accuracy or
performance mitigation. Transfer fixed-operand comparisons and runtime route
evidence. Inference: our block-scaled Q8 floating accumulation still needs its
original FMA order; a single integer sum or changed scales would change semantics.

[FairyFuse, April22 2026](https://arxiv.org/html/2604.20913) fuses eight sub-GEMVs
of a ternary widely-linear layer with masked add/subtract instructions. Its
29.6× headline compares48-thread ternary against1-thread FP32; matched1-thread
DRAM-cold gains are2.0–6.6×, and model throughput gains are1.24× versusQ4_K_M.
The model uses ternary QAT and quality metrics differ from FP16, so this is not
lossless compression of MOSS Q8 or exact output preservation. Transfer input
reuse, register accumulation and worker-region amortization, retaining matched
thread, format, cold/warm and full-input denominators.

## Selective real-weight encoder integration, 2026-10-09

The [encoder report](encoder-vnni.md) routes only the three measured M1500 Q8
shapes at16 workers behind research bit65536. Per-encode contexts survive
compute/read and own the quantizer/panels/barriers; actual callback counters
are collected after the join. All16 native CTests and36 Python checks pass.
All12 synthetic records and14 real-weight chunks are bit exact, including
21,504,000 encoder output floats and2,016 actual node executions.

The complete48-run fresh-process pilot passes full outputs/count/EOS, all24
same-build token traces, immutable artifacts, all12 actual candidate routes
and all three fixed five-percent latency gates. Same-build speech speedups
are1.0220–1.0318×; silence/real-empty1.0524–1.0577×. Full-input includes load.
All records remain, and shared load12.96–16.19/two repeats do not establish
confidence or replace the immutable quiet baseline. Encoder-only reference-
first observations (median1.0668×) are correctness-first, not counterbalanced.
Controller0/restoredauto1/authAPIunchanged; watcher stopped. No production
promotion, private payload publication, corpus accuracy claim or10× completion.

This applies the Integer Alibi method of fixed operands, intermediate bit
checks and identified executing kernels before whole-model comparisons. It
retains original Q8 scales/order and never assumes whole-K integer accumulation
or a different quantizer is lossless. Next investigate shared Q/K/V panels
through one operation with contiguous output slices, retaining the same full-
input gates. That proposal is implemented by the grouped-QKV experiment below; all parts of
the pipeline still need improvement to meet the original target.

## Shared conversion and fusion boundaries, 2026-10-09

[MonoMoE (2026)](https://arxiv.org/html/2609.04244) uses a persistent CUDA
weight-major pipeline combining routing, quantization, projections, activation
and reduction. Its H200 FP8 routed-MoE measurements compare the complete
operator boundary against tuned vLLM grouped GEMM, with1.02–1.54× kernel gains
across the reported shapes and up to18.7% TPOT reduction. Numerical validation
uses an FP32 reference after BF16/FP8 conversions, cosine similarity at least
0.998, and downstream task scores within reported sampling uncertainty.
That is weaker than raw-bit/token identity and is not CPU dense-Q8 evidence.
Inference: share input conversion and scratch only where the actual dependency
and consumer boundaries permit it, preserving our own quantizer and reductions.
Weight-major GPU streaming and HBM throughput cannot be assumed on this Xeon.

[ClusterFusion++ (April2026)](https://arxiv.org/html/2604.23553) fuses complete
Pythia/GPT-NeoX decoder blocks with CUDA cluster collectives and persistent
TensorMaps/buffers on an RTX5090. It reports up to1.34× TPOT speedup, but
occasional FP16 atomic mismatches and a99.4% overall token-match rate do not
satisfy MOSS's exact-token acceptance rule. Its single-pass variance formula,
changed reduction tree and approximate activation intrinsics need independent
bit proofs before adoption here. The paper's fusion ablation motivates measuring
the whole fused consumer region even when an isolated component is slower.
Inference: our grouped CPU QKV shares only conversion/panels and leaves attention,
bias and activation operators intact; it does not reproduce CUDA full-block fusion.

The [grouped QKV protocol](encoder-qkv.md) preserves independent Q8 scales and
FMA trees, owns one per-encode callback context per group, and exposes contiguous
views. Its gate counts both physical callbacks and logical consumers, tests
actual-worker failure and recovery, and adds a direct full-input pair comparison
against the previous65584 path. Native17 CTests/40Python checks, all14encoder chunks and60full-input runs pass.
All36 same-build token traces match and all11 full-input gates pass. The direct
grouped/separate change is−0.22% to+0.37% by two-repeat point estimates, with
no convincing incremental gain; the shared control includes the generalized
single-projection callback. Keep the result opt-in and retain its complete
dispersion/failed-to-improve evidence. The original tenfold goal is unachieved.

Next test real stateful batched target append against sequential decoding before
implementing drafts. Compare all hidden/logit bits and active K/V bytes at prefix
length boundaries, prove rewind and changed-token recovery, then measure actual
append+all-logits totals with matched workers and retained samples. The current
T>1 masked path can alter reduction order versusT1; any bit drift prevents timing.
Past-only ngram replay does not prove batched execution, latency or rollback.

## Stateful batched target verification, 2026-10-09

The [model-backed protocol and failed native matrix](target-batch.md) now implement
the proposed stateful gate. Frozen source ade6b8b/tree7388780 uses the existing
private T>1 run through a standalone friend accessor; no production decoder API
or math changes. All96 cases and980,917,056 immutable model bytes are checked.
Default17 CTests/40 Python checks pass;24 T1 and24 T2 cases are exact, but2 T4
and6 T8 cases differ in hidden/logit/KV state, sometimes persisting after rewind
and a changed token. All456 greedy/EOS decisions agree and outputs remain finite.
The strict gate therefore exits1 and timing is blocked. This finite T2 observation
cannot replace a separately declared candidate gate or actual audio/full-input
validation. Preserve the failed attempt and isolate first differing operators.

[SpecStream (27 September2026)](https://arxiv.org/abs/2609.33184) motivates separating
committed history from rejected candidate state. Its GPU offloading and throughput
evaluation are different from exact CPU arithmetic and the original single-input,
load-included denominator. See the protocol for the limited method inference and
the complementary consumer-hardware verification-cost study. There is no new
performance win or tenfold claim from this audit; production remains unchanged.
