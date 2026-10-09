# CPU frontend and encoder isolation

Research started 2026-10-09 from `4f40141`. The tenfold full-input objective
remains **active and unachieved**. This experiment measures mel, Whisper,
adaptor, prefill, decode and LM-head time directly. All newly tested
transcription routes are archived after their gates failed. Current normal
transcription retains the serial DFT, ordinary loader and opt-48 decoder
behavior. Exact parallel DFT and FFT remain explicit research APIs through
`WhisperMel::Transform`; neither is selected by an environment flag.

## Frozen full-model pilot

The experimental build is frozen at `de5234c`. Its opt-48 control separates the
same compiled source with profiling enabled from the independently frozen
validated opt-48 build. The original production binary is a third reference.
There were six discarded full warmups, then 36 measured fresh processes:
three 60-second inputs, six variants and two rounds with reversed variant
order. Sixteen physical-core workers were pinned; model filesystem caches were
warm. Model, input, binary, library and source hashes were checked. All 36 runs
completed with EOS, artifact integrity passed, and no concurrent MOSS process
was detected. Other CPU work continued; recorded one-minute load was 37.93–47.27.

| Archived selector | Scope | Arithmetic |
|---|---|---|
| 2048 (2096 with opt 48) | ggml workers process disjoint mel frames | Original rounded Hann products, FP32 DFT coefficients, sequential double reductions, mel projection and serial normalization |
| 4096 (4144 with opt 48) | Parallel 400-point real FFT | Double PocketFFT transform of the same rounded samples; differs from the FP32-coefficient DFT |
| 1024 (1072 with opt 48) | Pack only 144 `enc.blk.*` Q8 matrices into ggml AMX buffers | Original codes/scales preserved, different packed arithmetic; text decoder, adaptor and LM head untouched |

| Input | Frozen opt 48 wall | Same-build opt 48 wall | Exact DFT 2096 wall | Exact DFT output | Exact DFT latency gate |
|---|---:|---:|---:|---|---|
| English 60 s | 37.179 s | 44.848 s | 42.524 s | Identical, both rounds | Fail, 14.38% slower |
| German 60 s | 33.583 s | 32.892 s | 35.878 s | Identical, both rounds | Fail, 6.84% slower |
| Public speech 60 s | 48.452 s | 45.625 s | 47.654 s | Identical, both rounds | Pass, 1.65% faster |

The unchanged opt-48 control itself was 20.63% slower on English, while faster
on the other inputs. This small shared-host run cannot isolate instrumentation
or kernel overhead, or establish a reliable new full-model gain. It still
fails the predefined 5% regression gate. Exact DFT routing is therefore
archived; its kernel remains available to explicit research callers.

FFT changed English text despite retaining 674 tokens, segment count,
speakers, timestamps and EOS. German changed from 458 to 456 tokens. Both
changes repeated; public-clip output matched. Encoder-only AMX retained
English output but changed German from 458 to 436 tokens and public speech
from 490 to 476, in both rounds. Its packed buffer was 358,612,992 bytes and
peak RSS reached 2,052,236 KiB versus 1,702,384 KiB for frozen opt 48,
about 342 MiB more. Weight-byte preservation did not preserve model output.
No newly tested route is promoted. The expanded seven-case/120-second suite
and human DER rescoring were not run after these pilot failures. Raw-output
parity is separate from human-reference or full-corpus accuracy.

`mel-pilot-v1.json` preserves all rows and gates. The complete experimental
source is `mel-encoder-candidate-v1.patch`. It applies to a **fresh** checkout
at `4f40141` and reconstructs exactly the `de5234c` tree
`05dab2b52befc03fb3d2cf340f3367a7f358c457`. Use that isolated build for the
archived environment selectors. Current source ignores these three bits.

## Direct phase measurements

The instrumented opt-48 control had these per-case medians (two observations):

| Input | Inference | Mel | Whisper | Prefill | Decode | LM head | Graph build + allocation |
|---|---:|---:|---:|---:|---:|---:|---:|
| English | 43.683 s | 0.854 s | 11.132 s | 3.929 s | 25.054 s | 2.040 s | 0.408 s |
| German | 31.739 s | 0.909 s | 12.150 s | 4.174 s | 12.525 s | 1.379 s | 0.274 s |
| Public speech | 44.494 s | 0.830 s | 13.490 s | 4.724 s | 23.185 s | 1.606 s | 0.289 s |

Adaptor time was 0.025–0.072 s. Mel was about 1.9–2.9% of inference and graph
construction/allocation 0.65–0.94%. Decoder compute and Whisper encoding
remain the main measured phases. Backend compute includes worker scheduling,
barriers, conversions and memory traffic as well as arithmetic; this profiler
does not separate those costs. Removing metadata work alone cannot yield 10x.

`MTD_PROFILE=1` emits one numeric `CPU_PHASE_PROFILE` JSON object on stderr.
Counters belong to the calling thread and reset for each CLI transcription;
custom-op workers do not mutate them. Each phase records calls, graphs, total
seconds, graph construction, allocator work, input upload and backend compute.
Stage counters are subdivisions; do not add them to total seconds. Phase
totals also include local setup/readback. The `other` graph counters cover
unscoped calls but its total is not all unscoped work. Model load remains a
separate `BENCH_TIMING` field; CLI I/O, tokenizer/fusion and remaining generation
work are residuals. Profiling is disabled by default, reads its environment
switch once, and records no text, audio, prompt or credential.

## Native frontend probe and verification

`test_mel_cpu` writes a tiny synthetic-filter GGUF and compares all 240,000
normalized values to the independent retained serial DFT. Signals are silence,
seeded noise, boundary/interior impulses, a tone, a mixed signal and artificial
+/-1e12 alternating values. Both candidates run at one and 16 threads.
All 12 exact-DFT cases matched raw float bits. FFT's 12 cases were finite but
changed values: maximum error over the five ordinary signals was 4.292e-6;
the extreme signal had 0.36395 maximum and 0.07275 RMS error. Finite FFT output
does not assert bit parity, a numerical accuracy tolerance or model quality.
The legacy reference uses rounded coefficients, so drift alone does not say
which transform is closer to a mathematical reference.

The first pinned 16-thread probe samples were 5.14–17.72x faster for exact DFT
and 32.20–65.67x for FFT. These are single frontend samples per signal/variant,
including graph work, allocations, copies and normalization. They have no
confidence intervals and imply no full-model speedup. `mel-kernel-v1.jsonl`
contains all values, including one-thread samples. Its numeric opt labels
identify the historical probe; current tests select explicit `Transform` values.

A fresh intermediate build at `d2492fb`, with FFT/AMX routing removed but exact
DFT routing still present, passed nine native model-independent CTests and
three full-output/EOS smokes against the frozen reference. Those smoke walls
are diagnostics, not a matched speed gate; fresh build work overlapped some
of them. See `mel-intermediate-smoke-v1.json`. After exact DFT routing was also
removed, a separate fresh probes-only build at `0cb554b` passed all nine native
Ctests in 154.47 s. No final-probes-only full-model timing suite is claimed.
Twelve regression-gate and three audit unit tests pass. Apple Clang C++17
syntax checks pass; macOS runtime was not tested. `mel-provenance-v1.json`
pins actual compiled sources, binaries, libraries and report hashes for all
three roots. API health, active/enabled service and automatic processing
restored to 1 were verified after the terminal tests.

## Open-source dependency and reproduction

PocketFFT is unchanged from the [author's repository](https://github.com/mreineck/pocketfft)
at `c90e55b3d529f8efa40ed01a20de22405f45fc65`, under BSD-3-Clause with original
copyrights and LICENSE.md retained. MOSS's own changes remain MIT; weights
retain their separate license and are not distributed here. The FFT probe uses
a double, unnormalized forward real-to-complex transform, 400 real samples to
201 bins, ggml-owned workers, disabled PocketFFT threading and a 16-plan cache.

```bash
cmake -S . -B build -DMT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 4
ctest --test-dir build -LE model --output-on-failure
MTD_DEVICE=cpu MTD_THREADS=16 ./build/tests/test_mel_cpu > mel-kernel.jsonl
# Optional real GGUF filters (still synthetic probe signals):
MTD_DEVICE=cpu ./build/tests/test_mel_cpu /path/to/model.gguf
MTD_PROFILE=1 MTD_DEVICE=cpu MTD_THREADS=16 MTD_CPU_OPT=48 \
  ./build/moss-transcribe transcribe /path/to/model.gguf /path/to/audio.wav
```

To reproduce rejected routes, make another fresh checkout at `4f40141`, apply
`mel-encoder-candidate-v1.patch` and build it. A frozen root contains `source/`
and `build/`. Then run the public paired harness:

```bash
python3 cpu-lab/moss_cpu_regression.py --root /path/to/experimental-root \
  --reference-root /path/to/validated-opt48-root --reference-opt 48 \
  --variants baseline cache-reference 48 2096 4144 1072 \
  --cases meeting-60s dinner-60s vox-vmaiq-60s --repeats 2 \
  --pin-physical --profile --name mel-pilot
```

Supply model, audio and baseline paths through its options. Private fixtures
are represented by hashes only; the public VoxConverse cut is reproducible
from the root README's pinned dataset instructions. Never mix a changed source
or binary into an existing run directory. `moss_cpu_smoke.py` can compare a
later build's complete outputs to a frozen report; pass `--opt 48` for current
normal inference. It does not replace paired latency or human accuracy checks.

Next work should test matched thread budgets and compute scheduling, compatible
projection fusion without duplicate weight storage, and exact encoder/multirow
AMX arithmetic. The pinned multirow SGEMM path must be tested directly; exact
single-row dot results are not proof for it. Verified drafting and persistent
batching still need separate output and full-input gates, with throughput
reported separately from single-input latency.
