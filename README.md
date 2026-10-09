# moss-transcribe.cpp

**Brought to you by the [LocalAI](https://github.com/mudler/LocalAI) team**, the folks behind LocalAI, the open-source AI engine that runs any model (LLMs, vision, voice, image, video) on any hardware, no GPU required.

[![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/mudler/moss-transcribe.cpp-gguf)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LocalAI](https://img.shields.io/badge/LocalAI-Run_Locally-orange)](https://github.com/mudler/LocalAI)

moss-transcribe.cpp is a from-scratch C++17 inference port of [OpenMOSS MOSS-Transcribe-Diarize](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize), built on [ggml](https://github.com/ggml-org/ggml). It does joint long-form transcription, speaker diarization, and timestamping in a single pass, on CPU (and on GPU through ggml's backends), with no Python, PyTorch, or CUDA toolkit at inference time. Everything lives in one self-contained GGUF, and the output is bit-for-bit identical to the reference model: every component is validated at cosine 1.0 against the genuine PyTorch model, and the end-to-end transcript matches it exactly. On CPU it is about 1.8x faster than the reference PyTorch runtime on the same audio and threads.

![moss-transcribe.cpp vs PyTorch on CPU: identical transcript, moss-transcribe.cpp finishes first](benchmarks/media/moss_transcribe_race.gif)

> The same audio, side by side: the identical timestamped, speaker-labelled transcript, moss-transcribe.cpp (ggml CPU) gets there first, 1.6 to 1.8x faster than PyTorch and on about 1.5x less RAM ([full benchmarks](benchmarks/BENCHMARK.md), [square cut for social](benchmarks/media/moss_transcribe_race_square.gif)).

The model emits a compact, time-aligned, speaker-labelled transcript in one autoregressive stream, for example:

```text
[0.28][S01] And so, my fellow Americans, ask not what your country can do for you,[7.71][8.12][S02] ask what you can do for your country.[10.59]
```

Timestamps are in seconds, speakers are relative per-recording labels (`[S01]`, `[S02]`, and beyond) assigned in order of appearance. There is no separate ASR-plus-diarization pipeline: the model writes the speaker tags and times itself, and this port reproduces that stream token for token.

---

## What it is

MOSS-Transcribe-Diarize 0.9B is an end-to-end audio understanding model for multi-speaker transcription, diarization, and timestamps. moss-transcribe.cpp reimplements its full inference graph in ggml:

| Component | Specification |
|---|---|
| Audio encoder | Whisper-Medium encoder (24 layers, 80-mel, 30 s chunks) |
| Audio-text bridge | 4x temporal merge + MLP adaptor (VQAdaptor) |
| Text backbone | Qwen3-0.6B causal decoder (28 layers, GQA, QK-norm, NEOX RoPE) |
| Fusion | audio features replace `<|audio_pad|>` embeddings (masked_scatter) |
| Output | `[start][Sxx]text[end]` transcript with inline speaker tags and time markers |

Long audio is handled the same way the reference does it: split into 30 s chunks, encode each, concatenate, and interleave absolute-time markers into the audio token stream so the decoder can anchor timestamps across the whole recording.

---

## Performance

moss-transcribe.cpp is faster than the reference PyTorch runtime on CPU, on the same audio and the same thread budget, with a byte-identical transcript. Numbers below are a warm, isolated run on a 20-core x86 CPU at 8 threads (the sweet spot: the autoregressive decode is memory-bandwidth bound, so more than 8 threads does not help and 20 threads is slower than 8). RTF is processing-seconds over audio-seconds, so lower is faster and below 1.0 is faster than real time. "Inference" excludes the one-time model load (about 1.4 s for the mmap'd F32 GGUF).

| Audio | moss-transcribe.cpp | PyTorch (torch CPU) | Speedup |
| ----- | ------------------- | ------------------- | ------- |
| 11 s  | 6.5 s (RTF 0.59)    | 10.5 s (RTF 0.96)   | 1.62x   |
| 44 s  | 24.2 s (RTF 0.55)   | 43.0 s (RTF 0.98)   | 1.78x   |
| 132 s | 102.8 s (RTF 0.78)  | 162.1 s (RTF 1.23)  | 1.58x   |

Inference time (one-time model load excluded), warm run. Both engines emit the identical transcript. moss-transcribe.cpp stays under real time (RTF below 1.0) across the range; PyTorch crosses 1.0 by 132 s.

![RTF vs audio length: moss-transcribe.cpp vs PyTorch on CPU](benchmarks/media/rtf_vs_length.png)

Two honest caveats:

- **RTF grows with audio length.** The decode is autoregressive over a context that grows with the audio (more audio tokens, a longer transcript), so per-second cost rises with duration. This is inherent to the model, and it affects the reference the same way. For hour-long audio on CPU you are looking at a long run for either engine, the reference model targets GPU. moss-transcribe.cpp runs on GPU too (see below): CUDA is verified bit-exact on NVIDIA Blackwell, where it is roughly on par with PyTorch.
- **These are the F32 CPU numbers.** F16 and quantization (q8_0/q6_k/q5_k/q4_k) are available now and cut the model from 3.4 GB down to 511 MB with the transcript still byte-identical through q5_k (see [Quantization](#quantization)). The ggml GPU backends work too: CUDA is verified bit-exact on an NVIDIA Blackwell GPU and runs roughly on par with PyTorch there (full numbers in [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md)). The headline wins are correctness (bit-exact), portability (no Python/PyTorch/CUDA at inference), a real CPU speedup over PyTorch, and much smaller quantized models.

Full methodology and the reproducible harness are in [`benchmarks/BENCHMARK.md`](benchmarks/BENCHMARK.md).

---

## Build

Clone with submodules (ggml is vendored at `third_party/ggml`):

```sh
git clone --recursive https://github.com/mudler/moss-transcribe.cpp
cd moss-transcribe.cpp
cmake -B build -DMT_BUILD_TESTS=ON && cmake --build build -j
```

Use `-DGGML_NATIVE=OFF` for portable or CI builds. For the shared library (LocalAI / dlopen), build with `-DMT_SHARED=ON`.

### CMake options

| Option | Default | Purpose |
|---|---|---|
| `MT_BUILD_TESTS` | OFF | Compile and register ctest targets |
| `MT_BUILD_CLI` | ON | Build the `moss-transcribe` CLI |
| `MT_SHARED` | OFF | Build the library as a shared object |
| `MT_GGML_CUDA` | OFF | Forward GGML_CUDA to the submodule |
| `MT_GGML_METAL` | OFF | Forward GGML_METAL to the submodule |
| `MT_GGML_VULKAN` | OFF | Forward GGML_VULKAN to the submodule |
| `MT_GGML_HIP` | OFF | Forward GGML_HIP (ROCm) to the submodule |

To build for a GPU, forward its flag and rebuild, e.g. CUDA: `cmake -B build-cuda -DMT_GGML_CUDA=ON && cmake --build build-cuda -j` (Metal/Vulkan/HIP analogously). The backend auto-selects the GPU when present (`MTD_DEVICE=cpu` forces CPU). `scripts/gpu_verify.sh <gguf> <wav> [cuda|metal|vulkan|hip]` builds, checks the GPU transcript is byte-identical to CPU, and reports GPU vs CPU speed. GPU parity and benchmarks are validated on real hardware (a build machine with the ggml GPU toolchain).

---

## Get the model

Convert the HuggingFace checkpoint to a self-contained GGUF (needs a one-time Python environment for conversion only, never at inference):

```sh
python3 -m pip install -r scripts/requirements.txt
hf download OpenMOSS-Team/MOSS-Transcribe-Diarize --local-dir models/hf
python3 scripts/convert_moss_transcribe_to_gguf.py models/hf -o models/moss-transcribe-f32.gguf
```

The converter embeds everything the loader needs (all dims, the Whisper mel filterbank, the Qwen2 tokenizer, the time-marker parameters, the default prompt) as GGUF metadata and tensors. Nothing is hardcoded in the C++ and no sidecar config or vocab file is shipped.

### Quantization

The converter emits F32, F16, and q8_0 directly. The K-quants (q6_k/q5_k/q4_k), which the Python `gguf` writer cannot produce, come from the CLI `quantize` command against an F32 GGUF:

```sh
# F16 and q8_0 from the converter
python3 scripts/convert_moss_transcribe_to_gguf.py models/hf -o models/moss-transcribe-f16.gguf  --dtype f16
python3 scripts/convert_moss_transcribe_to_gguf.py models/hf -o models/moss-transcribe-q8_0.gguf --dtype q8_0

# K-quants (and q4_0/q5_0) from an F32 GGUF
./build/moss-transcribe quantize models/moss-transcribe-f32.gguf models/moss-transcribe-q6_k.gguf q6_k
./build/moss-transcribe quantize models/moss-transcribe-f32.gguf models/moss-transcribe-q5_k.gguf q5_k
./build/moss-transcribe quantize models/moss-transcribe-f32.gguf models/moss-transcribe-q4_k.gguf q4_k
```

Only the large `ggml_mul_mat`-fed weights are quantized (the Qwen3 and Whisper attention/FFN projections, the adaptor linears, and the token embedding, 343 tensors). Norms, biases, the conv stem, positional embeddings, and the mel filterbank stay F32. Size and accuracy on the JFK sample (CPU, greedy):

| dtype | size | vs f32 | wall (11 s, 8 threads) | speed vs f32 | transcript vs reference |
| ----- | ---- | ------ | ---------------------- | ------------ | ----------------------- |
| f32   | 3.4 GB | 100% | 7.83 s | 1.0x | byte-identical (the parity gate) |
| f16   | 1.8 GB | 50%  | 4.96 s | 1.6x | byte-identical |
| q8_0  | 942 MB | 27%  | 3.97 s | 2.0x | byte-identical |
| q6_k  | 733 MB | 21%  | 4.16 s | 1.9x | byte-identical |
| q5_k  | 619 MB | 18%  | 4.47 s | 1.8x | byte-identical |
| q5_0  | 619 MB | 18%  | 3.81 s | 2.1x | byte-identical |
| q4_k  | 511 MB | 15%  | 3.81 s | 2.1x | word-identical (one timestamp off 0.02 s) |
| q4_0  | 511 MB | 15%  | 3.57 s | 2.2x | word-identical (one timestamp off 0.07 s) |

![Quantization ladder: size and speed by dtype, byte-exact through q5](benchmarks/media/quant_ladder.png)

Quantization is a **speed** win as well as a size win: the autoregressive decode is memory-bandwidth bound, so the smaller quantized weights run up to about 2.2x faster than F32 on CPU. F16 through q5_0 reproduce the reference transcript exactly (greedy argmax is robust to the small weight noise); q4_k/q4_0 are word-for-word identical with a hair of timestamp drift. Prebuilt GGUFs are published at [mudler/moss-transcribe.cpp-gguf](https://huggingface.co/mudler/moss-transcribe.cpp-gguf).

---

## Running inference

```sh
# Transcribe + diarize an audio file (wav, 16 kHz mono is loaded and resampled as needed)
./build/moss-transcribe transcribe models/moss-transcribe-f32.gguf audio.wav

# Cap the generated length (default comes from the GGUF)
./build/moss-transcribe transcribe models/moss-transcribe-f32.gguf audio.wav --max-new 4096

# Print model metadata (arch, dims, mel params, vocab size, time-marker settings)
./build/moss-transcribe info models/moss-transcribe-f32.gguf
```

The default output is the raw model transcript, `[start][Sxx]text[end]` segments concatenated into one stream. Pass `--format srt`, `--format ass`, or `--format json` to parse it into structured `{start, end, speaker, text}` segments and export subtitles with the reference's speaker-aware merge and styling (`--format text` is the raw stream, the default). Logs go to stderr, so `--format json > out.json` is clean.

Thread count defaults to all cores, which for this model is usually not optimal. Set `MTD_THREADS` to tune it (8 is a good default on a 20-core box); the decode is bandwidth bound, so fewer busy threads often beat more.

---

## Verification and parity

This is a parity-first port. Every component was gated numerically against tensors dumped from the genuine PyTorch model before the next was built, not just checked end to end:

| Stage | Gate |
|---|---|
| Whisper log-mel front end | cosine 1.0 vs `input_features` |
| Whisper encoder (24 layers) | cosine 1.0 vs `encoder_hidden` |
| Time-merge + VQAdaptor | cosine 1.0 vs `audio_embeds` (merge bit-exact) |
| Qwen2 tokenizer | decode matches the reference text exactly |
| Audio-span + time markers | `input_ids` bit-exact (single and multi-chunk) |
| Qwen3 decoder | cosine 1.0 vs `lm_hidden`, argmax match |
| masked_scatter fusion | bit-exact vs `fused_embeds` |
| Greedy decode | `generated_ids` bit-exact |
| **End to end (wav to text)** | **transcript equals the reference exactly** |

Run the suite (tests labelled `model` skip cleanly when the GGUF and baselines are absent):

```sh
export MTD_TEST_GGUF=models/moss-transcribe-f32.gguf
export MTD_TEST_BASELINE=tests/fixtures/baseline_short.gguf
export MTD_TEST_BASELINE_LONG=tests/fixtures/baseline_long.gguf
ctest --test-dir build -L model --output-on-failure
```

Baselines are produced from the authors' real model and processor with `scripts/gen_baseline.py`, and the CPU speed comparison against upstream PyTorch is `scripts/bench_upstream.py`.

---

## Use it from LocalAI

For a production deployment (an OpenAI-compatible `/v1/audio/transcriptions` endpoint, a model gallery, concurrency, auth, and metrics), use [LocalAI](https://localai.io), which is built to embed ggml engines like this one. A dedicated backend is on the roadmap below.

---

## Roadmap

- **GPU flash-attention.** CUDA is verified bit-exact on Blackwell and on par with PyTorch; the decoder attention still uses a manual softmax + matmul rather than ggml's CUDA `flash_attn_ext`, which is the main remaining GPU headroom (long-context throughput). Metal/Vulkan/HIP compile through the same flags.
- **Flat C-API and LocalAI backend.** A `libmoss_transcribe.so` behind a stable C ABI, dlopened by a LocalAI `moss-transcribe-cpp` backend.

---

## Why moss-transcribe.cpp

The reference is a great model, but running it for inference drags in a heavy Python/PyTorch/transformers stack. moss-transcribe.cpp is a from-scratch C++17/ggml port focused purely on inference:

- **No Python at inference.** One self-contained GGUF and a small C++ binary (or, soon, a shared library behind a flat C API).
- **Bit-exact.** Component-level cosine 1.0 and an end-to-end transcript identical to the reference, proven against dumped reference tensors.
- **Faster on CPU** than the reference PyTorch runtime, on the same audio and threads.
- **Portable.** Runs on CPU today and on any ggml GPU backend as those land.

---

## Citation

If you use moss-transcribe.cpp, please cite this repository and the original model:

```bibtex
@software{moss_transcribe_cpp,
  title  = {moss-transcribe.cpp: a C++/ggml inference engine for MOSS-Transcribe-Diarize},
  author = {Di Giacinto, Ettore},
  url    = {https://github.com/mudler/moss-transcribe.cpp},
  year   = {2026}
}
```

MOSS-Transcribe-Diarize is by the [OpenMOSS / MOSI.AI team](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) (arXiv:2601.01554), released under Apache-2.0.

## Author

Ettore Di Giacinto ([@mudler](https://github.com/mudler)).

## License

moss-transcribe.cpp is released under the [MIT License](LICENSE). The MOSS-Transcribe-Diarize model weights keep their original Apache-2.0 license, check the model card on HuggingFace.

---

Built by the [LocalAI](https://github.com/mudler/LocalAI) team. If you want to run speech transcription and diarization (and LLMs, vision, voice, image, and video models) locally on any hardware with an OpenAI-compatible API, [give LocalAI a star](https://github.com/mudler/LocalAI).


## CPU optimization fork

This public fork preserves upstream history and MIT licensing. Our CPU source changes,
benchmark harness, API server, remote client, and deployment tooling are open source
under the same license; see `cpu-lab/NOTICE.txt`. This is a research branch of the C++
port, not a claim of full-corpus equivalence to the original PyTorch model.

The fastest validated option is `MTD_CPU_OPT=48` (cache + parallel decode softmax;
measurements and quality gates below). Bit 16 stores the F32 value cache transposed
and append new values directly. It removes repeated copies of the growing cache while
retaining reference attention matmul and softmax arithmetic. It is opt-in; unset/zero
keeps the original layout. CPU was tested; GPU use with this experimental flag has not
been validated. Keep experimental flash attention (bit 8) disabled: it changed outputs
and increased one public clip's diarization error. Do not combine bits 8 and 16.

```bash
git clone --recursive https://github.com/skirdey/moss-transcribe.cpp.git
cd moss-transcribe.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DMT_BUILD_TESTS=ON
cmake --build build -j 8
MTD_DEVICE=cpu MTD_THREADS=16 OMP_NUM_THREADS=16 MTD_CPU_OPT=48 \
  ./build/moss-transcribe transcribe /path/to/moss-transcribe-q8_0.gguf /path/to/audio.wav --max-new 4096
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s cpu-lab -p test_moss_cpu_regression.py
```

`cpu-lab/experiments.json` retains all measured and rejected experiments, pinned model,
source, library and binary hashes, and protocol details. On a shared Xeon Gold 5416S,
Q8, 16 threads with one worker per physical core, alternating matched runs measured:

| Case | Reference wall | Transposed cache wall | Speedup |
|---|---:|---:|---:|
| English meeting, 60 s | 35.40 s | 21.34 s | 1.66x |
| German conversation, 60 s | 28.41 s | 19.42 s | 1.46x |
| English meeting, 120 s | 112.39 s | 50.55 s | 2.22x |

All tested raw words, timestamps, speaker markers, token counts and EOS stops matched.
The seven 60-second cases had two paired repetitions; the 120-second stress case had
one pair. Memory rose about 7–8% on 60-second cases. These are small-suite results on a
shared loaded host, not latency guarantees. Private audio is not published; the public
accuracy smoke test uses the first three VoxConverse development rows, first 60 seconds.
Macro DER was unchanged: 5.15% with no collar, 3.51% with a 0.25-second collar, overlap
included. This is not the full VoxConverse or OpenBench benchmark and does not measure WER.

Reproduce the public data preparation with Python 3.12 and the pinned requirements:

```bash
python3 -m venv cpu-lab/.venv
cpu-lab/.venv/bin/pip install -r cpu-lab/moss_cpu_quality_requirements.txt
mkdir -p cpu-lab/data
curl -L 'https://huggingface.co/datasets/diarizers-community/voxconverse/resolve/3acfa1b45ca4b7419aee999d67d94c617f9c9d47/data/dev-00000-of-00005.parquet' \
  -o cpu-lab/data/vox-dev.parquet
cpu-lab/.venv/bin/python cpu-lab/moss_cpu_quality.py --data cpu-lab/data --prepare
```

The preparer verifies the pinned parquet hash and emits 16 kHz PCM16 cuts plus clipped
human references. See the dataset card for CC BY 4.0 terms and cite VoxConverse (Chung
et al., 2020), with Hugging Face packaging by diarizers-community.

`moss_cpu_regression.py` is currently a host-specific paired runner for hp-fury: the
candidate root has `source/` and `build/`; reference binary and model/audio paths are
explicit in its source. Supply `--variants baseline 16 --pin-physical --repeats 3`
and a unique `--name`. Public cases are `vox-rcxzg-60s`, `vox-fsaal-60s`, and
`vox-vmaiq-60s`; copy prepared cuts into the configured audio directory. Keep other
MOSS work paused during measurements and restore it afterward. The gate fails on
changed output, incomplete or unstable reference, concurrent MOSS inference, or a
median slowdown over 5%. `moss_cpu_quality.py --data ... --experiment ... --output ...`
adds a per-case DER gate (no permitted increase). Raw inference outputs remain local.

Native microbenchmarks isolate fused SwiGLU, flash attention, and value-cache matmul.
A kernel speedup alone is insufficient: flash attention was rejected by the output
and DER gates. Native cache tests had zero bitwise differences at lengths 1, 31, 256,
1024 and 2048. Upstream ctest had five passes and nine skips because model fixtures
were unavailable; skipped checks are not evidence of parity.

The historical `moss_cpu_candidate.py` patch applicator is for a fresh pinned upstream
checkout (190a569c13b4b247450f2fb3b2a431244e84833e) with the benchmark/recovery patch
already applied; do not run it on this fork, which already contains those changes.
Bits 1/2/4 are archived experiments without demonstrated end-to-end improvement.

`cpu-lab/moss_api.py`, `moss_output.py`, and `moss_remote.py` provide the authenticated,
bounded CPU API and client. The server serializes inference, rejects incomplete results,
and retries marker loops with bounded repetition penalties. API and client retain
pinned production provenance; the optimized candidate is not deployed to production.
`moss-api.service` and `setup_moss_api.py` preserve the historical hp-fury installation
recipe using the existing pinned production checkout/assets. The recipe enables user
lingering, boot startup and crash recovery; it does not reboot the shared machine.
It is host-specific, not a general-purpose installer. Never commit its generated token
or `data/` directory. API tests can run with `PYTHONPATH=cpu-lab python -m pytest
cpu-lab/test_moss_api.py` after installing pytest. The client needs NumPy, soundfile
and SciPy for audio conversion; the API and latency harness use the Python standard library.


### Parallel decode softmax research

`MTD_CPU_OPT=32` assigns independent attention heads across CPU workers for unmasked,
single-token decoding. The pinned ggml implementation partitions by query row, leaving
all heads on one worker in this shape. Our custom op retains the same scaling, maximum,
vector exponential/sum primitive, double reciprocal and float normalization. It reads
the head dimension from the query tensor, avoiding mutable or borrowed callback state.
Prefill and GPU softmax keep the reference path. `MTD_CPU_OPT=48` combines this with the
validated transposed value cache (16 + 32). This uses ggml's private CPU vector header
from the pinned submodule, so revalidate after any ggml upgrade.

The native `test_cpu_softmax` compares raw float bits against the eager CPU operation
at 1 and 16 threads, nine key lengths, three head/batch shapes and large or infinite
logits. All 54 tested shapes matched exactly. The final confirmation ran 63 matched
fresh-process measurements (seven 60-second cases, three variants, three repeats),
plus a three-way 120-second stress check. All complete raw outputs, words, speaker
markers, timestamps, token counts and EOS stops matched. Source, model, binary and
library hashes stayed unchanged, and no concurrent MOSS inference was observed.

| Input | Production | Validated cache (16) | Cache + parallel softmax (48) | Speedup over cache |
|---|---:|---:|---:|---:|
| English meeting, 60 s | 21.81 s | 14.54 s | 13.89 s | 1.047x |
| German dinner, 60 s | 16.50 s | 12.09 s | 11.64 s | 1.039x |
| VoxConverse rcxzg, 60 s | 19.65 s | 13.54 s | 12.89 s | 1.050x |
| VoxConverse fsaal, 60 s | 19.40 s | 13.44 s | 12.89 s | 1.043x |
| VoxConverse vmaiq, 60 s | 17.30 s | 12.44 s | 11.94 s | 1.042x |
| English meeting, 120 s (one run each) | 78.52 s | 34.08 s | 31.53 s | 1.081x |

Q8, 16 pinned physical-core workers on Xeon Gold 5416S, alternating variant order,
warm filesystem cache, wall time including model load, shared host. Silence and
actual empty speech were essentially unchanged. Peak RSS was 1,702,100 KiB at 60 s
and 2,083,376 KiB at 120 s; the incremental softmax change added under 0.1% relative
to the transposed-cache candidate. The transposed cache itself uses about 5–8% more
memory than production. These small paired samples are measured gains, not a
statistical confidence claim or full-corpus throughput claim.

The human-reference three-clip VoxConverse smoke subset had identical per-case DER
for all three variants: macro 5.1456% at zero collar and 3.5110% at 250 ms collar,
including overlap and optimal speaker permutation. There are no human WER references
in this subset. Full raw-text parity covers transcription regression relative to
production. See `cpu-lab/softmax-confirm-v3.json`, `softmax-long-v3.json` and
`softmax-quality-v3.json` for reproducible hashes, measurements and quality evidence.
The quality report records the final 63-run parity evidence used to validate its
scored outputs. Regression-gate unit tests pass all 12 cases.

The updated paired harness accepts `cache-reference` as a fixed binary at
`--reference-root`. It has a separate gate against that candidate, since beating the
older production binary alone does not prove a new improvement. Model, audio and
baseline paths are configurable, and before/after artifact hashes reject mutated
source or binaries. Run the three-way comparison, for example:

```bash
python3 cpu-lab/moss_cpu_regression.py --root /home/stan/hw-moss-softmax-v1 \
  --variants baseline cache-reference 48 --pin-physical --repeats 3 --name confirm
```

A separate fused eager-attention experiment retained exact dot and full-row softmax
arithmetic and passed 48 native float-bit tests. Its warm kernel timings improved, but
full model inference was substantially slower than the validated cache candidate.
It was rejected and removed from the active source. The archived patch preserves that
research, including its numerical test, without selecting it for normal inference.

The rejected patch `cpu-lab/eager-fused-candidate-v2.patch` applies to a fresh fork
checkout at `0f60f5361dc0e2fe4787d7a8aa4369853eddf530`; use opt bitmask 80
(16 + 64) only to reproduce that failed experiment in an isolated build. Its
per-case median slowdowns versus the validated cache were 43–60%.


### Tenfold performance research

The [CPU research log](cpu-lab/tenfold-research.md) reviews recent lossless
compression, persistent-kernel and verified-drafting papers against this specific
CPU and Q8 checkpoint. The 10x end-to-end objective remains unachieved. The log
contains the actual tensor/entropy audit, a rejected AMX packing experiment,
an exact-order AVX-512 dot and graph kernel, limitations and reproduction commands.
No new kernel from that research is selected in default or production inference.

The reusable graph kernel in `src/cpu_q8.*` passed 36 raw-float and activation-
conversion cases at 1/16 threads, including the 151,936-row LM head. Pinned warm
graphs were 1.18–1.43x faster at 16 threads. Its full-model integration still
failed the paired speed gate: English and German medians were 9–10% slower than
opt 48, with about 603 MiB more peak memory. All 18 measured outputs and EOS
stops matched. The integration is archived as
`cpu-lab/exact-decode-cast-v1.patch` against `338aa8d`; opt 816 is only for that
isolated reproduction. Active loader/decoder source retains opt 48 behavior.


### Frontend and phase isolation

The [mel/encoder research report](cpu-lab/mel-phases.md) adds direct phase and
graph-stage timings (`MTD_PROFILE=1`) and explicit numerical probes. Exact
parallel DFT matched all 12 native float-bit cases and all six paired speech
outputs. Its frontend samples were 5–18x faster, but the 36-run full-model pilot
failed the English/German latency gate; even the same-opt control was unstable
on this busy shared host. FFT and encoder-only AMX changed model outputs. All
three transcription routes are archived in `mel-encoder-candidate-v1.patch`
against `4f40141`; ordinary inference retains the serial frontend and loader.
The kernels remain explicit research transforms, and all nine model-independent
Ctests pass in a fresh probes-only build. No new end-to-end gain or production
promotion is claimed; the 10x goal remains unachieved. PocketFFT retains its
BSD-3-Clause license and attribution; our source changes remain MIT.

### Thread scheduling and private token replay

The [thread scheduling report](cpu-lab/thread-scheduling.md) documents matched
4/8/16-thread trials, the bounded passive-wait warmup failure, and optional
per-phase CPU budgets. `MTD_THREADS_DECODE` and `MTD_THREADS_LOGITS` can reduce
workers for those phases while the encoder retains the startup thread count;
the other phase budgets are also available. Values cannot exceed startup
MTD_THREADS, and defaults retain it throughout. Native tests check the actual
ggml callback worker count, nested scopes and exception restoration.

`MTD_TRACE_TOKENS=1` is an opt-in private diagnostic: the IDs reconstruct text
and must not be published. Chunked records and strict parser tests guard
against logger truncation. `moss_ngram_audit.py` reports past-only draft
acceptance and rejected token work from actual greedy traces. It is an oracle
replay estimate, with no target verification or measured speculative speedup.
Full-input paired gates and corpus accuracy checks remain separate.

The completed 30-run uniform-thread pilot observed 1.21–1.36x speedups at eight
threads; four threads failed the frozen-reference latency gate. A separate
18-run mixed-budget pilot passed output/EOS, artifact and latency gates at
1.10–1.14x speedup. Both ran on a busy shared host; these are not quiet-host or
10x gains, nor a comparison of mixed against global eight. All 48 timed outputs
matched the reference. The final native build passed ten model-independent
CTests and 25 Python tests. Ngram drafts saved few calls while increasing token
work, so no speculative-inference speedup or production promotion is claimed.

### CPU weight mapping and same-build latency gates

The [weight mapping report](cpu-lab/model-mapping.md) preserves the complete
read-only loader prototype and its tests as a reproducible patch. All 684
checkpoint tensors matched byte-for-byte, and all 40 paired outputs, tokens
and EOS matched. Mapping still failed full-input latency: 120-second speech
was 22.78% slower and silence 8.34% slower than copied storage in the same
binary. The route is archived; ordinary inference retains the original loader.

`moss_cpu_regression.py --candidate-reference 48` now also compares candidate
variants against that control in the same compiled binary, alongside production
and frozen-reference gates. The final copied build passed ten native CTests and
five output/EOS smokes; the updated harness passed 28 Python tests on hp-fury.
Full numeric reports and exact source/test hashes accompany the report. Loader
diagnostics and these busy-host trials do not establish a new end-to-end gain.
