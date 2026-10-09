# Read-only CPU weight mapping

The mapped transcription route was rejected by the same-build latency gate
and removed from ordinary inference. The source and tests are preserved in
`model-mapping-candidate-v2.patch` against public base
`378256b6f3926f6466fbc87a06f65d9e5e2f15f0`, and in the exact compiled public tree
`35600ba36fba801df381efe79d73e215b6dec239`. Use opt 4144 only in that isolated
candidate reproduction; current main retains copied weight storage.

The candidate's `MTD_CPU_OPT` bit 4096 attaches POSIX CPU tensors to read-only
GGUF file pages; opt4144 adds validated opt48 cache/softmax routing. Mapping
changes storage, without requantizing weights or selecting new matrix math.
Production continues using its pinned binary.

The file's bytes must remain immutable for the loader's lifetime. Atomic path
replacement/unlinking leaves an existing inode mapping valid; in-place writes
or truncation violate that contract. Every tensor range is checked against the
file size before any tensor is attached. Truncated payloads fail loading.
GPU, Windows, unavailable mapping and unsuitable scalar alignment use copied
storage. Small F16 promotion owns a separate writable buffer. The CPU buffer
wrapper borrows pages; the loader unmaps them after releasing that wrapper.
These lifetime rules follow [mmap(2)](https://man7.org/linux/man-pages/man2/mmap.2.html).

`BENCH_MODEL_STORAGE` reports the selected mode and tensor payload bytes.
`copiedBytes` counts tensor uploads into the ordinary backend buffer, not all
process memory traffic. Zero for mapped storage does not exclude metadata,
promotion, activation, KV-cache or page-fault work. The paired harness rejects
a requested mapping if its one storage record does not verify mapped mode,
zero copied payload and valid counters; silently falling back cannot pass.

## Native and checkpoint checks

The fresh hp-fury Release build passed all 11 model-independent CTests in
41.77 seconds and all 27 regression/trace Python tests. The new native test
compares raw F32/Q8 weights and graph output bits; checks promotion, double-load
rejection and mapped graph use after unlinking; rejects truncated payloads
before attachment; and exercises copied fallback with a valid custom-alignment
GGUF. Apple Clang syntax checks also passed. GPU/Windows runtime mapping was
not tested; their copied branch remains the default path.

The full pinned Q8 checkpoint has 684 tensors. Copied and mapped loaders matched
all **980,917,056** payload bytes before and after promotion. This checkpoint's
weights are Q8/F32, so the synthetic native test supplies the F16 promotion
case. The checkpoint probe shares one process/backend and a warmed file cache.
Its 0.8593-second copied and 0.0332-second mapped load diagnostics exclude lazy
page faults until the subsequent full-byte comparison. They are **not** a
full-input speedup, a cold-cache test or a tenfold result.

See `model-storage-probe-v2.json` and `model-mapping-validation-v2.json` for
numeric evidence, source/test hashes and test-log hashes. Compiled private
source `d10902310d8648b0dded6ba04bd1f8a5a897efa1` has exactly the public tree at
`35600ba36fba801df381efe79d73e215b6dec239` (tree
`e10b4ed18f36e925c9ae966977805b4f2718fdfd`). Pinned ggml is
`eced84c86f8b012c752c016f7fe789adea168e1e`.

The first native attempt passed ten checks but failed its alignment fixture:
setting GGUF alignment metadata did not change the writer's cached alignment,
so offsets disagreed with the header. The repaired fixture reopens its
tensorless header before adding tensors. Reverse-applying
`model-mapping-fixture-repair-v2.patch` to the public source above exactly
reconstructs the original failed native source tree. That attempt stopped
before any checkpoint probe or transcription benchmark. It is not a passing
validation result. Frozen experiment roots were retained separately.

## Full-input protocol

The paired experiment uses four variants: pinned production, frozen fastest
opt48, same-build copied opt48, and same-build mapped opt4144. It covers
English/German/public 60-second speech, 120-second English speech and
60-second silence, with two repetitions per variant/case. Each variant has
one discarded full-inference warmup; subsequent measurements use fresh
processes, warm filesystem cache and reversed variant order in the second
round. All use 16 workers pinned one per physical core, default wait policy,
no phase thread overrides, private token tracing off and `MTD_PROFILE=1`
(phase records are available from the instrumented candidates).
No builds or other MOSS inference may overlap timing. Other users' workloads
remain on this shared host.

Wall time includes model loading, lazy faults and complete generation. Gates
require the same audio hash/duration, complete raw output hash, generated-token
count, EOS and return code, no concurrent MOSS, unchanged artifacts and no
per-case median slowdown above 5%. Same-build copied comparison is calculated
separately after the frozen runner exits. This isolates the storage change
from unrelated new-build/reference timing variation. Sample parity is not a
full-corpus WER/DER result or a statistical confidence claim. The original
quiet opt48 latency target for tenfold work remains unchanged.

## Completed paired result: rejected

All 40 measured processes completed with matching input hashes/durations, raw
output hashes, token counts and EOS. Artifact and mapped-storage gates passed;
no competing MOSS was detected. One-minute host load before runs ranged from
33.88 to 41.29. The following are per-case medians in seconds, including load:

| Case | Production | Frozen opt48 | Same-build copied48 | Mapped4144 |
| --- | ---: | ---: | ---: | ---: |
| English 60s | 68.6493 | 37.3546 | 30.6147 | 31.0443 |
| German 60s | 49.4970 | 30.0385 | 30.2023 | 28.4711 |
| Public VoxConverse 60s | 53.6006 | 30.4072 | 31.4084 | 31.0166 |
| English 120s | 240.9816 | 80.0860 | 59.9179 | 73.5674 |
| Silence 60s | 14.7963 | 15.1290 | 14.7013 | 15.9274 |

Mapping was **22.78% slower** on 120-second speech and **8.34% slower** on
silence than the same-build copied control. It also failed silence against
production (+7.64%) and the frozen reference (+5.28%); the original runner
returned status 1. Its long-input peak RSS was approximately 1.99 GiB with
either storage mode, providing no meaningful resident-memory reduction.
The mapped route was archived; ordinary inference retains the complete
original copied loader, source and native test configuration.

`mmap-pilot-v2.json` retains all original report fields and adds one explicitly
separate `postProcessing` object for the same-build comparison. Removing that
object and serializing with the original indentation reconstructs the recorded
original report hash. The postprocessed gate did not change the frozen
runner's exit status. The current harness adds `--candidate-reference 48` to
enforce that comparison automatically in future paired trials. Its failure
test demonstrates that beating two older builds can still hide regression
against the control in the same binary.

The final copied build passed all ten native CTests in 24.81 seconds and five
output/EOS smokes, including long speech and silence. Every native source and
test file is byte-identical to public base `378256b6`. Its frozen Python suite
passed 27 tests; the separately uploaded final harness passed all 28 Python
tests on hp-fury. See `model-mapping-validation-v3.json` and
`model-mapping-final-smoke-v3.json` for exact hashes and provenance. Smoke wall
times are diagnostic, without a paired speed claim. Automatic processing was
restored and authenticated API health was OK after the final controller.

## Reproduction

Build the exact public compiled tree with its pinned submodule. The checkpoint
hash is `ed6c35d0d527c5d03171c3eb448e2150a42c76a51e3e73aa821e351c3da8307c`.
Do not modify it while mapped. Raw private audio/transcripts and model weights
are not distributed here. The public `vox-vmaiq-60s` cut comes from the pinned
VoxConverse preparation documented in the README.

```sh
git clone --recursive https://github.com/skirdey/moss-transcribe.cpp.git source
git -C source checkout 35600ba36fba801df381efe79d73e215b6dec239
git -C source submodule update --init
cmake -S source -B build -DCMAKE_BUILD_TYPE=Release -DMT_BUILD_TESTS=ON
cmake --build build -j 16
ctest --test-dir build -LE model --output-on-failure
python3 -m unittest discover -s source/cpu-lab -p test_moss_cpu_regression.py
python3 -m unittest discover -s source/cpu-lab -p test_moss_ngram_audit.py
MTD_THREADS=16 ./build/tests/moss_model_storage_bench /path/to/model.gguf
```

The host-specific paired runner needs the recorded model/audio cuts, production
binary/libraries and frozen reference build supplied at their explicit paths.
Keep `source/` and `build/` under the candidate root, pause automatic processing,
wait for existing API inference to finish, and restore processing in a `finally`
block. The API service can stay available. Use a new result name every time.

```sh
python3 source/cpu-lab/moss_cpu_regression.py --root /path/to/candidate-root \
  --reference-root /path/to/frozen-opt48 --reference-opt 48 \
  --baseline /path/to/production/moss-transcribe \
  --baseline-lib-dir /path/to/production/lib --model /path/to/model.gguf \
  --audio-dir /path/to/audio --variants baseline cache-reference 48 4144 \
  --cases meeting-60s dinner-60s vox-vmaiq-60s meeting-120s silence-60s \
  --repeats 2 --pin-physical --profile --name mapping-confirmation
```

That historic runner predates `--candidate-reference`. To enforce the new
automatic gate, use the current main harness against the isolated historic
candidate root and add `--candidate-reference 48` to the same arguments. The
historic compiled source remains unchanged. Applying the archived full
candidate patch to a fresh checkout of its stated base reconstructs its exact
public tree; do not apply it over unrelated later changes.

Our changes, probes and evidence remain MIT. No production promotion or tenfold
gain follows from the storage-byte proof or loader-only diagnostics.
