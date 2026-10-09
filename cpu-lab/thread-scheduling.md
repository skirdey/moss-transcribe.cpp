# CPU thread scheduling and past-only draft audit

The tenfold full-input latency objective remains active and unachieved. The
primary reference is the frozen, previously validated opt-48 Q8 build, whose
quiet 60 s speech runs took about 11.64–13.89 s including model loading. A new
shared-host control that happens to run much slower cannot replace that target.
Production opt 0 is measured separately. All source changes here are MIT.

## Protocol and reproducibility

The thread pilot uses fresh processes, the same model/audio hashes, 4096-token
EOS bound, full raw-output hashes, two repeats with reversed variant ordering,
common physical-core places 0–15, and numeric phase profiling. Outputs include
the complete text, timestamps and speaker markers. This sample-equivalence
gate is separate from public-corpus WER/DER/JER scoring. Other users' TTS jobs
remain active; host-load snapshots are recorded. CPU builds and other MOSS
inference are sequenced outside the timing suite. The API remains available;
laptop automatic processing is paused and restored by the private controller.

`thread-harness-v1.py` preserves the exact harness used for the first pilot.
Its root contained source at `cbf3878` and a symlink to the unchanged native
probes-only build at `0cb554b`. Native source files were compared before reuse;
the frozen hash identities and final artifact gate accompany the report.
No source or binary was rebuilt inside that measured directory.

```bash
# Use a frozen root with source/ and build/; supply your own model/audio paths.
python3 cpu-lab/thread-harness-v1.py --root /path/to/frozen-root \
  --reference-root /path/to/frozen-opt48-root --reference-opt 48 \
  --variants baseline cache-reference 48@16 48@8 48@4 \
  --cases meeting-60s dinner-60s vox-vmaiq-60s --repeats 2 \
  --pin-physical --profile --skip-warmup --name thread-pilot-default-v1
```

The follow-on suite skipped new warmups after the preceding returned warmups
and three full-model trace smokes primed the same model filesystem cache.
It completed all 30 timed rows. The startup thread budget is 16; `OPT@THREADS`
sets both MTD_THREADS and OMP_NUM_THREADS for that candidate. All variants
clear inherited OpenMP wait/dynamic/thread-limit settings; eight-thread spread
placement uses a subset of the same 16 places rather than a narrower mask.

The initial planned 36-run grid included `48@16-passive`. That variant's last
discarded warmup exceeded the explicit 300 s timeout, before any timed rows
were recorded. The preceding five warmup subprocesses returned, but their
discarded EOS outputs were not retained. The original before/after artifact
gate was not persisted on this failure; `thread-warmup-failure-v1.json` contains
later current-artifact hashes and this limitation. This is a bounded negative
observation, not a paired passive-wait performance result.

[GCC's wait-policy documentation](https://gcc.gnu.org/onlinedocs/libgomp/OMP_005fWAIT_005fPOLICY.html)
and [spin-count documentation](https://gcc.gnu.org/onlinedocs/libgomp/GOMP_005fSPINCOUNT.html)
motivated that experiment. The installed GCC 13.3 implementation and native
ggml OpenMP barriers were inspected. Setting PASSIVE/GOMP_SPINCOUNT=0 cannot
be assumed to help a graph with repeated small nodes and barriers.

## Completed uniform-thread results

[`thread-pilot-default-v1.json`](thread-pilot-default-v1.json) preserves every
timed row, source/library/model hashes, loads, phases, raw-output hashes and
both reference gates. All 30 runs reached EOS with identical full outputs,
token counts and input hashes; no competing MOSS was observed. The artifact
gate passed. Model-independent CTests for this unchanged probes-only binary
passed earlier; the recorded harness hash exactly matches the frozen file.

Median **full wall seconds including load**, two repeats per fixture/variant:

| 60 s fixture | Frozen opt 48, 16 threads | Candidate 16 | Candidate 8 | Candidate 4 | 8-thread/reference speedup |
|---|---:|---:|---:|---:|---:|
| English meeting | 41.484 | 42.192 | 34.207 | 59.608 | 1.213x |
| German dinner | 39.460 | 40.168 | 30.934 | 52.242 | 1.276x |
| Public VoxConverse vmaiq | 41.461 | 33.484 | 30.556 | 52.525 | 1.357x |

Eight threads passed the per-fixture frozen-reference latency/output gate.
Four threads failed it by 43.69%/32.39%/26.68% slowdown respectively, so the
whole candidate-grid exit status was 1. The old-production opt-0 gate passed
but is not the optimization target. No inference configuration was promoted.

The one-minute host load ranged from 36.85 to 47.01 on 16 physical cores.
The same-opt 16-thread candidate was 1.7–1.8% slower for English/German but
19.24% faster for VoxConverse, illustrating substantial shared-host or build
variation. These two-repeat timings do not establish a quiet-host gain or
isolate a particular kernel change. They remain far above the quiet 10x goal.

Direct candidate phase medians support the mixed-budget hypothesis: English
decoder time fell from 21.403 s at 16 threads to 11.791 s at eight, whereas
Whisper rose from 11.272 s to 13.193 s. German decoder time fell 18.284→8.557 s,
and Whisper rose 12.861→13.599 s. These are observations under shared load,
not isolated causality claims. Candidate peak RSS stayed about 1.58–1.62 GiB.

## Bounded phase worker budgets

`MTD_THREADS_WHISPER`, `MTD_THREADS_ADAPTOR`, `MTD_THREADS_PREFILL`,
`MTD_THREADS_DECODE` and `MTD_THREADS_LOGITS` are optional CPU budgets parsed
once at backend startup. Missing or invalid values keep MTD_THREADS; positive
values above the startup count clamp to it. The existing process-wide backend
and allocator remain serialized. RAII scopes restore the previous count after
a phase or exception, including nested scopes; they never enlarge an existing
persistent pool. Defaults retain the startup count throughout.

`test_cpu_threads` measures the actual ggml custom-operation callback worker
count, rather than just checking a stored setting. It verifies 16→8→4→8→16,
output bits, exception restoration and invalid/over-budget fallbacks with
persistent pool metadata enabled. It is registered as a model-independent
CTest. A fresh native build must pass it before model trials.

The mixed-phase pilot keeps Whisper, adaptor and prefill at 16 while decode
and logits use eight. The harness clears all inherited phase overrides; these
two flags apply only to numeric candidates. Production and frozen references
receive no overrides. A fresh root is used for this changed native source.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMT_BUILD_TESTS=ON
cmake --build build -j 16
ctest --test-dir build -LE model --output-on-failure
python3 -m unittest discover -s cpu-lab -p test_moss_cpu_regression.py
python3 -m unittest discover -s cpu-lab -p test_moss_ngram_audit.py
python3 cpu-lab/moss_cpu_regression.py --root /path/to/fresh-frozen-root \
  --reference-root /path/to/frozen-opt48-root --reference-opt 48 \
  --variants baseline cache-reference 48 \
  --cases meeting-60s dinner-60s vox-vmaiq-60s --repeats 2 \
  --pin-physical --profile --decode-threads 8 --logits-threads 8 \
  --name phase-thread-pilot-v1
```

## Actual-token replay audit

`MTD_TRACE_TOKENS=1` logs the actual greedy token IDs at generation completion.
It is off by default. **Keep these logs private: token IDs reconstruct the
transcript.** The first implementation's single array was truncated by the
logger's 2048-byte buffer, despite three matching full-output/EOS smokes. The
strict parser rejected it. The repaired logger emits at most 128 IDs per
record with EOS, total count and contiguous offset. Missing, reordered,
duplicate, truncated or inconsistent records and missing EOS are rejected.

```bash
python3 cpu-lab/moss_cpu_smoke.py --root /path/to/frozen-root \
  --reference-report /path/to/validated-paired-report.json --opt 48 \
  --threads 16 --trace-tokens --name private-trace-smoke
python3 cpu-lab/moss_ngram_audit.py /private/path/meeting-60s.log \
  /private/path/dinner-60s.log /private/path/vox-vmaiq-60s.log \
  > numeric-replay.json
```

Drafts copy only a matching ngram's known continuation from the committed
prefix plus the already available greedy seed. The replay uses future trace
tokens only to score acceptance, never to choose a draft. Maximum drafts of
2/4/8 are compared. Aggregate counts include rejected target-token work, the
complete EOS and log/script hashes; the report omits token IDs and text.

This is an **oracle acceptance/work estimate**. No batched target, causal-mask
verification, logits comparison, KV rollback or actual speculative speedup is
implemented. A reduced call count alone is not a numerical, quality or latency
gate. Longer target inputs can cost more, and timestamp/speaker markers can
have poor draft acceptance. Encoder and loading costs still limit full-input
speedup. Production promotion and any tenfold claim require separate evidence.

The repaired native build at `bbd30c4` passed all nine model-independent
CTests and 23 Python tests. All three full-output/EOS smokes matched the frozen
reference (674/458/490 tokens including EOS), with no concurrent MOSS and a
passing artifact gate. [`trace-smoke-v2.json`](trace-smoke-v2.json) preserves
source, binary, model, fixture and library hashes; its wall times are diagnostic
smokes rather than paired latency claims. The earlier truncated attempt is
retained in [`trace-truncation-failure-v1.json`](trace-truncation-failure-v1.json)
and [`trace-smoke-v1.json`](trace-smoke-v1.json).

[`ngram-replay-v2.json`](ngram-replay-v2.json) contains all nine aggregate replay
rows. The full repaired traces passed the count/offset/EOS parser. Results for
eight-token drafts, with ngram lengths 2–16:

| Fixture | Sequential target calls | Oracle replay calls | Call reduction factor | Accepted/proposed drafts | Target token-work ratio |
|---|---:|---:|---:|---:|---:|
| English meeting | 673 | 509 | 1.322x | 164/1635 | 3.186x |
| German dinner | 457 | 342 | 1.336x | 115/876 | 2.665x |
| Public vmaiq | 489 | 398 | 1.229x | 91/730 | 2.307x |

Two-token drafts had call reduction factors of 1.272/1.277/1.164x and work
ratios 1.473/1.344/1.342x respectively. Four-token drafts saved almost as many
calls as eight for much less rejected work; vmaiq's call count was identical
at four and eight. Even perfect amortization of calls would give only a modest
decoder improvement before accounting for extra target work, encoder, prefill
and loading. A real verifier could be slower. This evidence does not justify
prioritizing this ngram draft for the tenfold full-input objective.

## Public source identities

Reports retain the actual local frozen experiment commit IDs and file hashes.
Browser-created public commits have different history, so these mappings were
checked with Git diffs rather than assuming those IDs are publicly fetchable:

- Uniform-thread native source `0cb554b` has identical `src/`, `tests/`,
  CMakeLists and PocketFFT files at public PR #4 merge `dd6b1f8`. Its exact
  executed Python harness is the public `thread-harness-v1.py` (SHA-256 matches
  the report). The source root at `cbf3878` did not rebuild that native binary.
- Initial trace source `7abfbc7` has identical native source, tests and the five
  executed/validated Python scripts at public `53b4c6`.
- Repaired trace source `bbd30c4` is reproduced by public `53b4c6` plus
  [`trace-chunk-repair-v2.patch`](trace-chunk-repair-v2.patch); a full Git-tree
  diff found exactly the three patched files. This preserves the nine-CTest,
  23-Python-test build before the new scheduling scopes.
- Bounded worker source `ed49b96` has an identical **complete Git tree** at
  public `1ecadbd`. Later documentation/report commits do not alter its native
  source or Python harness/tests. The native/model trial root remains frozen.

To reproduce the repaired trace build separately, save the public patch first,
then use a fresh checkout at `53b4c6`, apply the saved patch and build/test as
above. A different build path can change ELF/RPATH binary hashes; compare the
recorded source files, compiler configuration, model and behavior as well.
