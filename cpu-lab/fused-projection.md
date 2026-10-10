# Exact-dot CPU projection fusion

Research bit 32768 replaces eligible one-row Qwen Q/K/V and gate/up groups with
one CPU custom operation per group. It loads original ordinary Q8_0 weights
and calls the exact pinned `from_float` and `vec_dot` routines. Normalization,
RoPE, attention, KV stores, residuals, SwiGLU, logits and generation are unchanged.
Prefill, batched inputs, mixed/strided weights and unsupported CPU dot traits
fall back to ordinary matrix operations. The path is off by default.

Each active GGML worker quantizes the activation into private aligned stack
storage, then computes its independent output rows. This repeats conversion
across workers, but requires neither shared mutable scratch nor a custom worker
barrier. The redundant work is included in graph/full-input timing. There are
no packed weight copies or extra weight bytes. Input width is bounded at 4096.

## Frozen validation

Compiled source `f2176e6fab24c9ea0b32e85fa98a7c814ba93e05` has exactly the public
`0e47797b9df1117a78cde0f0d2456becc6ee0e63` tree
`c2f63127427b6732cd5240ea42e9940d47570bce`. The
[native provenance](fused-projection-native-v1.json) records source/test/library/
binary/log hashes and the pinned GGML commit. All 15 native CTests passed in
29.89 seconds and 34 Python gate/parser checks passed on hp-fury.

The protected projection test has 24 cases at 1/16 workers, normal/zero/extreme
inputs, signed Q8 code extremes, tail widths, two input updates and raw operand
immutability checks. Outputs are finite and match all float bits. The actual
Qwen layer fixture adds 16 fused-mode records, comparing every output/cache
float bit for cached decode and causal prefill at 1/16 workers. It verifies
actual fusion in decode and ordinary prefill fallback. All differences are 0.

## Warm graph probe

All 36 cases at 1/8/16 workers preserve float bits and operands before and after
timing. Five alternating-order samples follow a discarded first round; reported
values are medians per graph invocation. Model load/audio/full generation are
excluded. Input conversion and redundant private worker work are included.

| Workers | Group | Ordinary graph microseconds | Fused graph microseconds | Observed ratio |
| --- | --- | ---: | ---: | ---: |
| 1 | Q/K/V | 223.204 | 216.734 | 1.0298x |
| 1 | gate/up | 335.133 | 307.853 | 1.0886x |
| 8 | Q/K/V | 43.455 | 47.466 | 0.9155x |
| 8 | gate/up | 90.629 | 91.988 | 0.9852x |
| 16 | Q/K/V | 40.372 | 33.419 | 1.2080x |
| 16 | gate/up | 50.342 | 42.563 | 1.1828x |

These are valid protected graph measurements, distinct from the withdrawn
older packed-weight fixture timings. They do not predict full-model gains.

## Complete paired full-input trial

The [original 48-run report](fused-projection-pilot-v1.json) uses fresh processes,
16 physical-core workers, discarded warmups and two opposite-order rounds over
60/120-second speech, public speech, silence and real empty speech. Wall time
includes model loading. Production, frozen fastest opt48 and opt48 in the same
new binary are separate controls; candidate 32816 adds fusion to opt48.

All 48 runs complete at EOS with identical full output/count/stop/return-code
fingerprints for each case and no concurrent MOSS. All 24 numeric runs also
match complete token sequence hashes. Artifacts and route counters pass. Fusion
is observed in 10 candidate runs; the two real-empty runs stop with a first
EOS and have no decode graphs. This is checked explicitly rather than treating
an unused path as exercised. All three latency gates pass; original runner exit 0.

Median fresh-process wall seconds:

| Case | Production | Frozen opt48 | Same-build opt48 | Fused32816 | Ratio to same-build |
| --- | ---: | ---: | ---: | ---: | ---: |
| meeting-60s | 29.458 | 17.646 | 17.754 | 17.394 | 1.0207x |
| dinner-60s | 22.338 | 15.241 | 15.142 | 15.109 | 1.0022x |
| vox-vmaiq-60s | 23.111 | 15.568 | 15.735 | 15.319 | 1.0271x |
| meeting-120s | 86.276 | 32.931 | 32.956 | 32.561 | 1.0121x |
| silence-60s | 8.658 | 8.583 | 8.608 | 8.608 | 1.0001x |
| empty-real-60s | 8.491 | 8.433 | 8.537 | 8.432 | 1.0124x |

Speech medians are only 0.2–2.7 percent faster than same-build opt48; two repeats
and changing shared-host load 14.36–30.01 do not establish statistical confidence
or a quiet-host gain. Large differences against production mostly include the
already-validated cache/softmax options, not projection fusion alone. Silence
was 0.29 percent slower than frozen opt48, below the fixed five-percent rejection
limit. Every slower individual run is retained. Full-sample equivalence does
not establish corpus-level accuracy or zero latency regression on unseen input.

Candidate peak RSS is 1.58–1.62 GiB for short speech and 1.99 GiB for 120 seconds;
these snapshots include shared mappings and are not exclusive physical memory.
Production remains its original validated binary/configuration. Automatic
processing was restored to 1 and authenticated API identity/health verified.
The original quiet opt48 baseline and tenfold target are unchanged and unmet.

## Next arithmetic experiment

An owned-vector AMX probe is public on `codex/amx-q8-lanes` at
`38075936cfccfba2ea19df0233084d3de4d310c3` (tree
`6d7b34ac84078ce68025fc8c55764615cc8126cd`). Sparse activation columns keep
eight separate four-code integer sums. It loads ordinary Q8 weight codes
directly, then preserves pinned FMA chains and the final horizontal-add tree.
A correction retains pinned PSIGNB behavior for activation -128 with a negative
weight. Tests include finite half-scale extremes and source/operand guards.
Supported-hardware arithmetic passed 96 cases and protected timing passed
144 cases with zero float-bit differences. Representative 16-worker widths
were about 2.4–3.0 times slower than pinned SIMD. It is not routed into MOSS;
retain this negative result before further arithmetic work.

The [Intel AMX intrinsics documentation](https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html)
establishes tile configuration and integer products. The
[Linux XSTATE documentation](https://www.kernel.org/doc/html/next/arch/x86/xstate.html)
specifies process-local permission for tile state. Inference: those facilities
can implement a new exact-order probe, but neither establishes numerical
identity or performance for this model. Require supported-hardware arithmetic,
protected timing and actual-model gates before considering integration.
