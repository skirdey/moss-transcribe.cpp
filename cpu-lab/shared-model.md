# Shared Q8 conversion in actual MOSS graphs

This experiment reuses the pinned CPU Q8 activation conversion for independent
projections. It keeps ordinary checkpoint weights, `ggml_mul_mat`, attention,
normalization, positions, KV cache stores and generation. It does not replace
F32/F16 weights or introduce a lower precision than the existing Q8 matmul.

The explicit research switches are 8192 for Qwen Q/K/V and gate/up, and 16384
for Whisper Q/K/V. The Qwen trial uses block conversion below 256 input rows
and shared `ggml_cast` above that threshold; Whisper uses shared cast. These
are trial policies based on the preceding synthetic probe, not universal
performance recommendations. The helper falls back to the original F32 input
unless all consumers are contiguous, two-dimensional Q8_0 weights with the
same input width on the selected CPU backend. Defaults keep the original path.

## Validation protocol

The native test exercises the actual Qwen layer builder at checkpoint widths,
with cached decode and causal prefill, RoPE, GQA, residuals and SwiGLU. It compares
all output and persistent cache float bits, checks finite values and retained
input/weight bytes, and performs two input updates at 1/16 workers. Sequence
lengths 1, 3, 64 and 257 cover decode, prefill and both conversion policies.
This is synthetic layer validation; full-model outputs need a separate gate.

The full-input runner compares production, the immutable fastest validated
opt48 binary and opt48 in the same new binary. Numeric candidate variants
8240, 16432 and 24624 isolate decoder, encoder and combined reuse. It includes
60/120-second speech, public speech, silence and real empty speech, with two
rounds in opposite variant order. Wall time includes model loading. Outputs,
speaker/timestamp markers, token counts and EOS must match all controls.

`--trace-tokens --candidate-reference 48` additionally requires identical
complete same-build token sequences with one terminal EOS. Older production
and frozen binaries lack the new trace facility; their existing full-output,
count and EOS gates remain distinct. Reconstructable token IDs and private
transcripts stay in private stderr files; public reports contain only canonical
hashes/counts. Native/Python tests reject incomplete trace tails, changed token
IDs hidden by the same text/count and silent fallback from the requested route.
Profiling records shared-conversion nodes/consumers by phase, proving the
requested model paths were constructed. Shared-host load and slower cases are
retained; these are not quiet-host tenfold measurements.

## Completed full-input pilot

Frozen V2 passed all 13 model-independent native CTests in 25.82 seconds and
33 Python gate/parser tests. The native actual-layer fixture produced 16 records
with zero output/cache float-bit differences. V1 failed two routing assertions
because the test expected DUP for `ggml_cast`, whose pinned implementation
uses CPY; quantization and inference source did not change in the correction.
[Validation provenance](shared-model-validation-v2.json) and the
[exact V1 reverse patch](shared-model-fixture-v1.patch) retain that failed attempt.

The [72-run original report](shared-model-pilot-v2.json) retains two rounds,
all six cases and all slower measurements. Every run completed at EOS without
concurrent MOSS. Full output hashes match production/frozen/same-build controls
for each case; all 48 numeric runs match complete same-build token hashes.
Artifact, route and token gates pass. Peak RSS remains approximately 1.58 GiB
for short speech and 2.00 GiB for long speech; consult the raw measurements
for each variant. These are peak RSS snapshots, not exclusive physical memory.

Median fresh-process wall seconds, including model loading:

| Case | Production | Frozen opt48 | Same-build opt48 | Decoder reuse | Encoder reuse | Both |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| meeting-60s | 36.848 | 22.961 | 22.985 | 23.052 | 23.031 | 22.833 |
| dinner-60s | 28.023 | 19.499 | 19.624 | 19.201 | 17.362 | 20.666 |
| vox-vmaiq-60s | 31.975 | 21.821 | 21.848 | 22.134 | 21.731 | 21.944 |
| meeting-120s | 129.071 | 52.541 | 53.208 | 53.226 | 53.082 | 53.173 |
| silence-60s | 14.754 | 14.377 | 14.476 | 14.529 | 14.205 | 14.480 |
| empty-real-60s | 12.194 | 14.471 | 14.657 | 14.428 | 14.129 | 14.402 |

The original runner exited **1**. Combined reuse on German speech failed both
the frozen control (+5.98%) and same-build control (+5.31%) five-percent
latency gates. The real-empty case failed production for every opt48 control
and candidate: +15.87% to +20.20%, despite identical empty output and one EOS
token. Decoder-only and encoder-only passed the two opt48 comparisons in this
trial, but neither passed all three controls. Shared load was substantial;
the first German encoder-reuse run also accelerated unchanged decoder work,
so its apparent gain cannot be attributed solely to shared conversion.

**No candidate is promoted.** Research flags remain off by default, production
uses its original validated binary, and automatic processing was restored to 1
with authenticated API health OK. Functional/sample equivalence and observed
routing do not establish corpus accuracy, a quiet-host gain or tenfold latency.
The frozen quiet opt48 baseline and original tenfold target are unchanged.

## Additional primary research checked 2026-10-09

| Primary source | Evidence and relevance to this host |
| --- | --- |
| [SpecQuant, uploaded Sep 18](https://arxiv.org/html/2609.21704) | Routes between quantized parents and reports benchmark accuracy differences. It does not establish bit-identical MOSS output or lossless weight compression. Inference: a future drafter could be lower precision while an unchanged Q8 target verifies every accepted token, but changing the target parent violates our equivalence gate. |
| [Mixture-of-Kittens, Sep 28](https://arxiv.org/html/2609.36070) | Deterministic MoE training megakernel on GB300 NVL72; reported production throughput is 1.41x on 512 GPUs. Fixed reduction order is relevant to an exact CPU fusion design. Inference: those training/multi-GPU gains cannot be transferred to this dense single-CPU inference model. |
| [Compiler-assisted speculative sampling, Feb 8](https://arxiv.org/html/2602.08060) | Measures draft/target costs and acceptance, and chooses draft length including zero. Its 1.68x result uses a Cortex-A55/Mali system with the target on one CPU core and drafter on GPU. Inference: an hp-fury trial needs measured CPU-only draft and batched verification cost, not an acceptance-only replay speed claim. |

These sources support further investigation, not a new latency or accuracy
claim for MOSS. The original tenfold full-input target remains unchanged.

## Next fusion experiment

The pinned M=1 CPU matmul converts an activation, synchronizes the worker pool,
then assigns independent output rows to the Q8 `vec_dot` trait. Shared conversion
alone still launches every projection as a separate graph operation. The next
candidate is one CPU custom operation for Q/K/V and another for gate/up, using
the original weight tensors and exact pinned dot trait. This could reduce graph
barriers without the previous packed-weight duplication/load cost. The public
custom callback has no worker-pool barrier argument; a first decode-only trial
can quantize the small input into private per-worker storage and fuse output-row
work. Its redundant conversion must be included in timing, and prefill should
retain the validated ordinary matrix path until separately proven.

The decode-only prototype is now public on `codex/fused-q8-projections` at
commit `0e47797b9df1117a78cde0f0d2456becc6ee0e63` (tree
`c2f63127427b6732cd5240ea42e9940d47570bce`). It routes actual Qwen groups
behind research bit32768, records route counters and has layer/cache fixtures.
Clang syntax checks and 34 Python gate/parser tests pass locally; hp-fury
compilation and timing are pending. This is not a measured gain or tenfold claim.
Deterministic scheduling alone does not imply identity to GGML's reduction
order. Require float bits on actual-width synthetic groups/layers, immutable
operands, full same-build tokens, EOS and all three latency controls before
retaining or promoting a fused path.
