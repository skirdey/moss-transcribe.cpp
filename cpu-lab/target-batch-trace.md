# Protected target-batch stage trace

The strict [state audit](target-batch.md) rejects eight of 96 batched appends even
though greedy token choices match. This diagnostic follows the actual decoder
graph to locate the first differing layer and operation. It does not implement
speculative generation, change layer arithmetic, expose a production batch API,
or measure a speedup. The original tenfold full-input goal remains unachieved.

The [expanded cache/matmul diagnosis](context-operations.md) found a limitation
of the original observation controls: equal final hidden/KV results can coexist
with overwritten intermediates when a same-shape allocation plan ignores new
output-retention flags. Its repair forces reservation only for the lab hook and
requires independent store/copy/replay checks. Original numeric difference
records are retained; they alone do not prove every captured tensor's lifetime.

## Observation protocol

One immutable Q8 checkpoint serves four separately owned decoder/KV states.
Prefill one state, clone every cache byte with readback, then run quiet serial,
quiet batch, captured serial, and captured batch appends. All backend calls are
serialized. The private optional `Qwen3Decoder::run` hook defaults to null; only
the friend lab accessor supplies it. The original graph builder remains the
source of the tensors and operations.

Before allocation, the hook finds 19 stages in each of 28 layers by actual model
weight and cache identities. It marks selected tensors as outputs and copies
their complete F32 storage after computation, while the graph context lives.
Stages cover attention norm, Q/K/V projection, Q/K norm, Q/K RoPE, attention
scores, probabilities, context, output projection, residual, FFN norm, gate, up,
activation, down, and layer output. No captured vectors or token IDs are written.

Require quiet and captured graphs to have exactly equal semantic metadata and
callback identities and logical cache-leaf roles. Compare complete metadata byte vectors; the numeric FNV
fingerprint is a convenience, not an equality proof. For custom2 nodes, inspect
the pinned GGML parameter structure and compare task count and function identity
separately. Require null userdata. Exclude struct padding, unused parameter
storage and callback addresses from the fingerprint; unsupported custom layouts
fail closed. All other operation parameters, node shapes, strides, source-node
indices and named leaf shapes/types must match.

Capture must preserve quiet hidden vectors and complete active K/V bits in both
serial and batch execution. A failing observation control forbids attribution.
Compare every captured stage element at each logical token position. For scores
and probabilities, compare only the valid causal prefix against the corresponding
serial row, and separately require every masked future probability to be finite
zero. Preserve prefix cache, inputs, positions and every loaded model weight byte.

The 26 fixtures include all eight original failures, twelve matched append-1/2
controls and six exact append-4/8 controls. Both builds must first reproduce all
96 original null-hook case records and their summary exactly. The frozen original
executable is also repeated without rebuilding in a new log directory. This
separates repeatability, capture correctness and batch parity.

## Native evidence

The unchanged V1 executable reproduced all 96 original case records and the
summary exactly. Its new repeat used 264 one-second observations with no MOSS
process overlap observed; one-minute host load ranged 8.38–16.18. Original source
and all six binary/library artifact hashes were checked before reuse.

V2 passed 17 native CTests (25.22 s) and 40 Python checks, then reproduced the
entire original 96-case matrix with the null hook. The trace stopped at graph
metadata equality before producing any tensor-stage or case records. V3 passed
18 CTests (25.68 s), including the metadata negative controls, and 40 Python
checks; its full null-hook matrix also matched exactly. Semantic custom2
parameter comparison alone did not resolve the graph guard failure.

V4 linked its diagnostic executable against the unchanged V3 static library and
GGML runtime, with exact core-source equality and before/after library hashes.
It kept the strict guard and reported 108 differing cache-leaf labels, zero
other leaf-label differences, and exact node counts, remaining metadata,
callback identities and cache roles. GGML assigns labels to unnamed leaves when
their first graph is expanded. The prefilling state and newly loaded cloned
states had different graph histories. V5 copies these descriptive labels along
with the complete cache bytes, still checks labels, and adds a deliberate
leaf-label mismatch rejection to the nonmodel controls. V2–V4 source and failed
logs remain frozen; their exact public revisions are retained.

V5 passed 18 native CTests (25.78 s) and 40 Python checks. All 96 null-hook
case records and the summary remained identical to V1. The full trace completed
26 cases and 13,832 stage records with exit 0: zero capture-induced hidden/KV
changes, nonfinite elements, masked-future probability errors, or changed bytes
among 980,917,056 loaded weight bytes. Semantic graph, label and callback guards
passed. The 352 active one-second samples observed no overlapping MOSS process;
one-minute host load ranged 4.86–10.68. Shorter overlaps were not excluded, and
other host jobs remained running.

All eight known final-state failures first differ at the captured layer-0
attention context (stage 10). Its preceding norm, projections, Q/K norm, RoPE,
valid attention scores and probabilities are bit-identical in all 26 fixtures.
The first context discrepancies are small: maximum absolute errors are
5.960464477539063e-8 or 1.1920928955078125e-7. Layer-0 attention projection and
layer output still match exactly for those cases; later layers can propagate
changes into final hidden/KV state. This locates the first observed tensor
boundary, not the precise faulty kernel or cache operation.

Final-state controls also distinguish two behaviors: all twelve append-1/2
controls and one large-append control are exact across every captured stage;
five of the six final-state-exact append-4/8 controls have small intermediate
attention-context differences while final hidden/KV state remains exact. Thus
final-state equality can hide intermediate arithmetic drift. No tolerance is
used to promote the eight failing cases, and no batch timing follows.

The complete numeric [report](target-batch-trace-native-v5.json) retains the
original repeat, both failed full builds, V4's field diagnostic, final controls,
compiler flags, source/binary/library/model/log hashes and exact public source
revisions. [All stage records](target-batch-trace-stages-v5.jsonl) are retained,
SHA-256 `ffa9af1b2e77e3e898978e9b92c07b20c504e2ff22a37e353587018613334667`.
The report SHA-256 is
`e796fe1d33bad257a8f41c642257f689f39d033307a1ad79019151a5dbac9184`.
Neither artifact contains tensor values, token IDs, transcripts, audio, weights
or credentials. Automatic processing was restored and the authenticated API
retained its validated production binary.

The next diagnostic must separate stored K/V correctness, value-probability
matmul arithmetic, and context permutation/copy at this boundary. Only a new
frozen candidate with complete state/logit/rollback gates can proceed to drafting
or full-input timing. The scalar/SIMD softmax-tail hypothesis is unsupported by
these first-boundary results: layer-0 probabilities match, so softmax is not the
first observed divergence in this suite.

## Reproduce

Use the exact source revision and model hash recorded in the numeric report.
Build with the pinned GGML submodule and native Release CPU flags:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMT_BUILD_TESTS=ON
cmake --build build -j 16
ctest --test-dir build -LE model --output-on-failure
cd cpu-lab
python3 -m unittest test_moss_cpu_regression test_moss_ngram_audit
cd ..
taskset -c 0-15 build/tests/moss_target_batch_gate /path/to/moss-transcribe-q8_0.gguf
taskset -c 0-15 build/tests/moss_target_batch_trace /path/to/moss-transcribe-q8_0.gguf
```

The metadata control alone needs no model:

```sh
build/tests/moss_target_batch_trace --audit-metadata
```

The full trace is built outside default CTest execution. Its metadata negative
controls are registered in CTest. Strict batch exit 1 means a complete arithmetic
rejection; exit 2 means incomplete/invalid evidence. Trace exit 0 means capture
controls passed and the trace is attributable, not that batched append passed
parity. Full-input latency, transcript/timestamp/speaker/token/EOS checks, draft
acceptance costs and human-reference accuracy remain separate gates.

## Research interpretation

[Vosti](https://arxiv.org/abs/2609.38981), submitted September 30, 2026, specifies
bitwise logits at logical output positions across execution variations. Its
engine proof ties KV entries to logical prefixes; its separate GPU kernel
analysis checks batch/query/cache-layout invariance. Our use is methodological:
test state ownership and actual kernel boundaries separately. These finite CPU
fixtures are not a formal proof or a transfer of GPU performance results.

[Thinking Machines' batch-invariance analysis](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
explains how runtime shape choices can change reduction arithmetic even when
repeated identical shapes are deterministic. This motivates comparing stages
at the same logical token, rather than trusting repeated runs or final argmax.
It does not identify this CPU failure's cause; the native trace must do that.

[LLM-42](https://arxiv.org/abs/2601.17768), revised January 30, 2026, uses a fast
batch path with fixed-shape verification and rollback. Replaying verification
has a real cost. If we explore this alternative, timing must include verifier
work and correction of accepted state, not just agreement of sampled tokens.
