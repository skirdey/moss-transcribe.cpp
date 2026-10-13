# First operator mismatch on actual encoded audio

The rejected batched target first differs from serial opt48 at **layer 0,
softmax probabilities, query 0**. Scores and the eight earlier attention stages
are byte-exact at that layer for all five queries. This is operator attribution
on the retained failing audio boundary; it is not a repaired kernel or a timing.

The same private 60-second input produces an 859-token encoded prefix. The
serial model commits 128 generated tokens, leaves its known seed outside the
cache at past987, and proposes four tokens using only committed history plus
that seed. The experiment clones the same owned state and appends these actual
five inputs. Cache capacity4971 and all phase budgets16 are retained.

## Why the length matters

Each serial query normalizes only its valid positions. The batched graph uses a
common length992 and masks future positions with negative infinity. Replaying
the pinned GGML softmax exactly reproduces each original result. Restricting
the batched scores to the serial valid length reproduces the serial result
exactly at the first layer. The masked positions themselves have zero probability.

| Query | Serial valid length | Batched length | Common vs valid probability bits differing | Maximum absolute difference |
|---|---:|---:|---:|---:|
|0|988|992|1775|5.960464477539063e-8|
|1|989|992|991|5.960464477539063e-8|
|2|990|992|2604|5.960464477539063e-8|
|3|991|992|3782|1.7881393432617188e-7|
|4|992|992|0|0|

All own-result softmax and context replays have zero bit differences. Both
serial and batched context replays use their actual valid-length dot product.
This isolates the first mismatch to length-dependent softmax arithmetic;
the earlier causal-context repair does not fix it. Later layers inherit changed
operands, so their truncated softmax replay is not an independent serial match.

Across the complete append,2048 hidden floats,303872 full-vocabulary logits
and104448 active KV values differ. The first changed final hidden row is query3,
which is later than the first changed intermediate operator. Comparing only a
final argmax cannot establish internal equivalence.

## Observer and replay controls

Six owned clones compare null-hook execution, metadata-only observation and
protected intermediate capture, for both serial and batched execution. Every
observer/capture hidden float, full vocabulary logit and active KV byte matches
its respective null-hook execution. Graph shapes, strides, semantic custom-op
parameters and callback identities match; padding and pointer addresses are
excluded. The non-model control rejects changed tasks, callbacks, userdata,
shapes and strides. All84 context operations are built and actually executed.

The capture covers all28 layers,19 stages and5 queries:2660 stage records and140
attention replays. Replay operands, encoded input, prefix cache, positions and
all980917056 checked weight bytes are unchanged. No nonfinite valid values or
nonzero masked probabilities appear. The trace exits0 because attribution
controls pass; the retained state mismatch remains a failure to repair.

## Evidence and reproduction

- [Complete numeric records](actual-audio-attention-v1.json); no audio, token IDs,
  raw hidden/logit/KV values or weights are exported.
- [Frozen source and native control provenance](actual-audio-attention-provenance-v1.json).
  Native source15c655fa431c6a216071adb16f9fba61de9eec71 and public
  [ab34d33](https://github.com/skirdey/moss-transcribe.cpp/commit/ab34d3305a4f466dd658a481b510de25d6e066a3)
  have identical tree8fe99d5a966e8d161c016617123229b260e14c55.
- Native GCC13.3 Release on the CPU test system passes23 default CTest in26.02s and46
  existing Python controls. The later numeric collector passes17 mutation and
  coverage checks, alongside those46 Python checks, on the laptop.
- This diagnostic intentionally retains the known private audio boundary.
  Other inputs are not a substitute for reproducing that boundary.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMT_BUILD_TESTS=ON
cmake --build build -j16
ctest --test-dir build -LE model --output-on-failure
# Explicit model-backed test; expensive capture is excluded from timing gates.
taskset -c 0-15 build/tests/moss_audio_attention_trace MODEL PRIVATE_AUDIO > PRIVATE_LOG
python3 cpu-lab/moss_attention_audit.py --log PRIVATE_LOG --output NUMERIC_REPORT
cd cpu-lab
python3 -m unittest test_moss_attention_audit
```

The next repair must preserve each query's serial softmax length and arithmetic,
then pass full hidden/logit/cache and complete output/token/EOS gates. Batch-cost
and acceptance failures remain separate: the previous full-input speculative
candidate was10–41% slower on speech. This evidence provides no latency gain,
production promotion or progress claim toward the10x timing target by itself.
