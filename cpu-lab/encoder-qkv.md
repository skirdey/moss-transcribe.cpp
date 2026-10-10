# Exact shared Q/K/V conversion and packing

Research bit131072 groups the encoder Q/K/V projections into one GGML custom
operation. The combined candidate is196656 (131072 +65536 +48);65584 keeps
three separate selective VNNI operations, and48 is the same-build ordinary
control. The supported shape is three contiguous Q8_0 [1024,1024] weights and
one contiguous F32 [1024,1500] input, CPU backend and16 actual workers.
Unsupported shapes, layouts, types, configured worker counts and ISA use the
ordinary encoder route. The default production settings remain unchanged.

One callback team converts the input and packs its panels once, then computes
three independent projections. A contiguous [1024,1500,3] parent exposes three
[1024,1500] views to the existing bias/attention consumers. Q8 codes/scales,
eight FMA accumulation chains and the original reduction tree are preserved;
there is no new quantizer or concatenated weight format. Per-encode ownership
spans graph compute and output read. Worker-zero allocation and cleanup remain
inside each callback, with barriers before publishing/consuming/freeing scratch.
An actual worker mismatch poisons every output slice and reports graph failure.

The parent stores V earlier and retains all slices through their consumers, so
fusion can increase peak memory and can lose performance despite less packing.
Complete full-input timing and RSS, rather than node counts, decide acceptance.

## Frozen provenance and checks

Native source `6d6fa55523f1b09ebafc25cb043aaa69575a9bf5`, tree
`bd800b672aa11183cd6ffe1a539a45236a39e0cb`, ggml
`eced84c86f8b012c752c016f7fe789adea168e1e`. Public source
`78c0bd4a3709294268bfc1bb19fa3d60f597cda3` has the identical tree.
The immutable experiment root is `/home/stan/hw-moss-encoder-qkv-v1`.

The protected test uses three distinct external weights, stable external input,
ordinary F32 matmuls, the prior single VNNI path and the grouped path. Normal,
zero, extreme signed codes and extreme half scales each run twice with changed
input and poisoned outputs. It checks every output float bit, every operand,
view offsets/contiguity, actual callbacks/workers and scratch cleanup. A factory
created for16 workers is deliberately dispatched with8 workers: every one of
4,608,000 output elements must become NaN. Restoring16 workers must recover
exact output on the same context. Fallback guards create no custom contexts.

The optional `test_cpu_encoder_qkv --benchmark` uses seven rotating/reversed
three-variant rounds and discards the first. All six samples and their medians
are retained. Quantization, packing, callback allocation/barriers/cleanup are
included; graph allocation, model load and complete inference are excluded.
Real encoder comparison uses identical mel and one loaded model, reference48
first and candidate196656 second, checking all14 chunks of six inputs.
Those encoder-only timings are unbalanced diagnostic observations.

Full-input acceptance uses five variants (production0, immutable frozen48,
same-build48, same-build65584 and196656), six complete inputs, two reversed
rounds and discarded warmups. All60 processes include model load; all36
same-build complete token sequences must match including unique terminal EOS.
Full output hashes/counts/EOS/return codes preserve speakers, timestamps and
real-empty behavior. Every candidate encoder chunk must execute96 custom
callbacks,144 logical projections and24 grouped QKV nodes at16 workers with
zero failures. An additional declared pair gate compares196656 only against
65584, without treating the older48 control as a new candidate. Every reference
and pair comparison rejects a median slowdown above5% on any case.

```sh
ctest --test-dir build -LE model --output-on-failure
cd cpu-lab
python3 -m unittest test_moss_cpu_regression test_moss_ngram_audit
```

On the isolated native root, run the real gate with `MTD_ENCODER_GATE_OPT=196656`
and `MTD_PROFILE=1`. Run `moss_cpu_regression.py` with
`--candidate-reference 48 --pair-reference 65584 --pair-candidate 196656`
and `--variants baseline cache-reference 48 65584 196656`, all six cases,
`--reference-opt 48 --repeats 2 --pin-physical --profile --trace-tokens`.
The private controller pauses automatic processing until idle and restores its
previous setting and authenticated API identity in `finally`. Concurrent MOSS
inference is sampled every0.5seconds by the full-input harness; the standalone
probe watcher samples once persecond. Other host jobs remain running.

## Native operator and encoder results

GCC13.3 compiled the actual VNNI body. All17 non-model CTests passed in25.27s,
and all40 Python regression/audit tests passed. The12 prior single-projection
and4 grouped records preserve every float bit through two changed-input passes;
all four actual-worker failure and recovery cases passed. All14 real-weight
chunks match21,504,000 encoder output elements, with1,344 actual callbacks,
2,016 logical projection executions and336 grouped QKV executions at16 workers.

The protected normal-input benchmark retained these six-sample medians.
The single-path control is in the same compiled binary, using the generalized
callback with one projection per context; it is not an unchanged PR17 binary:

| Variant | Median µs | Retained minimum–maximum µs |
|---|---:|---:|
| Ordinary F32, three projections |9319.053500|9217.488000–16506.439000|
| Same-build single VNNI, three callbacks |9636.909500|6907.586000–12864.330000|
| Shared QKV VNNI, one callback |9441.337000|6436.449000–12448.534000|

Grouped/single median gain is1.0207×, while grouped is1.31% slower than ordinary
F32 by these medians. The distributions are bimodal and overlap. This small,
unpinned standalone result does not establish a performance win or confidence;
all samples remain in the report, including losses. No full-input improvement
follows from this operator result.

## Complete full-input outcome

All60 runs passed the11 full-input gates. Every per-case full-output hash,
token count, EOS and return code matches; all36 same-build complete token
sequence hashes match, including unique terminal EOS. All12 grouped candidate
runs executed the expected192 callbacks/288consumers/48QKV nodes for60s,
or384/576/96 for120s, with16 actual workers and zero failures. Silence has
13tokens/EOS/return0; genuine empty speech has1token/EOS/return1/empty output.
Artifacts, model and source remained unchanged throughout the trial.

| Input | Same-build48 s | Separate65584 s | Grouped196656 s | Separate / grouped | Frozen48 / grouped |
|---|---:|---:|---:|---:|---:|
|meeting-60s|13.819189|13.517730|13.467672|1.003717×|1.024137×|
|dinner-60s|11.738755|11.338016|11.363222|0.997782×|1.030791×|
|vox-vmaiq-60s|12.064386|11.688997|11.688767|1.000020×|1.029983×|
|meeting-120s|30.176232|29.449490|29.374862|1.002541×|1.026414×|
|silence-60s|7.505712|7.130125|7.129956|1.000024×|1.052671×|
|empty-real-60s|7.440023|7.004960|7.005203|0.999965×|1.058052×|

The incremental grouped-versus-separate point estimates range from0.22% slower
on German speech to0.37% faster on English60s; other inputs are effectively
unchanged or0.25% faster on120s. This provides no convincing additional full-input
speed gain. The larger2.61–3.30% same-build48 speedups for speech include the
already implemented selective encoder VNNI path. They must not be attributed
to this grouping step. All declared5% regression thresholds pass; two repeats,
shared-host load12.89–16.02 and overlapping operator samples do not establish
statistical confidence. All raw numeric timings/profiles/RSS remain in the report.

The standalone watcher observed59 active one-second snapshots, zero concurrent
MOSS, with one-minute load8.61–12.90; shorter overlaps are not excluded.
The full-input harness observed zero additional MOSS processes at its0.5s
sample interval. Other host jobs were left running. Controller exited0, the
exact owned watcher was stopped afterward, automatic processing restored1,
and authenticated production API identity was unchanged and healthy.

The complete numeric artifact is [`encoder-qkv-native-v1.json`](encoder-qkv-native-v1.json),
SHA256 `9eda2625658b64036e8dc9198584e42709181d44d24edc5402f0b7fa487f536e`. It retains all source/compiler/artifact/log hashes,
synthetic/failure/recovery/realencoder records, all six micro samples pervariant,
all60 full-input records, all route/latency/token gates and terminal restoration
metadata. Private text, audio, weights, raw token IDs and credentials are excluded.
The route remains opt-in research, production is unchanged, and the original
quietopt48 full-input10× objective remains active and unachieved.
