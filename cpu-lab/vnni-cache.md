# Exact VNNI cache traversal

Input-tile-first traversal reuses each packed activation tile across the same
worker's weight subset. It retains the prior weight-tile-first inline kernel
and all four earlier controls. Original Q8 weights/scales, panel construction,
wrapped -128 negation, eight FMA chains and final reduction order are unchanged.
No MOSS model routing or production change is added. All three encoder-shaped
matrices improve in this prequantized operator experiment; complete-model gain
remains unmeasured.

## Frozen source and native validation

Compiled local `abf51f8c2cb5b1f1518726f4313225edae7bcc74` equals public
`332d201e371dbd707a9927fb2f9ecbcf4a50c428`, tree
`7b55ae0752e7582be5a417060105a55f6b509d20`. The clean separate native source
pins ggml `eced84c86f8b012c752c016f7fe789adea168e1e`. GCC13.3 on Xeon Gold
5416S compiles the actual AVX512 VNNI/F16C/OpenMP body. Fifteen CTests pass
in22.65 seconds, and34 Python checks pass. Unsupported ISA or incompatible
pinned dot traits returns77 without portable fallback timing.

Both invocations together pass2,097,152 exhaustive half conversions under all
sixteen rounding/FTZ/DAZ modes, restoring original MXCSR. All78 arithmetic and
111 protected benchmark shape records are exact, including30 normal timed and
81 arithmetic-only edge controls. Raw source, weights, activations and graph
operands remain immutable after two input updates and timing. All initialized
panels, inactive columns and ABI padding are checked. New output bits equal
pinned dot and ordinary prequantized GGML. Runtime ownership arrays confirm
every weight tile has the identical worker in old and new traversals, including
uneven small partitions with idle workers. The unsigned panel-difference count
aggregates all three parallel variants; zero requires each variant to match.

Build Release with MT_BUILD_TESTS=ON; run `build/tests/moss_vnni_q8_cache`, then
the same command with `--benchmark`. The target is outside default CTest. All
cases, six retained samples, source/compiler/library/artifact/log hashes and
private-controller/watcher checksums are in
[`vnni-cache-native-v1.json`](vnni-cache-native-v1.json).

## Paired timing boundary

Six variants rotate/reverse over seven rounds, discarding first. Each reported
median averages the central two of six retained sorted samples. Primary total
includes allocation, full packing/initialization, computation, scatter and
cleanup. It excludes F32-to-Q8 conversion, graph callback integration, model
loading, audio, attention and full generation. Separate seven-round diagnostic
passes cover four candidates; their timestamp barrier is absent from primary
totals. Packing includes team creation and execution includes join. Stage
medians are not additive or the same samples as total medians.

| Workers | K / N / M | Ordinary GGML µs | Prior inline µs | Cache traversal µs | GGML / cache | Inline / cache |
|---:|---|---:|---:|---:|---:|---:|
| 1 | 1024 / 1024 / 64 | 1298.405 | 945.952 | 932.439 | 1.3925× | 1.0145× |
| 16 | 1024 / 128 / 16 | 18.285 | 17.040 | 15.282 | 1.1965× | 1.1150× |
| 16 | 3072 / 128 / 16 | 32.484 | 29.739 | 29.897 | 1.0865× | 0.9947× |
| 16 | 1024 / 1024 / 64 | 137.595 | 114.722 | 113.040 | 1.2172× | 1.0149× |
| 16 | 1024 / 4096 / 64 | 680.922 | 438.873 | 433.419 | 1.5710× | 1.0126× |
| 16 | 1024 / 1024 / 1500 | 2907.965 | 2479.910 | 2232.770 | 1.3024× | 1.1107× |
| 16 | 1024 / 4096 / 1500 | 11441.108 | 9296.800 | 8467.055 | 1.3512× | 1.0980× |
| 16 | 4096 / 1024 / 1500 | 11148.786 | 13615.330 | 8460.782 | 1.3177× | 1.6092× |

Cache traversal is faster in23 of30 paired cases versus ordinary GGML and
21 versus prior inline traversal; seven and nine respectively are slower.
The encoder contraction now takes8460.782µs versus11148.786µs ordinary GGML
and13615.330µs prior inline. The largest cache sample9687.764µs is below the
smallest ordinary sample11041.166µs in that shape. This is one unpinned shared-host
experiment, not an isolated scheduling-causality proof or full-input speedup.
Retained samples expose variation: the ordinary contraction has one15082.807µs
sample. Operator medians are neither summed into model latency nor compared
against the immutable quiet opt48 full-input tenfold baseline.

The controller exits0, restores automatic processing to1, and verifies
authenticated API health and unchanged production identity. The owned watcher
is stopped afterward. Twelve one-second probe-active snapshots observe zero
concurrent MOSS processes, with one-minute shared load7.59–8.27. Shorter overlaps
cannot be excluded; other jobs stay active. Source/evidence is MIT with
dependency licenses retained; private audio, text, raw tokens, weights and
credentials are absent.

## Next gate

Add a separate F32-input GGML custom-op experiment and include quantization,
callback/team overhead and allocation/cleanup in its paired totals. First
compare output bits with ordinary F32-input GGML, whose selected route may
differ from the prequantized control. Only a viable exact result proceeds to
actual layers, cache updates, complete token/output/EOS checks and matched
full-input timings. The original full-input10× goal remains active and unachieved.
