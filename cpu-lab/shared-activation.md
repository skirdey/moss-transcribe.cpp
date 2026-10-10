# Shared Q8 activation conversion

This experiment isolates conversion reuse while retaining ordinary Q8 weights
and `ggml_mul_mat`. It adds an explicit CPU research operator, with no model
routing or production deployment. No end-to-end improvement is established.

The pinned CPU Q8 matmul converts each F32 activation row to Q8_0 before its
dot/GEMM path. Q/K/V or gate/up consumers therefore repeat conversion of the
same input. The candidate shares one Q8 tensor, computed with that exact CPU
`from_float` function. It partitions independent blocks among existing graph
workers, including single-row decode. Compared with `ggml_cast`, it avoids the
intermediate F32 row copy and per-worker scratch. It introduces no additional
activation precision loss relative to the existing Q8 matmul. The reference
already quantizes those activations; this is not lossless F32-to-Q8 conversion.

Compatible inputs are F32 with a contiguous scalar dimension whose length is
divisible by 32; outer row strides may differ. Other inputs return nullptr.
The caller must select the CPU backend. Two higher dimensions are also covered
by the batch conversion test. No weight repacking, custom reduction or AMX
buffer is used. Each consumer remains the ordinary reference matrix operation.

## Native fixture failure and repeated-input repair

The first fresh build passed ten checks but failed the new probe's finite-output
gate. Its small tail cases passed; the exception stopped the suite before any
graph benchmark ran. A second frozen diagnostic build identified all three
variants producing NaN on the second input update (K=N=1024, M=1, one worker).

The fixture allocated synthetic inputs with the graph allocator and only set
`GGML_TENSOR_FLAG_INPUT`. In pinned `ggml-alloc.c`, that flag allocates inputs
early, while a leaf's storage may be freed/reused after its last consumer.
Repeated subgraphs therefore reused overwritten operands. Real model weights
have separate loader-owned storage. The repaired fixture retains its synthetic
operands and byte-checks every weight/input after each update and after timing.
The converter itself was unchanged. This separates buffer-lifetime failure
from an arithmetic difference.

Public historic tree `f261a6ae1c14f1a8465dc36e5a8a1a5e560dc5dc` is exactly the
first failed source. `activation-finite-diagnostic-v2.patch` reproduces the
second diagnostic source; `activation-input-lifetime-repair-v3.patch` reproduces
the passing conversion probe. The latter has exactly the public tree at
`129f28865896c020535ec13cdb08b34660ac944b`. Original roots/logs remain frozen;
failed attempts are not passing validation.

The older `test_cpu_q8` graph microbenchmark also lacked explicit retention
and post-timing operand checks. Its first audit checked only the activation
input and detected no mutation, so that audit's expected-failure check failed.
The complete audit checked weights too: **the original graph mutated synthetic
weight storage after its first computation**, while the activation input stayed
unchanged (K=32, N=17, one worker). No later temporary or timing was added.
The reference consumes the ordinary weight tensor first; the custom path uses
a separately owned packed vector, permitting reclamation of ordinary storage.

The earlier warm graph ratios are withdrawn: an initial output comparison did
not validate the reference operands in repeated timed graphs. Initial graph
parity, the separately owned-vector dot probe and fresh-process full-model
opt816 pilot remain distinct evidence. Real loader weights live outside the
graph allocator. The guarded graph test retains operands, checks them after
timing and recomputes all variants. `--audit-unretained-input` restores the
original flags and detects mutation of either weights or activations before
timing. The failed input-only audit remains recorded. An intermediate design
that added a later temporary was prepared but never compiled; its exact source
delta is retained in `activation-audit-fixture-v5-unused.patch`.

## Completed conversion-only graph result

All **72/72** synthetic cases matched activation bytes and raw output float
bits for both shared `ggml_cast` and shared block conversion. All finite,
operand-immutability, artifact and no-concurrent-MOSS gates passed. Each case
checks two input updates. Shapes include tail K=32/N=17, actual Qwen Q/K/V
widths 2048/1024/1024, gate/up width 3072, strided rows, single-row decode,
64/128-row prefill and the actual 1024-wide encoder's 1500-row input. The
1280-wide shape is additional synthetic coverage, not this checkpoint's width.
Normal, zero and signed-byte-extreme weights/inputs use 1/8/16 workers.

Only normal cases are timed, using the second input update. Five cyclic-order
samples follow one discarded round; each times complete projection-group graph
computation, including conversion and dispatch. Weights and graph allocation
are outside timing. Each variant reads the same immutable operands and uses
ordinary matrix math. The process is pinned to one worker per physical core;
default wait policy is retained. Other users' workloads remain on hp-fury.
One-minute load changed from 17.33 to 15.67 during the 15.18-second probe.

The 16-thread medians below are microseconds per complete group:

| Group | Eager | Shared cast | Shared blocks | Eager / blocks |
| --- | ---: | ---: | ---: | ---: |
| Qwen decode, 1 row, Q/K/V 2048/1024/1024 | 57.958 | 52.497 | 52.263 | 1.1090 |
| Gate/up, 3 strided rows, N=3072 | 66.962 | 61.750 | 67.654 | 0.9898 |
| Qwen prefill, 64 rows, Q/K/V 2048/1024/1024 | 762.864 | 734.865 | 734.600 | 1.0385 |
| Gate/up prefill, 128 rows, N=3072 | 2180.344 | 2166.755 | 2159.329 | 1.0097 |
| Encoder Q/K/V, 1500 rows, N=1024 | 18166.320 | 15123.423 | 18146.956 | 1.0011 |

Shared cast was about 1.20x on the final encoder case, while block conversion
was effectively neutral. Some other thread/shape combinations lost performance.
All samples and controls are retained in `activation-graph-v3.json`; isolated
best cases cannot establish a whole-model gain or select universal dispatch.
No model load, audio frontend, attention, generation, corpus WER/DER or
single-input latency appears in this synthetic benchmark.

The final guarded build passed all **12 native CTests in 24.77 seconds** and
all **28 Python checks** on hp-fury. Its exact public compiled tree is
`6a341de73dc776dca67752b5bc8a82dbf0bd3b64`; the numerical conversion benchmark
used the earlier, separately frozen passing tree above. Native attempts,
source hashes, log hashes and both audit outcomes are preserved in
`activation-validation-v6.json`. Processing was restored and authenticated
API health was OK. Default model routing and the production binary are unchanged.

## Reproduction and next gate

Use the exact passing conversion source with pinned ggml:

```sh
git clone --recursive https://github.com/skirdey/moss-transcribe.cpp.git source
git -C source checkout 129f28865896c020535ec13cdb08b34660ac944b
git -C source submodule update --init
cmake -S source -B build -DMT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 16
ctest --test-dir build -LE model --output-on-failure
python3 source/cpu-lab/moss_activation_bench.py --root /path/to/root \
  --name fresh-activation-probe
```

Keep `source/` and `build/` under that root. The runner requires 16 allowed
physical cores and no other MOSS inference; it hashes code/binary/libraries
before and after the bounded probe. Pause automatic processing, wait for API
idle and restore processing in a `finally` block. Do not overwrite old results.
Current main also includes the old graph test's input-retention repair/audit;
that later change does not alter the conversion operator or measured source.

The next experiment can compare a shape-aware choice of shared cast versus
shared blocks in the actual model, keeping ordinary weights and reference
matrix operations. Require identical full raw text/speaker/timestamps, token
counts, EOS/empty behavior and all three matched latency gates: production,
frozen fastest opt48 and same-build opt48. Conversion speed alone cannot
establish correctness through downstream attention or a tenfold result.

All own source and evidence remain MIT. Private audio/text/token IDs, weights
and credentials are excluded. The 10x objective remains active and unachieved.
