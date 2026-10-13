# MOSS CPU speed comparison

**CPU optimizations: 1.58× overall speedup (37% less wall time), 1.71× on recorded speech.**

The larger test uses **24 clips, 17.9 minutes of audio, three repeats and 288 measured runs**. It compares the original CPU build at opt0 with the optimized build at opt48, using the same Q8_0 model and 16 physical CPU cores. Each timing includes model loading and the complete answer.

| Scope | Original CPU | Optimized CPU | Speedup |
|---|---:|---:|---:|
|All 24 clips|403.80s|254.98s|**1.58×**|
|14 recorded speech clips|357.30s|208.73s|**1.71×**|
|8 short generated speech clips|31.64s|31.39s|**1.01×**|
|2 silence / empty controls|14.86s|14.86s|**1.00×**|

## Every clip stays in the comparison

| Clip | Duration | Original opt0 | Optimized opt48 | Speedup | Same output |
|---|---:|---:|---:|---:|---|
|Speech A, 60s|60.00s|21.91s|13.84s|1.58×|yes|
|Speech B, 60s|60.00s|16.65s|11.74s|1.42×|yes|
|Public speech A, 60s|60.00s|17.40s|11.99s|1.45×|yes|
|Speech A, 120s|120.00s|67.09s|29.98s|2.24×|yes|
|Silence, 60s|60.00s|7.53s|7.48s|1.01×|yes|
|Empty real audio, 60s|60.00s|7.33s|7.38s|0.99×|yes|
|Public speech B, 60s|60.00s|19.55s|12.94s|1.51×|yes|
|Public speech C, 60s|60.00s|19.60s|12.89s|1.52×|yes|
|Recording A, 20% offset, 30s|30.00s|8.23s|6.53s|1.26×|yes|
|Recording A, 50% offset, 60s|60.00s|21.81s|13.79s|1.58×|yes|
|Recording A, 80% offset, 120s|120.00s|60.58s|28.47s|2.13×|yes|
|Recording B, 20% offset, 30s|30.00s|8.03s|6.43s|1.25×|yes|
|Recording B, 50% offset, 60s|60.00s|23.36s|14.39s|1.62×|yes|
|Recording B, 80% offset, 120s|120.00s|44.65s|24.26s|1.84×|yes|
|Room mix, 33.82s|33.82s|13.29s|10.34s|1.29×|yes|
|Same room turns in sequence, 43.88s|43.88s|15.14s|11.14s|1.36×|yes|
|Generated German|4.41s|3.97s|3.92s|1.01×|yes|
|Generated English|4.56s|3.92s|3.87s|1.01×|yes|
|Generated Spanish|6.24s|3.97s|3.92s|1.01×|yes|
|Generated French|4.92s|3.92s|3.92s|1.00×|yes|
|Generated Hindi|4.82s|3.92s|3.92s|1.00×|yes|
|Generated Italian|6.05s|4.02s|3.97s|1.01×|yes|
|Generated Russian|5.08s|4.02s|3.97s|1.01×|yes|
|Generated Chinese|2.45s|3.87s|3.87s|1.00×|yes|

Two cases are slightly slower than the original: empty real audio by 0.69%, and the short Hindi sample by 0.01%. Every clip passes the predeclared five-percent slowdown limit. Short generated clips and empty audio have little decoding work, so they benefit little from the cache and softmax improvements.

## How the overall number is calculated

For each clip and setting, take the middle time of three runs. Add those times for all clips, then divide the original total by the optimized total. No clip is removed, and individual speedup ratios are not averaged. The overall totals are 403.80s and 254.98s, giving 1.5836558× and 36.854965% less wall time.

The earlier six-clip pilot gave 1.69× overall and 1.84× on its four speech clips. The expanded set contains more short clips and controls with little gain. The same optimized build is used in both tests. On the expanded set it is within 0.37% of the previous optimized reference overall, and every clip remains within the fixed five-percent limit.

## What the checks establish

All 288 runs reach EOS. Complete output, including words, speaker labels and timestamps, is byte-identical across the original, fixed optimized reference, current opt0 and current opt48 settings. The 144 current-build runs also match complete token-ID hashes, including EOS. The older binaries expose full-output, token-count and EOS checks rather than complete token traces.

Before/after checks confirm the model, binaries, libraries and runtime source are unchanged. Half-second process checks detect no overlapping MOSS job. Automatic ingestion is restored after the test and the authenticated API remains healthy.

The general runner exits 1 because it also compares the deliberately unoptimized opt0 control with the faster reference. Those 14 expected slow-control failures stay in the numeric report. The optimized opt48 setting passes all three separate comparisons: original, fixed optimized reference and same-build opt0. The source and token checks pass.

This test measures whether the optimization keeps the existing answers. Absolute word or diarization accuracy is not scored here. Some excerpts share a recording, the two room arrangements share turns, and existing clips may overlap the source recordings. Eight clips are generated speech. Results describe this selected set of distinct inputs; three repeats on a shared system provide no confidence interval.

## Method and reproducibility

- Intel Xeon Gold 5416S, CPU backend, 16 workers bound to physical cores 0–15.
- Sequential fresh processes with warm OS file cache; all model loading and inference costs included.
- Four settings per clip: original0, fixed reference48, current0 and current48. Four complete warmups are discarded. Setting order reverses in the second round.
- Excerpts use fixed 20%, 50% and 80% offsets with 30s, 60s and 120s lengths, chosen before observing results.
- Runtime source 35036546f5aade49479728369f6cdfe56b3855fe; public exact-tree counterpart [8382623](https://github.com/skirdey/moss-transcribe.cpp/commit/8382623c2014f85881f0680a0d8f10ce414b20b0).
- GGML eced84c86f8b012c752c016f7fe789adea168e1e.
- Model SHA256 ed6c35d0d527c5d03171c3eb448e2150a42c76a51e3e73aa821e351c3da8307c.
- [Predeclared clip list and settings](cpu-larger-corpus-protocol-v1.json), [all numeric measurements and failures](cpu-larger-corpus-v1.json), and [unrounded comparison](cpu-speedup-summary-v2.json).
- [Earlier six-clip summary](cpu-speedup-summary-v1.json) remains available for comparison.

Personal machine names, directory paths, network details, full system inventory and unrelated machine activity are omitted from these new public files. All timings, counts, hashes and failed checks needed to assess the comparison remain. Audio, transcript text, token IDs and model values are kept separately.

Recreate the summary from the published numeric files:

```sh
python3 cpu-lab/moss_speedup_summary.py \
  --report cpu-lab/cpu-larger-corpus-v1.json \
  --protocol cpu-lab/cpu-larger-corpus-protocol-v1.json \
  --output comparison.json
```

The collector checks complete coverage, unique runs, input/source/binary identities, stored gates and exit status. Missing runs, duplicate runs and changed inputs are rejected. Slow, incomplete or different answers remain visible. Its 12 checks pass alongside 63 existing Python checks.

The tested CPU profile is:

```sh
MTD_DEVICE=cpu MTD_THREADS=16 MTD_CPU_OPT=48 ./build/moss-transcribe transcribe MODEL AUDIO --max-new 4096
```

## Deployment and next work

The live API still uses the original production build. These gains are measured in the tested optimized CLI and are available in the public source. The 10× full-input target is measured against the previous optimized reference, and remains unachieved.

[The failed guessing experiment](strict-speculative.md) remains disabled. Checking several guesses at once was 10–41% slower on speech and changed an answer. [The next diagnostic](actual-audio-attention.md) identifies the first rounding difference; its repair and timing checks remain pending.

Code, scripts and numeric comparison files are MIT licensed.

Published numeric report SHA256: `4d11deccfbc6085f7cee1d8161864bd819b6f8087c5b4cc6bd5c9b3507bc55cb`.
