# PocketFFT

Header and BSD-3-Clause license vendored unchanged from
https://github.com/mreineck/pocketfft at commit
`c90e55b3d529f8efa40ed01a20de22405f45fc65` (branch `cpp`).
See LICENSE.md and the header for copyrights and license terms.

Used only by the opt-in CPU mel FFT experiment. Each ggml worker transforms
one frame at a time using the unnormalized forward real transform in double
precision. PocketFFT threading is disabled; ggml owns the worker team.
A 16-plan cache avoids repeatedly creating the same 400-point plan.
This computes different floating-point values from the legacy DFT with its
FP32 twiddle tables; it is not a bit-exact substitute.
