#ifndef MT_CPU_ACTIVATION_HPP
#define MT_CPU_ACTIVATION_HPP

#include "ggml.h"
#include <cstddef>

namespace mt {
// Research CPU operator: convert independent Q8_0 blocks once for several
// ordinary ggml_mul_mat consumers. No packed-weight copy or new dot arithmetic.
// Returns nullptr for incompatible inputs. The caller must select a CPU backend.
ggml_tensor* cpu_shared_q8_activation(ggml_context* ctx, ggml_tensor* input);
enum class CpuSharedActivationMode { Blocks, Cast };
// Explicit research routing. Reuse only when every consumer is a dense Q8_0
// weight with the same input width on the selected CPU backend. Otherwise keep
// the original F32 input; F16/F32 weights must never acquire Q8 activations.
ggml_tensor* cpu_shared_projection_input(ggml_context* ctx, ggml_tensor* input,
    ggml_tensor* const* weights, size_t count, CpuSharedActivationMode mode);
}
#endif
