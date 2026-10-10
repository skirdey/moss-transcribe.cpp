#ifndef MT_CPU_ACTIVATION_HPP
#define MT_CPU_ACTIVATION_HPP

#include "ggml.h"

namespace mt {
// Research CPU operator: convert independent Q8_0 blocks once for several
// ordinary ggml_mul_mat consumers. No packed-weight copy or new dot arithmetic.
// Returns nullptr for incompatible inputs. The caller must select a CPU backend.
ggml_tensor* cpu_shared_q8_activation(ggml_context* ctx, ggml_tensor* input);
}
#endif
