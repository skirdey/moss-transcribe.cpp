#pragma once
#include "ggml.h"

namespace mt {
// CPU single-query softmax with independent heads assigned across workers.
// q provides the head dimension; no borrowed userdata or mutable scale state.
ggml_tensor* cpu_decode_softmax(ggml_context* ctx, ggml_tensor* scores, ggml_tensor* q);
}
