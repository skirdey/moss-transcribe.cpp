// MIT. Decode-only projection fusion research; ordinary weights and pinned dots.
#pragma once
#include "ggml.h"
#include <cstddef>
namespace mt {
// Returns a concatenated [sum(output widths),1] F32 tensor or nullptr when the
// exact M=1 CPU Q8 path is unavailable. Caller selects views of each projection.
ggml_tensor* cpu_decode_q8_projections(ggml_context* ctx,ggml_tensor* input,
    ggml_tensor* const* weights,size_t count);
}
