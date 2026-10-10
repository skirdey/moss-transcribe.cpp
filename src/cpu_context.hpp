// MIT. Research-only causal context for transposed F32 CPU cache views.
#pragma once
#include "ggml.h"
#include <cstdint>

namespace mt {
// Requires one batch, multiple queries, a nonempty prefix, and causal
// probabilities. Returns nullptr for unsupported layouts. No borrowed userdata.
ggml_tensor* cpu_causal_context(ggml_context* ctx, ggml_tensor* values,
                               ggml_tensor* probabilities);
struct CpuContextCounts { uint64_t built, executed; };
// Process-local cumulative diagnostics; the model probe checks per-append deltas.
CpuContextCounts cpu_causal_context_counts();
}
