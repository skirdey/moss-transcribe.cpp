#include "cpu_activation.hpp"
#include "backend.hpp"
#include "cpu_profile.hpp"
#include "ggml-cpu.h"

#include <algorithm>
#include <cstdint>

namespace mt {
namespace {
void convert(ggml_tensor* dst, int ith, int nth, void*) {
    const auto* x = dst->src[0];
    const int64_t block_size = ggml_blck_size(GGML_TYPE_Q8_0);
    const int64_t blocks = x->ne[0] / block_size;
    const int64_t total = ggml_nelements(x) / block_size;
    // Partition independent blocks across the existing graph workers, including
    // M=1. Unlike ggml_cast, no intermediate F32 row copy or per-worker scratch.
    int64_t cursor = total * ith / nth;
    const int64_t end = total * (ith + 1) / nth;
    const auto from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    while (cursor < end) {
        const int64_t row = cursor / blocks, first = cursor % blocks;
        const int64_t count = std::min(end - cursor, blocks - first);
        const int64_t i1 = row % x->ne[1];
        const int64_t i2 = (row / x->ne[1]) % x->ne[2];
        const int64_t i3 = row / (x->ne[1] * x->ne[2]);
        const auto* src = reinterpret_cast<const float*>(
            static_cast<const char*>(x->data) + i1*x->nb[1] + i2*x->nb[2] + i3*x->nb[3]);
        auto* out = static_cast<char*>(dst->data) + row*dst->nb[1];
        from_float(src + first*block_size, out + first*ggml_type_size(GGML_TYPE_Q8_0),
                   count*block_size);
        cursor += count;
    }
}
}

ggml_tensor* cpu_shared_q8_activation(ggml_context* ctx, ggml_tensor* x) {
    if (!x || x->type != GGML_TYPE_F32 || x->nb[0] != sizeof(float)
        || x->ne[0] <= 0 || x->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0)) return nullptr;
    ggml_tensor* args[] = {x};
    return ggml_custom_4d(ctx, GGML_TYPE_Q8_0, x->ne[0], x->ne[1], x->ne[2], x->ne[3],
                          args, 1, convert, GGML_N_TASKS_MAX, nullptr);
}

ggml_tensor* cpu_shared_projection_input(ggml_context* ctx, ggml_tensor* x,
    ggml_tensor* const* weights, size_t count, CpuSharedActivationMode mode) {
    if (!x || !weights || count < 2 || x->type != GGML_TYPE_F32
        || x->nb[0] != sizeof(float) || x->ne[0] <= 0
        || x->ne[0] % ggml_blck_size(GGML_TYPE_Q8_0)
        || !ggml_backend_is_cpu(backend())) return x;
    for (size_t i=0; i<count; ++i) {
        const auto* w=weights[i];
        if (!w || w->type != GGML_TYPE_Q8_0 || w->ne[0] != x->ne[0]
            || !ggml_is_contiguous(w) || w->ne[2] != 1 || w->ne[3] != 1) return x;
    }
    auto q=mode==CpuSharedActivationMode::Cast ? ggml_cast(ctx,x,GGML_TYPE_Q8_0)
        : cpu_shared_q8_activation(ctx,x);
    if (!q) return x;
    cpu_profile_record_shared_q8(mode==CpuSharedActivationMode::Cast, count);
    return q;
}
}
