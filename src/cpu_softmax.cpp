#include "cpu_softmax.hpp"
#include "ggml-cpu/vec.h" // Exact primitives from the pinned ggml CPU backend.

#include <cmath>
#include <vector>

namespace mt {
namespace {
void parallel_decode_softmax(ggml_tensor* dst, const ggml_tensor* scores,
                             const ggml_tensor* q, int ith, int nth, void*) {
    GGML_ASSERT(scores->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scores->ne[1] == 1 && q->ne[1] == 1);
    GGML_ASSERT(scores->nb[0] == sizeof(float) && ggml_is_contiguous(dst));
    const int n = static_cast<int>(scores->ne[0]);
    const float scale = 1.0f / std::sqrt(static_cast<float>(q->ne[0]));
    // Reuse per-worker scratch, avoiding per-layer allocations after growth.
    thread_local std::vector<float> scratch;
    scratch.resize(n);
    const int64_t rows = scores->ne[2] * scores->ne[3];
    for (int64_t row = ith; row < rows; row += nth) {
        const int64_t head = row % scores->ne[2];
        const int64_t batch = row / scores->ne[2];
        const float* src = reinterpret_cast<const float*>(
            static_cast<const char*>(scores->data) + head * scores->nb[2] + batch * scores->nb[3]);
        float* out = reinterpret_cast<float*>(
            static_cast<char*>(dst->data) + head * dst->nb[2] + batch * dst->nb[3]);
        // Preserve the eager reference sequence, including its double sum and
        // float conversion for the reciprocal. Only independent rows move.
        ggml_vec_cpy_f32(n, scratch.data(), src);
        ggml_vec_scale_f32(n, scratch.data(), scale);
        float max = -INFINITY;
        ggml_vec_max_f32(n, &max, scratch.data());
        ggml_float sum = ggml_vec_soft_max_f32(n, out, scratch.data(), max);
        GGML_ASSERT(sum > 0.0);
        sum = 1.0 / sum;
        ggml_vec_scale_f32(n, out, sum);
    }
}
}

ggml_tensor* cpu_decode_softmax(ggml_context* ctx, ggml_tensor* scores, ggml_tensor* q) {
    return ggml_map_custom2(ctx, scores, q, parallel_decode_softmax, GGML_N_TASKS_MAX, nullptr);
}
}
