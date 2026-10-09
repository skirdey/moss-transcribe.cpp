#include "cpu_q8.hpp"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <climits>
#include <cstring>

#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__F16C__) && !defined(__AVXVNNIINT8__)
#include <immintrin.h>
#define MT_EXACT_Q8_NATIVE 1
#else
#define MT_EXACT_Q8_NATIVE 0
#endif

namespace mt {
static_assert(sizeof(Q8Tile) == 16 * sizeof(block_q8_0), "full tiles add no bytes");

bool cpu_exact_q8_supported() {
#if MT_EXACT_Q8_NATIVE
    ggml_cpu_init();
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    return ggml_cpu_has_avx512_vnni() && ggml_cpu_has_f16c()
        && traits->vec_dot_type == GGML_TYPE_Q8_0 && traits->nrows == 1;
#else
    return false;
#endif
}

std::unique_ptr<PackedQ8> cpu_pack_q8(const ggml_tensor* w, const void* ordinary) {
    if (!cpu_exact_q8_supported() || !ordinary || !w || w->type != GGML_TYPE_Q8_0
        || !ggml_is_contiguous(w) || w->ne[2] != 1 || w->ne[3] != 1
        || w->ne[0] <= 0 || w->ne[0] > INT_MAX || w->ne[0] % 32
        || w->ne[1] <= 0 || w->ne[1] > INT_MAX - 15) return nullptr;
    auto p = std::make_unique<PackedQ8>();
    p->k = static_cast<int>(w->ne[0]); p->n = static_cast<int>(w->ne[1]);
    const int blocks = p->k / 32, tiles = (p->n + 15) / 16;
    p->tiles.resize(static_cast<size_t>(tiles) * blocks);
    const auto* source = static_cast<const block_q8_0*>(ordinary);
    for (int tile = 0; tile < tiles; ++tile) for (int b = 0; b < blocks; ++b) {
        auto& dst = p->tiles[static_cast<size_t>(tile) * blocks + b];
        for (int lane = 0; lane < 16 && tile * 16 + lane < p->n; ++lane) {
            const auto& src = source[static_cast<size_t>(tile * 16 + lane) * blocks + b];
            dst.scales[lane] = src.d;
            for (int g = 0; g < 8; ++g) std::memcpy(dst.groups[g] + lane * 4, src.qs + g * 4, 4);
        }
    }
    return p;
}

bool PackedQ8::round_trip_exact(const void* ordinary) const {
    const auto* source = static_cast<const block_q8_0*>(ordinary);
    if (!source) return false;
    const int blocks = k / 32;
    for (int row = 0; row < n; ++row) for (int b = 0; b < blocks; ++b) {
        const auto& tile = tiles[static_cast<size_t>(row / 16) * blocks + b];
        block_q8_0 restored{};
        restored.d = tile.scales[row % 16];
        for (int g = 0; g < 8; ++g) std::memcpy(restored.qs + g * 4, tile.groups[g] + row % 16 * 4, 4);
        if (std::memcmp(&restored, source + static_cast<size_t>(row) * blocks + b, sizeof(restored))) return false;
    }
    return true;
}

namespace {
bool eligible(const ggml_tensor* x, const PackedQ8* p) {
    return p && x && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x)
        && x->ne[0] == p->k && x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1;
}

#if MT_EXACT_Q8_NATIVE
void dot(const Q8Tile* p, const block_q8_0* x, int blocks, float* y, int lanes) {
    __m512 acc[8];
    for (auto& a : acc) a = _mm512_setzero_ps();
    const auto zero = _mm512_setzero_si512();
    for (int b = 0; b < blocks; ++b) {
        const auto scale = _mm512_mul_ps(
            _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p[b].scales))),
            _mm512_set1_ps(ggml_fp16_to_fp32(x[b].d)));
        for (int g = 0; g < 8; ++g) {
            int32_t four; std::memcpy(&four, x[b].qs + g * 4, 4);
            const auto q = _mm512_loadu_si512(p[b].groups[g]);
            const auto input = _mm512_set1_epi32(four);
            const auto negative = _mm512_cmp_epi8_mask(q, zero, _MM_CMPINT_LT);
            const auto signed_input = _mm512_mask_sub_epi8(input, negative, zero, input);
            const auto sum = _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(q), signed_input);
            acc[g] = _mm512_fmadd_ps(scale, _mm512_cvtepi32_ps(sum), acc[g]);
        }
    }
    // Preserve all eight eager accumulator chains and its horizontal add tree.
    const auto even = _mm512_add_ps(_mm512_add_ps(acc[4], acc[0]), _mm512_add_ps(acc[6], acc[2]));
    const auto odd = _mm512_add_ps(_mm512_add_ps(acc[5], acc[1]), _mm512_add_ps(acc[7], acc[3]));
    _mm512_mask_storeu_ps(y, static_cast<__mmask16>((1u << lanes) - 1), _mm512_add_ps(even, odd));
}

void compute(ggml_tensor* dst, const ggml_tensor*, const ggml_tensor* x, int ith, int nth, void* userdata) {
    const auto& p = *static_cast<const PackedQ8*>(userdata);
    GGML_ASSERT(x->type == GGML_TYPE_Q8_0 && ggml_is_contiguous(x));
    const int blocks = p.k / 32, tiles = (p.n + 15) / 16;
    for (int tile = (static_cast<int64_t>(tiles) * ith) / nth;
         tile < (static_cast<int64_t>(tiles) * (ith + 1)) / nth; ++tile)
        dot(p.tiles.data() + static_cast<size_t>(tile) * blocks,
            static_cast<const block_q8_0*>(x->data), blocks,
            static_cast<float*>(dst->data) + tile * 16, std::min(16, p.n - tile * 16));
}
#endif
}

ggml_tensor* cpu_decode_quantize(ggml_context* ctx, ggml_tensor* x, const PackedQ8* p) {
    return eligible(x, p) ? ggml_cast(ctx, x, GGML_TYPE_Q8_0) : nullptr;
}

ggml_tensor* cpu_decode_mul_mat(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x,
                               const PackedQ8* p, ggml_tensor* quantized) {
#if MT_EXACT_Q8_NATIVE
    if (eligible(x, p) && w->type == GGML_TYPE_Q8_0 && w->ne[0] == p->k && w->ne[1] == p->n) {
        if (!quantized) quantized = cpu_decode_quantize(ctx, x, p);
        GGML_ASSERT(quantized->type == GGML_TYPE_Q8_0 && quantized->ne[0] == p->k
                    && ggml_nelements(quantized) == p->k);
        // The public custom-op API inherits its first argument's shape. This
        // unused prototype costs N*4 bytes; packed weights live with the loader.
        auto* shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p->n, 1);
        return ggml_map_custom2(ctx, shape, quantized, compute, GGML_N_TASKS_MAX, const_cast<PackedQ8*>(p));
    }
#else
    (void)p; (void)quantized;
#endif
    return ggml_mul_mat(ctx, w, x);
}
}
