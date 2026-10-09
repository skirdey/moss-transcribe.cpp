#ifndef MT_CPU_Q8_HPP
#define MT_CPU_Q8_HPP

#include "ggml.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace mt {
// Immutable, loader-owned copy. Ordinary weights remain available for prefill
// and embedding lookups. Sixteen complete rows are interleaved without changing
// any scale or code; the final partial tile is padded.
struct Q8Tile { uint16_t scales[16]; int8_t groups[8][64]; };
struct PackedQ8 {
    int k = 0, n = 0;
    std::vector<Q8Tile> tiles;
    bool round_trip_exact(const void* ordinary) const;
};

bool cpu_exact_q8_supported();
std::unique_ptr<PackedQ8> cpu_pack_q8(const ggml_tensor* weight, const void* ordinary);

// Optional shared conversion for projections of the same single-token input.
// Returns null on unsupported shapes; the matmul helper then uses eager ggml.
ggml_tensor* cpu_decode_quantize(ggml_context* ctx, ggml_tensor* input, const PackedQ8* packed);
ggml_tensor* cpu_decode_mul_mat(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input,
                               const PackedQ8* packed, ggml_tensor* quantized = nullptr);
}
#endif
