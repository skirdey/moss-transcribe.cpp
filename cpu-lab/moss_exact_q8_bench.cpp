// MIT research probe. Preserve the pinned x86 VNNI dot's eight accumulator
// chains and final horizontal-add tree while computing 16 output rows at once.
// This is a warm, prequantized M=1 dot probe, not full-model latency evidence.
#include "ggml.h"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__F16C__) && defined(_OPENMP)
#include <immintrin.h>
#include <omp.h>

struct Tile {
    ggml_half scales[16];
    int8_t groups[8][64]; // Each group holds 4 codes for each of 16 output rows.
};
static_assert(sizeof(Tile) == 16 * sizeof(block_q8_0), "packing adds no bytes");

static std::vector<Tile> pack(const std::vector<block_q8_0>& w, int k, int n) {
    const int blocks = k / 32, tiles = (n + 15) / 16;
    std::vector<Tile> result(tiles * blocks);
    for (int tile = 0; tile < tiles; ++tile) for (int b = 0; b < blocks; ++b) {
        auto& dst = result[tile * blocks + b];
        for (int lane = 0; lane < 16 && tile * 16 + lane < n; ++lane) {
            const auto& src = w[(tile * 16 + lane) * blocks + b];
            dst.scales[lane] = src.d;
            for (int g = 0; g < 8; ++g)
                std::memcpy(dst.groups[g] + lane * 4, src.qs + g * 4, 4);
        }
    }
    return result;
}

static bool round_trip(const std::vector<block_q8_0>& w,
                       const std::vector<Tile>& p, int k, int n) {
    const int blocks = k / 32;
    for (int row = 0; row < n; ++row) for (int b = 0; b < blocks; ++b) {
        const auto& tile = p[(row / 16) * blocks + b];
        block_q8_0 restored{};
        restored.d = tile.scales[row % 16];
        for (int g = 0; g < 8; ++g)
            std::memcpy(restored.qs + g * 4, tile.groups[g] + row % 16 * 4, 4);
        if (std::memcmp(&restored, &w[row * blocks + b], sizeof(restored))) return false;
    }
    return true;
}

static void exact_dot(const Tile* p, const block_q8_0* x, int blocks, float* y, int lanes) {
    __m512 acc[8];
    for (auto& a : acc) a = _mm512_setzero_ps();
    const auto zero = _mm512_setzero_si512();
    for (int b = 0; b < blocks; ++b) {
        const auto scale = _mm512_mul_ps(
            _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p[b].scales))),
            _mm512_set1_ps(ggml_fp16_to_fp32(x[b].d)));
        for (int g = 0; g < 8; ++g) {
            int32_t four;
            std::memcpy(&four, x[b].qs + g * 4, 4);
            const auto q = _mm512_loadu_si512(p[b].groups[g]);
            const auto input = _mm512_set1_epi32(four);
            // Match PSIGNB semantics, including -128 wraparound. On this host
            // the pinned reference uses VNNI, without int16 pair saturation.
            const auto negative = _mm512_cmp_epi8_mask(q, zero, _MM_CMPINT_LT);
            const auto signed_input = _mm512_mask_sub_epi8(input, negative, zero, input);
            const auto sum = _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(q), signed_input);
            acc[g] = _mm512_fmadd_ps(scale, _mm512_cvtepi32_ps(sum), acc[g]);
        }
    }
    // Exact operation tree from hsum_float_8 in the pinned x86/quants.c.
    const auto even = _mm512_add_ps(_mm512_add_ps(acc[4], acc[0]), _mm512_add_ps(acc[6], acc[2]));
    const auto odd  = _mm512_add_ps(_mm512_add_ps(acc[5], acc[1]), _mm512_add_ps(acc[7], acc[3]));
    _mm512_mask_storeu_ps(y, static_cast<__mmask16>((1u << lanes) - 1), _mm512_add_ps(even, odd));
}

int main() {
    ggml_cpu_init();
    if (!ggml_cpu_has_avx512_vnni()) return 77;
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    bool exact = true;
    for (int threads : {1, 16}) for (int k : {32, 1024, 3072})
    for (int n : {17, 1024, 3072}) for (const char* distribution : {"normal", "extreme", "zero"}) {
        const int blocks = k / 32, tiles = (n + 15) / 16;
        std::mt19937 random(20261009); std::normal_distribution<float> normal(0, 1);
        std::vector<float> source(k * n), input(k), eager(n), candidate(n);
        for (auto& f : source) f = normal(random);
        for (auto& f : input) f = normal(random);
        std::vector<block_q8_0> w(blocks * n), x(blocks);
        for (int row = 0; row < n; ++row) traits->from_float(source.data() + row * k, w.data() + row * blocks, k);
        traits->from_float(input.data(), x.data(), k);
        if (std::strcmp(distribution, "normal")) {
            const int8_t codes[] = {-128, -127, 0, 127};
            const bool all_zero = !std::strcmp(distribution, "zero");
            for (size_t b = 0; b < w.size(); ++b) {
                w[b].d = ggml_fp32_to_fp16(all_zero ? 0.f : 1.f / (1 + b % 9));
                for (int j = 0; j < 32; ++j) w[b].qs[j] = all_zero ? 0 : codes[(b + j) % 4];
            }
            for (size_t b = 0; b < x.size(); ++b) {
                x[b].d = ggml_fp32_to_fp16(all_zero ? 0.f : -1.f / (1 + b % 5));
                for (int j = 0; j < 32; ++j) x[b].qs[j] = all_zero ? 0 : codes[(b * 3 + j) % 4];
            }
        }
        auto packing_start = std::chrono::steady_clock::now();
        auto packed = pack(w, k, n);
        const double packing_us = std::chrono::duration<double>(std::chrono::steady_clock::now() - packing_start).count() * 1e6;
        if (!round_trip(w, packed, k, n)) return 2;
        auto compute = [&](bool use_packed) {
            if (use_packed) {
                #pragma omp parallel for num_threads(threads) schedule(static)
                for (int t = 0; t < tiles; ++t)
                    exact_dot(packed.data() + t * blocks, x.data(), blocks, candidate.data() + t * 16, std::min(16, n - t * 16));
            } else {
                #pragma omp parallel for num_threads(threads) schedule(static)
                for (int row = 0; row < n; ++row)
                    traits->vec_dot(k, eager.data() + row, 0, w.data() + row * blocks, 0, x.data(), 0, 1);
            }
        };
        compute(false); compute(true);
        size_t differences = 0; float max_abs = 0;
        for (int row = 0; row < n; ++row) {
            if (!std::isfinite(eager[row]) || !std::isfinite(candidate[row])) return 2;
            differences += std::memcmp(&eager[row], &candidate[row], sizeof(float)) != 0;
            max_abs = std::max(max_abs, std::fabs(eager[row] - candidate[row]));
        }
        exact &= differences == 0;
        std::vector<double> a, b;
        for (int round = 0; round < 6; ++round) for (int step = 0; step < 2; ++step) {
            const bool use_packed = (round + step) % 2;
            auto start = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < 5; ++repeat) compute(use_packed);
            double us = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() * 1e6 / 5;
            if (round) (use_packed ? b : a).push_back(us);
        }
        std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
        std::printf("{\"threads\":%d,\"K\":%d,\"N\":%d,\"M\":1,\"distribution\":\"%s\",\"weightRoundTripExact\":true,\"floatBitDifferences\":%zu,\"maxAbsError\":%.9g,\"eagerUs\":%.3f,\"packedUs\":%.3f,\"speedup\":%.4f,\"packingUs\":%.3f,\"packedBytes\":%zu,\"weightBytes\":%zu,\"inputQuantizationExcluded\":true}\n",
                    threads, k, n, distribution, differences, max_abs, a[2], b[2], a[2] / b[2], packing_us, packed.size() * sizeof(Tile), w.size() * sizeof(block_q8_0));
    }
    return exact ? 0 : 1;
}
#else
int main() { std::fprintf(stderr, "Requires a native AVX512 VNNI/BW/F16C build with OpenMP\n"); return 77; }
#endif
