// MIT. Owned-vector, exact-order AMX batch probe. No MOSS model routing.
// Transform activations, retain original Q8 weights, and compare both the
// pinned dot routine and an ordinary GGML graph before reporting timing.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__linux__) && defined(__AMX_INT8__) && defined(__AMX_TILE__) \
    && defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
    && defined(__F16C__) && defined(_OPENMP) && !defined(__AVXVNNIINT8__)
#include <immintrin.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Shape { int k, n, m; };
struct alignas(64) Panel {
    int8_t groups[8][16][4]{};
    ggml_half scales[16]{};
    uint64_t minus128[8]{};
};
struct alignas(64) Config {
    uint8_t palette = 1, start = 0, reserved[14]{};
    uint16_t columns[16]{};
    uint8_t rows[16]{};
};
static_assert(sizeof(Config) == 64, "AMX configuration ABI");

std::vector<Panel> pack_input(const std::vector<block_q8_0>& input, int k, int m) {
    const int blocks = k / 32, tiles = (m + 15) / 16;
    std::vector<Panel> result(size_t(tiles) * blocks);
    for (int t = 0; t < tiles; ++t) for (int b = 0; b < blocks; ++b) {
        auto& p = result[size_t(t) * blocks + b];
        for (int c = 0; c < 16 && t * 16 + c < m; ++c) {
            const auto& x = input[size_t(t * 16 + c) * blocks + b];
            p.scales[c] = x.d;
            for (int g = 0; g < 8; ++g) {
                std::memcpy(p.groups[g][c], x.qs + 4 * g, 4);
                for (int j = 0; j < 4; ++j)
                    if (x.qs[4 * g + j] == -128)
                        p.minus128[g] |= uint64_t(1) << (4 * c + j);
            }
        }
    }
    return result;
}

// Two weight rows and sixteen input rows. Each AMX product reduces only four
// codes, so all eight original floating accumulation chains remain separate.
// Outputs have the actual GGML layout y[input_row * N + weight_row].
template<int R>
void tile(const block_q8_0* weights, const Panel* panels, int blocks,
          int n, int columns, float* output) {
    Config config;
    config.columns[0] = 4; config.rows[0] = R;
    config.columns[1] = 64; config.rows[1] = 1;
    config.columns[2] = 64; config.rows[2] = R;
    _tile_loadconfig(&config);
    __m512 acc[R][8];
    for (auto& row : acc) for (auto& a : row) a = _mm512_setzero_ps();
    alignas(64) int32_t sums[R][16];
    const __mmask16 active = static_cast<__mmask16>((1u << columns) - 1);
    const auto offsets = _mm512_mullo_epi32(_mm512_setr_epi32(
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), _mm512_set1_epi32(n));
    for (int b = 0; b < blocks; ++b) {
        const auto xs = _mm512_cvtph_ps(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(panels[b].scales)));
        __m512 scales[R];
        for (int r = 0; r < R; ++r) scales[r] = _mm512_mul_ps(xs,
            _mm512_set1_ps(ggml_fp16_to_fp32(weights[r * blocks + b].d)));
        #pragma GCC unroll 8
        for (int g = 0; g < 8; ++g) {
            _tile_loadd(0, weights[b].qs + 4 * g, blocks * sizeof(block_q8_0));
            _tile_loadd(1, panels[b].groups[g], 64);
            _tile_zero(2);
            _tile_dpbssd(2, 0, 1);
            _tile_stored(2, sums, 64);
            // Preserve pinned PSIGNB's wrapped negation of activation -128.
            const uint64_t negative = panels[b].minus128[g];
            if (negative) for (int r = 0; r < R; ++r)
                for (int c = 0; c < columns; ++c) for (int j = 0; j < 4; ++j) {
                    const int w = weights[r * blocks + b].qs[4 * g + j];
                    if (w < 0 && (negative & (uint64_t(1) << (4 * c + j))))
                        sums[r][c] += 256 * w;
                }
            #pragma GCC unroll 2
            for (int r = 0; r < R; ++r)
                acc[r][g] = _mm512_fmadd_ps(scales[r],
                    _mm512_cvtepi32_ps(_mm512_load_si512(sums[r])), acc[r][g]);
        }
    }
    for (int r = 0; r < R; ++r) {
        const auto even = _mm512_add_ps(_mm512_add_ps(acc[r][4], acc[r][0]),
                                      _mm512_add_ps(acc[r][6], acc[r][2]));
        const auto odd = _mm512_add_ps(_mm512_add_ps(acc[r][5], acc[r][1]),
                                     _mm512_add_ps(acc[r][7], acc[r][3]));
        _mm512_mask_i32scatter_ps(output + r, active, offsets,
                                 _mm512_add_ps(even, odd), sizeof(float));
    }
    _tile_release();
}

struct Resources {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ~Resources() {
        if (alloc) ggml_gallocr_free(alloc);
        if (buffer) ggml_backend_buffer_free(buffer);
        if (ctx) ggml_free(ctx);
    }
};
size_t differences(const std::vector<float>& a, const std::vector<float>& b) {
    size_t result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "finite outputs");
        result += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
    }
    return result;
}

void trial(ggml_backend_t backend, int threads, Shape s,
           const char* distribution, bool timing) {
    const int blocks = s.k / 32, weight_tiles = (s.n + 1) / 2;
    const int input_tiles = (s.m + 15) / 16;
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::mt19937 random(20261010);
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> source(size_t(s.k) * s.m), row(s.k);
    std::vector<block_q8_0> weights(size_t(blocks) * s.n), x(size_t(blocks) * s.m);
    std::vector<float> reference(size_t(s.n) * s.m), candidate(reference.size()), graph_result(reference.size());
    for (auto& f : source) f = normal(random);
    for (int r = 0; r < s.n; ++r) {
        for (auto& f : row) f = normal(random);
        traits->from_float(row.data(), weights.data() + r * blocks, s.k);
    }
    const std::array<ggml_half,8> scale_edges = {0x0000,0x8000,0x0001,0x8001,0x0400,0x8400,0x7bff,0xfbff};
    if (std::strcmp(distribution, "normal")) {
        const int8_t codes[] = {-128,-127,0,127};
        const bool zero = !std::strcmp(distribution, "zero");
        for (size_t b = 0; b < weights.size(); ++b) {
            weights[b].d = ggml_fp32_to_fp16(zero ? 0.f : 1.f / (1 + b % 9));
            if (!std::strcmp(distribution, "scales")) weights[b].d = scale_edges[b % 8];
            for (int j = 0; j < 32; ++j) weights[b].qs[j] = zero ? 0 : codes[(b + j) % 4];
        }
    }
    const auto saved_weights = weights;
    std::vector<block_q8_0> saved_x;
    std::vector<float> saved_source;
    Resources leaves, ops;
    leaves.ctx = ggml_init({1024 * 1024, nullptr, true});
    ops.ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    require(leaves.ctx && ops.ctx, "contexts");
    auto w_tensor = ggml_new_tensor_2d(leaves.ctx, GGML_TYPE_Q8_0, s.k, s.n);
    auto x_tensor = ggml_new_tensor_2d(leaves.ctx, GGML_TYPE_Q8_0, s.k, s.m);
    // Stable external storage survives every standalone/graph timing round.
    leaves.buffer = ggml_backend_alloc_ctx_tensors(leaves.ctx, backend);
    require(leaves.buffer, "external operands");
    auto y_tensor = ggml_mul_mat(ops.ctx, w_tensor, x_tensor);
    ggml_set_output(y_tensor);
    auto graph = ggml_new_graph(ops.ctx);
    ggml_build_forward_expand(graph, y_tensor);
    ops.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(ops.alloc, graph), "ordinary graph allocation");
    ggml_backend_tensor_set(w_tensor, weights.data(), 0, ggml_nbytes(w_tensor));
    auto compute = [&](int variant) {
        if (variant == 1) {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "ordinary graph compute");
            return;
        }
        if (variant == 2) {
            // Includes panel allocation, conversion layout and scale/mask setup.
            auto packed = pack_input(x, s.k, s.m);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (int wt = 0; wt < weight_tiles; ++wt) for (int mt = 0; mt < input_tiles; ++mt) {
                const auto* w = weights.data() + size_t(wt * 2) * blocks;
                const auto* p = packed.data() + size_t(mt) * blocks;
                auto* y = candidate.data() + size_t(mt * 16) * s.n + wt * 2;
                const int columns = std::min(16, s.m - mt * 16);
                if (wt * 2 + 1 < s.n) tile<2>(w, p, blocks, s.n, columns, y);
                else tile<1>(w, p, blocks, s.n, columns, y);
            }
        } else {
            // Same contiguous weight-row ownership and worker budget as AMX.
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (int wt = 0; wt < weight_tiles; ++wt) for (int mt = 0; mt < input_tiles; ++mt)
                for (int r = wt * 2; r < std::min(s.n, wt * 2 + 2); ++r)
                    for (int c = mt * 16; c < std::min(s.m, mt * 16 + 16); ++c)
                        traits->vec_dot(s.k, reference.data() + size_t(c) * s.n + r, 0,
                            weights.data() + size_t(r) * blocks, 0,
                            x.data() + size_t(c) * blocks, 0, 1);
        }
    };
    auto immutable = [&] {
        require(!std::memcmp(weights.data(), saved_weights.data(), weights.size() * sizeof(block_q8_0)), "weight bytes changed");
        require(!std::memcmp(x.data(), saved_x.data(), x.size() * sizeof(block_q8_0)), "activation bytes changed");
        require(!std::memcmp(source.data(), saved_source.data(), source.size() * sizeof(float)), "F32 source bytes changed");
        std::vector<block_q8_0> actual_w(weights.size()), actual_x(x.size());
        ggml_backend_tensor_get(w_tensor, actual_w.data(), 0, ggml_nbytes(w_tensor));
        ggml_backend_tensor_get(x_tensor, actual_x.data(), 0, ggml_nbytes(x_tensor));
        require(!std::memcmp(actual_w.data(), weights.data(), ggml_nbytes(w_tensor)), "graph weight bytes changed");
        require(!std::memcmp(actual_x.data(), x.data(), ggml_nbytes(x_tensor)), "graph input bytes changed");
    };
    size_t bits = 0, graph_bits = 0, panel_bytes_different = 0;
    auto parity = [&] {
        compute(0); compute(1); compute(2); immutable();
        ggml_backend_tensor_get(y_tensor, graph_result.data(), 0, ggml_nbytes(y_tensor));
        bits += differences(reference, candidate);
        graph_bits += differences(reference, graph_result);
        auto packed = pack_input(x, s.k, s.m);
        for (int c = 0; c < s.m; ++c) for (int b = 0; b < blocks; ++b) {
            const auto& p = packed[size_t(c / 16) * blocks + b];
            const auto& original = x[size_t(c) * blocks + b];
            panel_bytes_different += p.scales[c % 16] != original.d;
            for (int g = 0; g < 8; ++g) for (int j = 0; j < 4; ++j)
                panel_bytes_different += p.groups[g][c % 16][j] != original.qs[4 * g + j];
        }
    };
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) for (auto& f : source) f *= -.5f;
        for (int c = 0; c < s.m; ++c)
            traits->from_float(source.data() + size_t(c) * s.k, x.data() + size_t(c) * blocks, s.k);
        if (std::strcmp(distribution, "normal")) {
            const int8_t codes[] = {-128,-127,0,127};
            const bool zero = !std::strcmp(distribution, "zero");
            for (size_t b = 0; b < x.size(); ++b) {
                x[b].d = ggml_fp32_to_fp16(zero ? 0.f : (pass ? 1.f : -1.f) / (1 + b % 5));
                if (!std::strcmp(distribution, "scales")) x[b].d = scale_edges[(b * 3 + pass) % 8];
                for (int j = 0; j < 32; ++j) x[b].qs[j] = zero ? 0 : codes[(b * 3 + j + pass) % 4];
            }
        }
        saved_x = x; saved_source = source;
        ggml_backend_tensor_set(x_tensor, x.data(), 0, ggml_nbytes(x_tensor));
        parity();
    }
    std::array<double,3> us{};
    if (timing && !std::strcmp(distribution, "normal")) {
        require(!bits && !graph_bits && !panel_bytes_different, "initial exactness before timing");
        std::array<std::vector<double>,3> samples;
        const int repeats = s.k >= 1024 ? 2 : 10;
        for (int round = 0; round < 6; ++round) for (int step = 0; step < 3; ++step) {
            const int v = round % 2 ? (round + 2 - step) % 3 : (round + step) % 3;
            const auto begin = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < repeats; ++repeat) compute(v);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() * 1e6 / repeats;
            if (round) samples[v].push_back(elapsed);
        }
        for (int v = 0; v < 3; ++v) { std::sort(samples[v].begin(), samples[v].end()); us[v] = samples[v][2]; }
        parity();
    }
    std::printf("{\"threads\":%d,\"K\":%d,\"N\":%d,\"M\":%d,\"distribution\":\"%s\",\"inputUpdates\":2,\"floatBitDifferences\":%zu,\"ordinaryGraphBitDifferences\":%zu,\"panelByteDifferences\":%zu,\"ordinaryWeightBytes\":%zu,\"packedWeightBytes\":0,\"activationPanelBytes\":%zu,\"panelConstructionIncluded\":true,\"inputQuantizationExcluded\":true,\"dotLoopUs\":%.3f,\"ordinaryGraphUs\":%.3f,\"amxUs\":%.3f,\"graphSpeedup\":%.4f,\"fullInputLatency\":false}\n",
        threads,s.k,s.n,s.m,distribution,bits,graph_bits,panel_bytes_different,
        weights.size()*sizeof(block_q8_0),size_t(input_tiles)*blocks*sizeof(Panel),
        us[0],us[1],us[2],us[2] ? us[1]/us[2] : 0);
    std::fflush(stdout);
    require(!bits && !graph_bits && !panel_bytes_different, "exact pinned dot, ordinary graph and panel bytes");
}
}

int main(int argc, char** argv) {
    ggml_backend_t backend = nullptr;
    try {
        const bool timing = argc == 2 && !std::strcmp(argv[1], "--benchmark");
        require(argc == 1 || timing, "usage: moss_amx_q8_batch [--benchmark]");
        ggml_cpu_init();
        const auto* t = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (!ggml_cpu_has_avx512_vnni() || !__builtin_cpu_supports("amx-int8")
            || t->nrows != 1 || t->vec_dot_type != GGML_TYPE_Q8_0) return 77;
        uint64_t features = 0;
        if (syscall(SYS_arch_prctl, 0x1021, &features) || !(features & (uint64_t(1) << 18))
            || syscall(SYS_arch_prctl, 0x1023, 18)) return 77;
        backend = ggml_backend_cpu_init(); require(backend, "CPU backend");
        const std::vector<Shape> shapes = {{32,17,1},{96,17,2},{32,31,7},{96,31,16},
            {96,17,31},{1024,128,16},{3072,128,16},{1024,1024,64},{1024,4096,64}};
        for (int threads : timing ? std::vector<int>{1,8,16} : std::vector<int>{1,16}) {
            ggml_backend_cpu_set_n_threads(backend, threads);
            for (auto s : shapes) for (const char* d : {"normal","zero","extreme","scales"})
                trial(backend, threads, s, d, timing);
            // Encoder row count is checked at both budgets; only the 16-worker
            // actual-width shape is timed to bound the shared-host trial.
            if (!timing || threads == 16) trial(backend, threads, {1024,1024,1500}, "normal", timing);
        }
        ggml_backend_free(backend); return 0;
    } catch (const std::exception& e) {
        if (backend) ggml_backend_free(backend);
        std::fprintf(stderr, "moss_amx_q8_batch: %s\n", e.what()); return 1;
    }
}
#else
int main() {
    std::fprintf(stderr, "Requires Linux AMX INT8/TILE and pinned AVX512 VNNI/F16C/OpenMP; no portable fallback timing\n");
    return 77;
}
#endif
