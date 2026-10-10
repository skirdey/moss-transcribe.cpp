// MIT. Paired cache traversal probe. No MOSS model routing.
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
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
    && defined(__F16C__) && defined(_OPENMP) && !defined(__AVXVNNIINT8__)
#include <immintrin.h>
#include <omp.h>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Shape { int k, n, m; };
struct alignas(64) Panel {
    int8_t groups[8][16][4];
    ggml_half scales[16];
};
static_assert(sizeof(Panel) == 576, "Dense four-code activation panel ABI");

using Clock = std::chrono::steady_clock;
double microseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count() * 1e6;
}
struct Phases { double allocation = 0, packing = 0, execution = 0, cleanup = 0; };

void pack_one(Panel& p, const block_q8_0* input, int blocks, int m, int t, int b) {
    // Worker-owned initialization includes unused columns and ABI padding.
    std::memset(&p, 0, sizeof(p));
    for (int c = 0; c < 16 && t * 16 + c < m; ++c) {
        const auto& x = input[size_t(t * 16 + c) * blocks + b];
        p.scales[c] = x.d;
        for (int g = 0; g < 8; ++g) std::memcpy(p.groups[g][c], x.qs + 4 * g, 4);
    }
}

std::vector<Panel> pack_input(const std::vector<block_q8_0>& input, int k, int m,
                            Phases* phases = nullptr) {
    const int blocks = k / 32, tiles = (m + 15) / 16;
    const auto begin = phases ? Clock::now() : Clock::time_point{};
    std::vector<Panel> result(size_t(tiles) * blocks);
    const auto allocated = phases ? Clock::now() : Clock::time_point{};
    for (int t = 0; t < tiles; ++t) for (int b = 0; b < blocks; ++b) {
        auto& p = result[size_t(t) * blocks + b];
        for (int c = 0; c < 16 && t * 16 + c < m; ++c) {
            const auto& x = input[size_t(t * 16 + c) * blocks + b];
            p.scales[c] = x.d;
            for (int g = 0; g < 8; ++g) {
                std::memcpy(p.groups[g][c], x.qs + 4 * g, 4);
            }
        }
    }
    if (phases) {
        phases->allocation = microseconds(begin, allocated);
        phases->packing = microseconds(allocated, Clock::now());
    }
    return result;
}

// Two weight rows and sixteen input rows. Each VNNI instruction reduces only four
// codes, so all eight original floating accumulation chains remain separate.
// Outputs have the actual GGML layout y[input_row * N + weight_row].
inline float inline_half(ggml_half h, bool round_down) {
    // The pinned software converter subtracts equal positive values for half
    // +0. Under downward rounding that produces -0; F16C alone produces +0.
    if (h == 0 && round_down) return -0.0f;
    return _cvtsh_ss(h);
}

template<int R, bool InlineHalf = false>
__attribute__((noinline))
void tile(const block_q8_0* weights, const Panel* panels, int blocks,
          int n, int columns, float* output) {
    const bool round_down = InlineHalf && ((_mm_getcsr() & 0x6000u) == 0x2000u);
    __m512 acc[R][8];
    for (auto& row : acc) for (auto& a : row) a = _mm512_setzero_ps();
    const __mmask16 active = static_cast<__mmask16>((1u << columns) - 1);
    const auto offsets = _mm512_mullo_epi32(_mm512_setr_epi32(
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), _mm512_set1_epi32(n));
    for (int b = 0; b < blocks; ++b) {
        const auto xs = _mm512_cvtph_ps(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(panels[b].scales)));
        __m512 scales[R];
        for (int r = 0; r < R; ++r) {
            const auto h = weights[r * blocks + b].d;
            const float converted = InlineHalf ? inline_half(h, round_down) : ggml_fp16_to_fp32(h);
            scales[r] = _mm512_mul_ps(xs, _mm512_set1_ps(converted));
        }
        #pragma GCC unroll 8
        for (int g = 0; g < 8; ++g) {
            const auto codes = _mm512_load_si512(panels[b].groups[g]);
            #pragma GCC unroll 2
            for (int r = 0; r < R; ++r) {
                int32_t raw;
                std::memcpy(&raw, weights[r * blocks + b].qs + 4 * g, sizeof(raw));
                const auto w = _mm512_set1_epi32(raw);
                // PSIGNB's wrapped negation is performed directly, including
                // literal -128. abs(-128) is unsigned 128 for DPBUSD.
                const auto zero = _mm512_setzero_si512();
                const auto negative = _mm512_cmp_epi8_mask(w, zero, _MM_CMPINT_LT);
                const auto signed_codes = _mm512_mask_sub_epi8(codes, negative, zero, codes);
                const auto sums = _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(w), signed_codes);
                acc[r][g] = _mm512_fmadd_ps(scales[r], _mm512_cvtepi32_ps(sums), acc[r][g]);
            }
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
void half_conversion_audit() {
    const unsigned saved = _mm_getcsr();
    size_t bit_differences = 0;
    // Exercise every half payload (including NaNs) under all rounding modes
    // and FTZ/DAZ settings. Mask exceptions during this isolated diagnostic;
    // restore the exact original MXCSR before any arithmetic/timing trial.
    for (unsigned round = 0; round < 4; ++round) for (unsigned flags = 0; flags < 4; ++flags) {
        _mm_setcsr((saved & ~0xe07fu) | 0x1f80u | (round << 13) |
                   ((flags & 1) ? 0x8000u : 0) | ((flags & 2) ? 0x40u : 0));
        for (unsigned code = 0; code < 65536; ++code) {
            const float reference = ggml_fp16_to_fp32(static_cast<ggml_half>(code));
            const float converted = inline_half(static_cast<ggml_half>(code), round == 1);
            bit_differences += std::memcmp(&reference, &converted, sizeof(float)) != 0;
        }
    }
    _mm_setcsr(saved);
    std::printf("{\"kind\":\"halfConversion\",\"halfPatterns\":65536,\"mxcsrModes\":16,\"comparisons\":1048576,\"floatBitDifferences\":%zu,\"originalMxcsrRestored\":%s}\n",
                bit_differences, _mm_getcsr() == saved ? "true" : "false");
    std::fflush(stdout);
    require(!bit_differences && _mm_getcsr() == saved, "exhaustive half conversion and MXCSR restoration");
}
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
    std::vector<float> reference(size_t(s.n) * s.m), candidate(reference.size()),
        parallel_result(reference.size()), inline_result(reference.size()),
        cache_result(reference.size()), graph_result(reference.size());
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
    auto execute_tile = [&](const Panel* packed, int wt, int mt, float* output, bool inline_half = false) {
        const auto* w = weights.data() + size_t(wt * 2) * blocks;
        const auto* p = packed + size_t(mt) * blocks;
        auto* y = output + size_t(mt * 16) * s.n + wt * 2;
        const int columns = std::min(16, s.m - mt * 16);
        if (inline_half) {
            if (wt * 2 + 1 < s.n) tile<2,true>(w, p, blocks, s.n, columns, y);
            else tile<1,true>(w, p, blocks, s.n, columns, y);
        } else {
            if (wt * 2 + 1 < s.n) tile<2,false>(w, p, blocks, s.n, columns, y);
            else tile<1,false>(w, p, blocks, s.n, columns, y);
        }
    };
    size_t parallel_panel_differences = 0;
    std::vector<int> inline_owners(weight_tiles, -1), cache_owners(weight_tiles, -1);
    auto compute = [&](int variant, Phases* phases = nullptr, bool check_panels = false) {
        if (variant == 1) {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "ordinary graph compute");
            return;
        }
        if (variant == 2) {
            // Includes panel allocation, conversion layout and scale setup.
            auto packed = pack_input(x, s.k, s.m, phases);
            const auto begin = phases ? Clock::now() : Clock::time_point{};
            #pragma omp parallel for num_threads(threads) schedule(static)
            for (int wt = 0; wt < weight_tiles; ++wt) for (int mt = 0; mt < input_tiles; ++mt)
                execute_tile(packed.data(), wt, mt, candidate.data());
            if (phases) {
                const auto done = Clock::now();
                phases->execution = microseconds(begin, done);
                std::vector<Panel>().swap(packed);
                phases->cleanup = microseconds(done, Clock::now());
            }
        } else if (variant >= 3 && variant <= 5) {
            const size_t count = size_t(input_tiles) * blocks;
            const auto begin = phases ? Clock::now() : Clock::time_point{};
            // Default initialization allocates without a serial zero-fill.
            // Each worker fully initializes its panels before the implicit barrier.
            std::unique_ptr<Panel[]> packed(new Panel[count]);
            const auto allocated = phases ? Clock::now() : Clock::time_point{};
            Clock::time_point packed_done{};
            #pragma omp parallel num_threads(threads)
            {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < count; ++i)
                    pack_one(packed[i], x.data(), blocks, s.m, int(i / blocks), int(i % blocks));
                // Diagnostic-only extra barrier; absent from primary timings.
                if (phases) {
                    #pragma omp single
                    { packed_done = Clock::now(); }
                }
                if (variant == 5) {
                    // Match GCC's static contiguous weight-row partition, then
                    // reuse each input panel across that worker's weight subset.
                    const int nth = omp_get_num_threads(), ith = omp_get_thread_num();
                    const int q = weight_tiles / nth, remainder = weight_tiles % nth;
                    const int begin_wt = ith * q + std::min(ith, remainder);
                    const int end_wt = begin_wt + q + (ith < remainder);
                    if (check_panels) for (int wt = begin_wt; wt < end_wt; ++wt)
                        cache_owners[wt] = ith;
                    for (int mt = 0; mt < input_tiles; ++mt)
                        for (int wt = begin_wt; wt < end_wt; ++wt)
                            execute_tile(packed.get(), wt, mt, cache_result.data(), true);
                    // Retain the prior omp-for's end barrier and team join.
                    #pragma omp barrier
                } else {
                    #pragma omp for schedule(static)
                    for (int wt = 0; wt < weight_tiles; ++wt) {
                        if (check_panels && variant == 4) inline_owners[wt] = omp_get_thread_num();
                        for (int mt = 0; mt < input_tiles; ++mt)
                            execute_tile(packed.get(), wt, mt,
                                variant == 4 ? inline_result.data() : parallel_result.data(), variant == 4);
                    }
                }
            }
            if (phases) {
                const auto done = Clock::now();
                phases->allocation = microseconds(begin, allocated);
                phases->packing = microseconds(allocated, packed_done); // includes team creation
                phases->execution = microseconds(packed_done, done); // includes team join
                packed.reset();
                phases->cleanup = microseconds(done, Clock::now());
            } else if (check_panels) {
                // Check actual parallel output, including inactive columns/padding.
                for (size_t i = 0; i < count; ++i) {
                    const auto& p = packed[i];
                    const int t = int(i / blocks), b = int(i % blocks);
                    for (int c = 0; c < 16; ++c) {
                        const bool valid = t * 16 + c < s.m;
                        const auto* original = valid ? &x[size_t(t * 16 + c) * blocks + b] : nullptr;
                        parallel_panel_differences += p.scales[c] != (valid ? original->d : 0);
                        for (int g = 0; g < 8; ++g) for (int j = 0; j < 4; ++j)
                            parallel_panel_differences += p.groups[g][c][j] != (valid ? original->qs[4*g+j] : 0);
                    }
                    const auto* bytes = reinterpret_cast<const unsigned char*>(&p);
                    for (size_t j = 544; j < sizeof(Panel); ++j)
                        parallel_panel_differences += bytes[j] != 0;
                }
            }
        } else {
            // Same contiguous weight-row ownership and worker budget as VNNI.
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
    size_t bits = 0, graph_bits = 0, panel_bytes_different = 0, parallel_bits = 0,
        inline_bits = 0, cache_bits = 0, ownership_differences = 0;
    auto parity = [&] {
        std::fill(candidate.begin(), candidate.end(), NAN);
        std::fill(parallel_result.begin(), parallel_result.end(), NAN);
        std::fill(inline_result.begin(), inline_result.end(), NAN);
        std::fill(cache_result.begin(), cache_result.end(), NAN);
        std::fill(inline_owners.begin(), inline_owners.end(), -1);
        std::fill(cache_owners.begin(), cache_owners.end(), -1);
        compute(0); compute(1); compute(2); compute(3, nullptr, true);
        compute(4, nullptr, true); compute(5, nullptr, true); immutable();
        for (int wt = 0; wt < weight_tiles; ++wt) {
            require(inline_owners[wt] >= 0 && cache_owners[wt] >= 0, "all weight tiles owned");
            ownership_differences += inline_owners[wt] != cache_owners[wt];
        }
        ggml_backend_tensor_get(y_tensor, graph_result.data(), 0, ggml_nbytes(y_tensor));
        bits += differences(reference, candidate);
        parallel_bits += differences(reference, parallel_result);
        inline_bits += differences(reference, inline_result);
        cache_bits += differences(reference, cache_result);
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
    std::array<double,6> us{};
    std::array<Phases,4> audit{};
    std::array<std::vector<double>,6> samples;
    if (timing && !std::strcmp(distribution, "normal")) {
        require(!bits && !graph_bits && !panel_bytes_different && !parallel_bits &&
                !parallel_panel_differences && !inline_bits && !cache_bits &&
                !ownership_differences, "initial exactness before timing");
        const int repeats = s.m >= 1500 ? 1 : s.k >= 1024 ? 2 : 10;
        for (int round = 0; round < 7; ++round) for (int step = 0; step < 6; ++step) {
            const int v = round % 2 ? (round + 5 - step) % 6 : (round + step) % 6;
            const auto begin = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < repeats; ++repeat) compute(v);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() * 1e6 / repeats;
            if (round) samples[v].push_back(elapsed);
        }
        for (int v = 0; v < 6; ++v) {
            std::sort(samples[v].begin(), samples[v].end());
            us[v] = (samples[v][2] + samples[v][3]) * .5;
        }
        // Separate diagnostic passes. These timers and the extra single barrier
        // do not contaminate the paired primary total measurement above.
        std::array<std::array<std::vector<double>,4>,4> phase_samples;
        for (int round = 0; round < 7; ++round) for (int step = 0; step < 4; ++step) {
            const int v = (round + step) % 4;
            Phases p; compute(v + 2, &p); immutable();
            require(!differences(reference, v == 3 ? cache_result : v == 2 ? inline_result : v == 1 ? parallel_result : candidate), "diagnostic outputs");
            if (round) {
                phase_samples[v][0].push_back(p.allocation); phase_samples[v][1].push_back(p.packing);
                phase_samples[v][2].push_back(p.execution); phase_samples[v][3].push_back(p.cleanup);
            }
        }
        for (int v = 0; v < 4; ++v) {
            for (auto& a : phase_samples[v]) std::sort(a.begin(), a.end());
            auto median = [&](int stage) { return (phase_samples[v][stage][2] + phase_samples[v][stage][3]) * .5; };
            audit[v] = {median(0),median(1),median(2),median(3)};
        }
        parity();
    }
    std::printf("{\"threads\":%d,\"K\":%d,\"N\":%d,\"M\":%d,\"distribution\":\"%s\",\"inputUpdates\":2,\"floatBitDifferences\":%zu,\"ordinaryGraphBitDifferences\":%zu,\"panelByteDifferences\":%zu,\"parallelFloatBitDifferences\":%zu,\"inlineFloatBitDifferences\":%zu,\"cacheFloatBitDifferences\":%zu,\"cacheWorkerOwnershipDifferences\":%zu,\"parallelPanelByteDifferences\":%zu,\"ordinaryWeightBytes\":%zu,\"packedWeightBytes\":0,\"activationPanelBytes\":%zu,\"panelConstructionIncluded\":true,\"inputQuantizationExcluded\":true,\"dotLoopUs\":%.3f,\"ordinaryGraphUs\":%.3f,\"serialVnniUs\":%.3f,\"parallelVnniUs\":%.3f,\"graphSpeedup\":%.4f,\"serialToParallelSpeedup\":%.4f,\"inlineVnniUs\":%.3f,\"graphToInlineSpeedup\":%.4f,\"parallelToInlineSpeedup\":%.4f,\"cacheVnniUs\":%.3f,\"graphToCacheSpeedup\":%.4f,\"inlineToCacheSpeedup\":%.4f,\"diagnosticSeparatePasses\":true,\"parallelDiagnosticExtraBarrier\":true,\"serialAllocationUs\":%.3f,\"serialPackingUs\":%.3f,\"serialExecutionUs\":%.3f,\"serialCleanupUs\":%.3f,\"parallelAllocationUs\":%.3f,\"parallelPackingAndTeamUs\":%.3f,\"parallelExecutionAndJoinUs\":%.3f,\"parallelCleanupUs\":%.3f,\"inlineAllocationUs\":%.3f,\"inlinePackingAndTeamUs\":%.3f,\"inlineExecutionAndJoinUs\":%.3f,\"inlineCleanupUs\":%.3f,\"cacheAllocationUs\":%.3f,\"cachePackingAndTeamUs\":%.3f,\"cacheExecutionAndJoinUs\":%.3f,\"cacheCleanupUs\":%.3f,\"fullInputLatency\":false",
        threads,s.k,s.n,s.m,distribution,bits,graph_bits,panel_bytes_different,
        parallel_bits,inline_bits,cache_bits,ownership_differences,parallel_panel_differences,
        weights.size()*sizeof(block_q8_0),size_t(input_tiles)*blocks*sizeof(Panel),
        us[0],us[1],us[2],us[3],us[3] ? us[1]/us[3] : 0,us[3] ? us[2]/us[3] : 0,us[4],us[4] ? us[1]/us[4] : 0,us[4] ? us[3]/us[4] : 0,us[5],us[5] ? us[1]/us[5] : 0,us[5] ? us[4]/us[5] : 0,
        audit[0].allocation,audit[0].packing,audit[0].execution,audit[0].cleanup,
        audit[1].allocation,audit[1].packing,audit[1].execution,audit[1].cleanup,
        audit[2].allocation,audit[2].packing,audit[2].execution,audit[2].cleanup,
        audit[3].allocation,audit[3].packing,audit[3].execution,audit[3].cleanup);
    std::printf(",\"sortedPrimarySamplesUs\":[");
    for (int v = 0; v < 6; ++v) {
        std::printf("%s[", v ? "," : "");
        for (size_t i = 0; i < samples[v].size(); ++i)
            std::printf("%s%.3f", i ? "," : "", samples[v][i]);
        std::printf("]");
    }
    std::printf("]}\n");
    std::fflush(stdout);
    require(!bits && !graph_bits && !panel_bytes_different && !parallel_bits && !parallel_panel_differences && !inline_bits && !cache_bits && !ownership_differences, "exact serial/parallel pinned dot, ordinary graph and initialized panel bytes");
}
}

int main(int argc, char** argv) {
    ggml_backend_t backend = nullptr;
    try {
        const bool timing = argc == 2 && !std::strcmp(argv[1], "--benchmark");
        require(argc == 1 || timing, "usage: moss_vnni_q8_cache [--benchmark]");
        ggml_cpu_init();
        const auto* t = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (!ggml_cpu_has_avx512_vnni()
            || t->nrows != 1 || t->vec_dot_type != GGML_TYPE_Q8_0) return 77;
        half_conversion_audit();
        backend = ggml_backend_cpu_init(); require(backend, "CPU backend");
        const std::vector<Shape> shapes = {{32,17,1},{96,17,2},{32,31,7},{96,31,16},
            {96,17,31},{1024,128,16},{3072,128,16},{1024,1024,64},{1024,4096,64}};
        for (int threads : timing ? std::vector<int>{1,8,16} : std::vector<int>{1,16}) {
            ggml_backend_cpu_set_n_threads(backend, threads);
            for (auto s : shapes) for (const char* d : {"normal","zero","extreme","scales"})
                trial(backend, threads, s, d, timing);
            // Encoder row count is checked at both budgets; only the 16-worker
            // actual-width shape is timed to bound the shared-host trial.
            if (!timing || threads == 16)
                for (auto s : std::vector<Shape>{{1024,1024,1500},{1024,4096,1500},{4096,1024,1500}})
                    trial(backend, threads, s, "normal", timing);
        }
        ggml_backend_free(backend); return 0;
    } catch (const std::exception& e) {
        if (backend) ggml_backend_free(backend);
        std::fprintf(stderr, "moss_vnni_q8_cache: %s\n", e.what()); return 1;
    }
}
#else
int main() {
    std::fprintf(stderr, "Requires pinned AVX512 VNNI/BW/VL/F16C/OpenMP; no portable fallback timing\n");
    return 77;
}
#endif
