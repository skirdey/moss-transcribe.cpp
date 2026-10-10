// MIT. F32 GGML custom-op integration probe; no MOSS model routing.
// Includes activation quantization, packing, worker coordination and cleanup.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <new>
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

void pack_one(Panel& p, const block_q8_0* input, int blocks, int m, int t, int b) {
    // Worker-owned initialization includes unused columns and ABI padding.
    std::memset(&p, 0, sizeof(p));
    for (int c = 0; c < 16 && t * 16 + c < m; ++c) {
        const auto& x = input[size_t(t * 16 + c) * blocks + b];
        p.scales[c] = x.d;
        for (int g = 0; g < 8; ++g) std::memcpy(p.groups[g][c], x.qs + 4 * g, 4);
    }
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


// Reusable barrier for exactly the GGML callback workers. No nested team, no
// dependency on private GGML threadpool internals. Serialized graph executions
// own this context; it is not safe to run one graph concurrently with itself.
struct WorkerBarrier {
    std::atomic<int> arrived{0};
    std::atomic<unsigned> generation{0};
    void wait(int nth) {
        const unsigned observed = generation.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) == nth - 1) {
            arrived.store(0, std::memory_order_relaxed);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            while (generation.load(std::memory_order_acquire) == observed) _mm_pause();
        }
    }
};
struct Scratch {
    WorkerBarrier barrier;
    block_q8_0* quant = nullptr;
    Panel* panels = nullptr;
    std::atomic<bool> allocation_failed{false};
    bool cache_order = false, audit = false;
    int observed_workers = 0;
    size_t quant_byte_differences = 0, panel_byte_differences = 0;
    const block_q8_0* expected_quant = nullptr;
    std::vector<int> owners, visits;
};
void f32_operation(ggml_tensor* dst, int ith, int nth, void* userdata) {
    auto& scratch = *static_cast<Scratch*>(userdata);
    const auto* w = dst->src[0];
    const auto* x = dst->src[1];
    const int k = int(w->ne[0]), n = int(w->ne[1]), m = int(x->ne[1]);
    const int blocks = k / 32, weight_tiles = (n + 1) / 2, input_tiles = (m + 15) / 16;
    const size_t count = size_t(input_tiles) * blocks;
    if (ith == 0) {
        scratch.observed_workers = nth;
        scratch.quant = new (std::nothrow) block_q8_0[size_t(m) * blocks];
        scratch.panels = new (std::nothrow) Panel[count];
        scratch.allocation_failed.store(!scratch.quant || !scratch.panels);
    }
    scratch.barrier.wait(nth); // publish allocations
    if (scratch.allocation_failed.load()) {
        if (ith == 0) { delete[] scratch.quant; delete[] scratch.panels;
            scratch.quant = nullptr; scratch.panels = nullptr; }
        return;
    }
    if (scratch.audit) scratch.visits[ith]++;
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const auto* input = static_cast<const float*>(x->data);
    for (int c = ith; c < m; c += nth)
        traits->from_float(input + size_t(c) * k, scratch.quant + size_t(c) * blocks, k);
    scratch.barrier.wait(nth); // quantized rows available to packing workers
    for (size_t i = ith; i < count; i += nth)
        pack_one(scratch.panels[i], scratch.quant, blocks, m, int(i / blocks), int(i % blocks));
    scratch.barrier.wait(nth); // initialized panels visible to every consumer
    const int q = weight_tiles / nth, remainder = weight_tiles % nth;
    const int first = ith * q + std::min(ith, remainder), last = first + q + (ith < remainder);
    if (scratch.audit) for (int wt = first; wt < last; ++wt) scratch.owners[wt] = ith;
    auto compute_tile = [&](int wt, int mt) {
        const auto* weight = static_cast<const block_q8_0*>(w->data) + size_t(wt * 2) * blocks;
        const auto* panels = scratch.panels + size_t(mt) * blocks;
        auto* output = static_cast<float*>(dst->data) + size_t(mt * 16) * n + wt * 2;
        const int columns = std::min(16, m - mt * 16);
        if (wt * 2 + 1 < n) tile<2,true>(weight, panels, blocks, n, columns, output);
        else tile<1,true>(weight, panels, blocks, n, columns, output);
    };
    if (scratch.cache_order) {
        for (int mt = 0; mt < input_tiles; ++mt)
            for (int wt = first; wt < last; ++wt) compute_tile(wt, mt);
    } else {
        for (int wt = first; wt < last; ++wt)
            for (int mt = 0; mt < input_tiles; ++mt) compute_tile(wt, mt);
    }
    scratch.barrier.wait(nth); // all consumers finished before cleanup
    if (ith == 0) {
        if (scratch.audit) {
            const auto* actual = reinterpret_cast<const unsigned char*>(scratch.quant);
            const auto* expected = reinterpret_cast<const unsigned char*>(scratch.expected_quant);
            for (size_t j = 0; j < size_t(m) * blocks * sizeof(block_q8_0); ++j)
                scratch.quant_byte_differences += actual[j] != expected[j];
            for (size_t i = 0; i < count; ++i) {
                const auto& p = scratch.panels[i];
                const int t = int(i / blocks), b = int(i % blocks);
                for (int c = 0; c < 16; ++c) {
                    const auto* original = t * 16 + c < m ? &scratch.expected_quant[size_t(t * 16 + c) * blocks + b] : nullptr;
                    scratch.panel_byte_differences += p.scales[c] != (original ? original->d : 0);
                    for (int g = 0; g < 8; ++g) for (int j = 0; j < 4; ++j)
                        scratch.panel_byte_differences += p.groups[g][c][j] != (original ? original->qs[4*g+j] : 0);
                }
                const auto* bytes = reinterpret_cast<const unsigned char*>(&p);
                for (size_t j = 544; j < sizeof(Panel); ++j) scratch.panel_byte_differences += bytes[j] != 0;
            }
        }
        delete[] scratch.quant; delete[] scratch.panels;
        scratch.quant = nullptr; scratch.panels = nullptr;
    }
    // GGML's following node barrier/join includes the cleanup by worker zero.
}

void trial(ggml_backend_t backend, int threads, Shape s, const char* distribution, bool timing) {
    const int blocks = s.k / 32, weight_tiles = (s.n + 1) / 2;
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::mt19937 random(20261010);
    std::normal_distribution<float> normal(0, 1);
    std::vector<float> source(size_t(s.k) * s.m), row(s.k);
    std::vector<block_q8_0> weights(size_t(blocks) * s.n), quant(size_t(blocks) * s.m);
    for (auto& f : source) f = normal(random);
    const float input_edges[] = {-1000.f,1000.f,-0.f,1e-12f,-1e-12f,.5f,-.5f};
    if (std::strcmp(distribution,"normal"))
        for (size_t i = 0; i < source.size(); ++i)
            source[i] = !std::strcmp(distribution,"zero") ? 0.f : input_edges[i % 7];
    for (int r = 0; r < s.n; ++r) {
        for (auto& f : row) f = normal(random);
        traits->from_float(row.data(), weights.data() + size_t(r) * blocks, s.k);
    }
    const std::array<ggml_half,8> scale_edges = {0x0000,0x8000,0x0001,0x8001,0x0400,0x8400,0x7bff,0xfbff};
    if (!std::strcmp(distribution,"extreme") || !std::strcmp(distribution,"scales")) {
        const int8_t codes[] = {-128,-127,0,127};
        for (size_t b = 0; b < weights.size(); ++b) {
            weights[b].d = !std::strcmp(distribution,"scales") ? scale_edges[b % 8] : ggml_fp32_to_fp16(1.f / (1 + b % 9));
            for (int j = 0; j < 32; ++j) weights[b].qs[j] = codes[(b + j) % 4];
        }
    }
    const auto saved_weights = weights;
    std::vector<float> saved_source;
    std::vector<block_q8_0> saved_quant;
    Scratch prior, cache;
    cache.cache_order = true;
    for (auto* scratch : {&prior,&cache}) {
        scratch->owners.resize(weight_tiles, -1); scratch->visits.resize(threads, 0);
    }
    Resources leaves, ops;
    leaves.ctx = ggml_init({1024 * 1024,nullptr,true});
    ops.ctx = ggml_init({8 * 1024 * 1024,nullptr,true});
    require(leaves.ctx && ops.ctx,"contexts");
    auto w = ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_Q8_0,s.k,s.n);
    auto x = ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_F32,s.k,s.m);
    auto q = ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_Q8_0,s.k,s.m);
    leaves.buffer = ggml_backend_alloc_ctx_tensors(leaves.ctx,backend);
    require(leaves.buffer,"stable external operands");
    ggml_tensor* args[] = {w,x};
    auto ordinary = ggml_mul_mat(ops.ctx,w,x);
    auto prequantized = ggml_mul_mat(ops.ctx,w,q);
    auto old_order = ggml_custom_4d(ops.ctx,GGML_TYPE_F32,s.n,s.m,1,1,args,2,f32_operation,GGML_N_TASKS_MAX,&prior);
    auto new_order = ggml_custom_4d(ops.ctx,GGML_TYPE_F32,s.n,s.m,1,1,args,2,f32_operation,GGML_N_TASKS_MAX,&cache);
    std::array<ggml_tensor*,4> outputs = {ordinary,prequantized,old_order,new_order};
    std::array<ggml_cgraph*,4> graphs;
    auto all = ggml_new_graph(ops.ctx);
    for (int i = 0; i < 4; ++i) {
        ggml_set_output(outputs[i]); graphs[i] = ggml_new_graph(ops.ctx);
        ggml_build_forward_expand(graphs[i],outputs[i]); ggml_build_forward_expand(all,outputs[i]);
    }
    ops.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(ops.alloc,all),"graph allocation");
    ggml_backend_tensor_set(w,weights.data(),0,ggml_nbytes(w));
    std::vector<float> reference(size_t(s.n) * s.m), f32_result(reference.size()),
        q8_result(reference.size()), prior_result(reference.size()), cache_result(reference.size()), poison(reference.size(),NAN);
    auto compute = [&](int i) {
        require(ggml_backend_graph_compute(backend,graphs[i]) == GGML_STATUS_SUCCESS,"graph compute");
        require(!prior.allocation_failed.load() && !cache.allocation_failed.load(),"custom scratch allocation");
        if (i >= 2) {
            auto& scratch = i == 2 ? prior : cache;
            require(scratch.observed_workers == threads,"actual callback worker budget");
            require(!scratch.quant && !scratch.panels,"callback cleanup");
        }
    };
    auto immutable = [&] {
        require(!std::memcmp(weights.data(),saved_weights.data(),weights.size()*sizeof(block_q8_0)),"weight vector bytes changed");
        require(!std::memcmp(source.data(),saved_source.data(),source.size()*sizeof(float)),"F32 vector bytes changed");
        require(!std::memcmp(quant.data(),saved_quant.data(),quant.size()*sizeof(block_q8_0)),"Q8 control bytes changed");
        std::vector<float> actual_x(source.size());
        std::vector<block_q8_0> actual_w(weights.size()), actual_q(quant.size());
        ggml_backend_tensor_get(x,actual_x.data(),0,ggml_nbytes(x));
        ggml_backend_tensor_get(w,actual_w.data(),0,ggml_nbytes(w));
        ggml_backend_tensor_get(q,actual_q.data(),0,ggml_nbytes(q));
        require(!std::memcmp(actual_x.data(),source.data(),ggml_nbytes(x)),"external F32 operand changed");
        require(!std::memcmp(actual_w.data(),weights.data(),ggml_nbytes(w)),"external weight operand changed");
        require(!std::memcmp(actual_q.data(),quant.data(),ggml_nbytes(q)),"external Q8 control changed");
    };
    size_t f32_bits = 0, q8_bits = 0, prior_bits = 0, cache_bits = 0, ownership_differences = 0;
    auto parity = [&] {
        for (auto y : outputs) ggml_backend_tensor_set(y,poison.data(),0,ggml_nbytes(y));
        for (int r = 0; r < s.n; ++r) for (int c = 0; c < s.m; ++c)
            traits->vec_dot(s.k,reference.data()+size_t(c)*s.n+r,0,weights.data()+size_t(r)*blocks,0,quant.data()+size_t(c)*blocks,0,1);
        for (auto* scratch : {&prior,&cache}) {
            scratch->audit = true; scratch->expected_quant = quant.data();
            std::fill(scratch->owners.begin(),scratch->owners.end(),-1);
            std::fill(scratch->visits.begin(),scratch->visits.end(),0);
        }
        for (int i = 0; i < 4; ++i) compute(i);
        immutable();
        ggml_backend_tensor_get(ordinary,f32_result.data(),0,ggml_nbytes(ordinary));
        ggml_backend_tensor_get(prequantized,q8_result.data(),0,ggml_nbytes(prequantized));
        ggml_backend_tensor_get(old_order,prior_result.data(),0,ggml_nbytes(old_order));
        ggml_backend_tensor_get(new_order,cache_result.data(),0,ggml_nbytes(new_order));
        f32_bits += differences(reference,f32_result); q8_bits += differences(reference,q8_result);
        prior_bits += differences(f32_result,prior_result); cache_bits += differences(f32_result,cache_result);
        for (int wt = 0; wt < weight_tiles; ++wt) {
            require(prior.owners[wt] >= 0 && cache.owners[wt] >= 0,"all weight tiles owned");
            ownership_differences += prior.owners[wt] != cache.owners[wt];
        }
        for (auto* scratch : {&prior,&cache}) {
            require(std::all_of(scratch->visits.begin(),scratch->visits.end(),[](int visits){return visits == 1;}),"each actual callback worker visited once");
            scratch->audit = false;
        }
    };
    auto exact = [&] { return !f32_bits && !q8_bits && !prior_bits && !cache_bits && !ownership_differences &&
        !prior.quant_byte_differences && !cache.quant_byte_differences && !prior.panel_byte_differences && !cache.panel_byte_differences; };
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) for (auto& f : source) f *= -.5f;
        for (int c = 0; c < s.m; ++c) traits->from_float(source.data()+size_t(c)*s.k,quant.data()+size_t(c)*blocks,s.k);
        saved_source = source; saved_quant = quant;
        ggml_backend_tensor_set(x,source.data(),0,ggml_nbytes(x));
        ggml_backend_tensor_set(q,quant.data(),0,ggml_nbytes(q));
        parity();
    }
    std::array<double,3> us{};
    std::array<std::vector<double>,3> samples;
    const bool do_timing = timing && !std::strcmp(distribution,"normal") && exact();
    if (do_timing) {
        const int repeats = s.m >= 1500 ? 1 : s.k >= 1024 ? 2 : 10;
        const int indices[] = {0,2,3}; // prequantized graph is arithmetic-only
        for (int round = 0; round < 7; ++round) for (int step = 0; step < 3; ++step) {
            const int v = round % 2 ? (round + 2 - step) % 3 : (round + step) % 3;
            const auto begin = std::chrono::steady_clock::now();
            for (int repeat = 0; repeat < repeats; ++repeat) compute(indices[v]);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()*1e6/repeats;
            if (round) samples[v].push_back(elapsed);
        }
        for (int v = 0; v < 3; ++v) {
            std::sort(samples[v].begin(),samples[v].end()); us[v] = (samples[v][2]+samples[v][3])*.5;
        }
        parity();
    }
    std::printf("{\"threads\":%d,\"K\":%d,\"N\":%d,\"M\":%d,\"distribution\":\"%s\",\"inputUpdates\":2,\"ordinaryF32ToPinnedBitDifferences\":%zu,\"ordinaryQ8ToPinnedBitDifferences\":%zu,\"priorToF32BitDifferences\":%zu,\"cacheToF32BitDifferences\":%zu,\"workerOwnershipDifferences\":%zu,\"priorQuantByteDifferences\":%zu,\"cacheQuantByteDifferences\":%zu,\"priorPanelByteDifferences\":%zu,\"cachePanelByteDifferences\":%zu,\"actualWorkers\":%d,\"allWorkersVisitedOnce\":true,\"scratchCleaned\":true,\"inputQuantizationIncluded\":true,\"callbackAllocationCleanupIncluded\":true,\"nestedTeams\":false,\"fullInputLatency\":false,\"timed\":%s,\"ordinaryF32GraphUs\":%.3f,\"weightFirstF32GraphUs\":%.3f,\"cacheF32GraphUs\":%.3f,\"graphToCacheSpeedup\":%.4f,\"weightFirstToCacheSpeedup\":%.4f,\"sortedPrimarySamplesUs\":[",
        threads,s.k,s.n,s.m,distribution,f32_bits,q8_bits,prior_bits,cache_bits,ownership_differences,
        prior.quant_byte_differences,cache.quant_byte_differences,prior.panel_byte_differences,cache.panel_byte_differences,
        cache.observed_workers,do_timing ? "true" : "false",us[0],us[1],us[2],us[2] ? us[0]/us[2] : 0,us[2] ? us[1]/us[2] : 0);
    for (int v = 0; v < 3; ++v) {
        std::printf("%s[",v ? "," : "");
        for (size_t i = 0; i < samples[v].size(); ++i) std::printf("%s%.3f",i ? "," : "",samples[v][i]);
        std::printf("]");
    }
    std::printf("]}\n"); std::fflush(stdout);
    require(exact(),"exact ordinary F32 route, Q8/pinned control, custom outputs, quantized bytes, panels and ownership");
}
}
int main(int argc, char** argv) {
    ggml_backend_t backend = nullptr;
    try {
        const bool timing = argc == 2 && !std::strcmp(argv[1],"--benchmark");
        require(argc == 1 || timing,"usage: moss_vnni_q8_f32 [--benchmark]");
        ggml_cpu_init();
        const auto* t = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if (!ggml_cpu_has_avx512_vnni() || t->nrows != 1 || t->vec_dot_type != GGML_TYPE_Q8_0) return 77;
        half_conversion_audit();
        backend = ggml_backend_cpu_init(); require(backend,"CPU backend");
        const std::vector<Shape> shapes = {{32,17,1},{96,17,2},{32,31,7},{96,31,16},{96,17,31},
            {1024,128,16},{3072,128,16},{1024,1024,64},{1024,4096,64}};
        for (int threads : timing ? std::vector<int>{1,8,16} : std::vector<int>{1,16}) {
            ggml_backend_cpu_set_n_threads(backend,threads);
            for (auto s : shapes) for (const char* d : {"normal","zero","extreme","scales"}) trial(backend,threads,s,d,timing);
            if (!timing || threads == 16) for (auto s : std::vector<Shape>{{1024,1024,1500},{1024,4096,1500},{4096,1024,1500}})
                trial(backend,threads,s,"normal",timing);
        }
        ggml_backend_free(backend); return 0;
    } catch (const std::exception& e) {
        if (backend) ggml_backend_free(backend);
        std::fprintf(stderr,"moss_vnni_q8_f32: %s\n",e.what()); return 1;
    }
}
#else
int main() {
    std::fprintf(stderr,"Requires pinned AVX512 VNNI/BW/VL/F16C/OpenMP; no fallback timing\n"); return 77;
}
#endif
