// Graph-level exact-output gate, including activation conversion and ggml's
// persistent workers. Packing is excluded from graph timings, reported separately.
#include "cpu_q8.hpp"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main(int argc,char** argv) {
    const bool audit=argc==2 && !std::strcmp(argv[1],"--audit-unretained-input");
    if(argc>1 && !audit) return 2;
    if (!mt::cpu_exact_q8_supported()) return 77;
    auto backend = ggml_backend_cpu_init();
    const auto* traits = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    bool passed = true;
    for (int threads : {1, 16}) {
        ggml_backend_cpu_set_n_threads(backend, threads);
        for (const auto dims : {std::pair<int,int>{32,17}, {1024,1024}, {1024,2048},
                                {1024,3072}, {3072,1024}, {1024,151936}}) {
        for (const char* distribution : {"normal", "extreme", "zero"}) {
            const int k = dims.first, n = dims.second, blocks = k / 32;
            std::mt19937 random(20261009); std::normal_distribution<float> normal(0, 1);
            std::vector<block_q8_0> raw(static_cast<size_t>(blocks) * n);
            std::vector<float> row(k), input(k);
            for (int r = 0; r < n; ++r) {
                for (auto& f : row) f = normal(random);
                traits->from_float(row.data(), raw.data() + static_cast<size_t>(r) * blocks, k);
            }
            for (auto& f : input) f = normal(random);
            if (std::strcmp(distribution, "normal")) {
                const bool zero = !std::strcmp(distribution, "zero");
                const int8_t codes[] = {-128,-127,0,127};
                for (size_t b = 0; b < raw.size(); ++b) {
                    raw[b].d = ggml_fp32_to_fp16(zero ? 0.f : 1.f / (1 + b % 9));
                    for (int j = 0; j < 32; ++j) raw[b].qs[j] = zero ? 0 : codes[(b+j) % 4];
                }
                const float values[] = {-1000.f, 1000.f, -0.f, 1e-12f, -1e-12f};
                for (int j = 0; j < k; ++j) input[j] = zero ? 0.f : values[j % 5];
            }
            auto ctx = ggml_init({2*1024*1024, nullptr, true});
            auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, n);
            auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, 1);
            ggml_set_input(w); ggml_set_input(x);
            // INPUT allocates early but permits reuse after the last consumer.
            // Preserve these synthetic buffers across the timed subgraphs.
            if(!audit) { ggml_set_output(w); ggml_set_output(x); }
            auto start = std::chrono::steady_clock::now();
            auto pack = mt::cpu_pack_q8(w, raw.data());
            const double packing_us = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()*1e6;
            if (!pack || !pack->round_trip_exact(raw.data())) return 2;
            auto reference = ggml_mul_mat(ctx, w, x);
            auto q = mt::cpu_decode_quantize(ctx, x, pack.get());
            auto shared = mt::cpu_decode_mul_mat(ctx, w, x, pack.get(), q);
            auto separate = mt::cpu_decode_mul_mat(ctx, w, x, pack.get());
            if (shared->op != GGML_OP_MAP_CUSTOM2 || separate->op != GGML_OP_MAP_CUSTOM2) return 2;
            for (auto t : {reference, shared, separate, q}) ggml_set_output(t);
            auto graph = ggml_new_graph(ctx);
            for (auto t : {reference, shared, separate, q}) ggml_build_forward_expand(graph, t);
            auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (!ggml_gallocr_alloc_graph(alloc, graph)) return 2;
            ggml_backend_tensor_set(w, raw.data(), 0, raw.size()*sizeof(block_q8_0));
            ggml_backend_tensor_set(x, input.data(), 0, input.size()*sizeof(float));
            if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
            if(audit) {
                std::vector<float> after(input.size());
                ggml_backend_tensor_get(x,after.data(),0,after.size()*sizeof(float));
                const bool mutated=std::memcmp(input.data(),after.data(),input.size()*sizeof(float));
                std::printf("{\"auditUnretainedInput\":true,\"threads\":%d,\"K\":%d,\"N\":%d,\"inputMutatedAfterFirstGraph\":%s,\"timingRan\":false}\n",
                    threads,k,n,mutated ? "true":"false");
                ggml_gallocr_free(alloc);ggml_free(ctx);
                if(mutated) {ggml_backend_free(backend);return 0;}
                continue;
            }
            std::vector<float> a(n), b(n), c(n);
            ggml_backend_tensor_get(reference, a.data(), 0, a.size()*sizeof(float));
            ggml_backend_tensor_get(shared, b.data(), 0, b.size()*sizeof(float));
            ggml_backend_tensor_get(separate, c.data(), 0, c.size()*sizeof(float));
            std::vector<block_q8_0> expected(blocks), actual(blocks);
            traits->from_float(input.data(), expected.data(), k);
            ggml_backend_tensor_get(q, actual.data(), 0, actual.size()*sizeof(block_q8_0));
            const bool conversion_exact = !std::memcmp(expected.data(), actual.data(), actual.size()*sizeof(block_q8_0));
            size_t differences = 0;
            for (int r = 0; r < n; ++r) {
                differences += std::memcmp(&a[r], &b[r], sizeof(float)) != 0;
                differences += std::memcmp(&a[r], &c[r], sizeof(float)) != 0;
                passed &= std::isfinite(a[r]) && std::isfinite(b[r]) && std::isfinite(c[r]);
            }
            passed &= !differences && conversion_exact;
            double rt = 0, ct = 0;
            if (!std::strcmp(distribution, "normal")) {
                auto rg = ggml_new_graph(ctx); ggml_build_forward_expand(rg, reference);
                auto cg = ggml_new_graph(ctx); ggml_build_forward_expand(cg, shared);
                std::vector<double> rtimes, ctimes;
                for (int round = 0; round < 6; ++round) for (int step = 0; step < 2; ++step) {
                    const bool optimized = (round+step) % 2;
                    auto begin = std::chrono::steady_clock::now();
                    for (int repeat = 0; repeat < 20; ++repeat)
                        if (ggml_backend_graph_compute(backend, optimized ? cg : rg) != GGML_STATUS_SUCCESS) return 2;
                    const double us = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()*1e6/20;
                    if (round) (optimized ? ctimes : rtimes).push_back(us);
                }
                std::sort(rtimes.begin(), rtimes.end()); std::sort(ctimes.begin(), ctimes.end());
                rt = rtimes[2]; ct = ctimes[2];
            }
            // Check the operands again after repeated graph execution, and
            // recompute all variants with the same bytes. A single initial
            // output comparison cannot validate a mutated timed input.
            std::vector<float> input_after(input.size());
            std::vector<block_q8_0> weight_after(raw.size());
            ggml_backend_tensor_get(x,input_after.data(),0,input_after.size()*sizeof(float));
            ggml_backend_tensor_get(w,weight_after.data(),0,weight_after.size()*sizeof(block_q8_0));
            const bool operands_exact=!std::memcmp(input.data(),input_after.data(),input.size()*sizeof(float))
                && !std::memcmp(raw.data(),weight_after.data(),raw.size()*sizeof(block_q8_0));
            passed &= operands_exact;
            if (ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 2;
            ggml_backend_tensor_get(reference,a.data(),0,a.size()*sizeof(float));
            ggml_backend_tensor_get(shared,b.data(),0,b.size()*sizeof(float));
            ggml_backend_tensor_get(separate,c.data(),0,c.size()*sizeof(float));
            for(int row=0;row<n;++row) {
                passed &= std::isfinite(a[row]) && std::isfinite(b[row]) && std::isfinite(c[row]);
                passed &= !std::memcmp(&a[row],&b[row],sizeof(float)) && !std::memcmp(&a[row],&c[row],sizeof(float));
            }
            std::printf("{\"threads\":%d,\"K\":%d,\"N\":%d,\"M\":1,\"distribution\":\"%s\",\"activationQuantizationExact\":%s,\"floatBitDifferences\":%zu,\"packingUs\":%.3f,\"referenceGraphUs\":%.3f,\"candidateGraphUs\":%.3f,\"speedup\":%.4f}\n",
                threads,k,n,distribution,conversion_exact ? "true" : "false",differences,packing_us,rt,ct,ct ? rt/ct : 0);
            if(!operands_exact)std::fprintf(stderr,"Q8 graph operands mutated after warm timing\n");
            ggml_gallocr_free(alloc); ggml_free(ctx);
        }
        }
        // Prefill remains the same eager op, with ordinary weights.
        auto ctx = ggml_init({2*1024*1024, nullptr, true});
        auto w = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, 32, 17);
        auto x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 32, 16);
        std::vector<block_q8_0> zeros(17);
        auto pack = mt::cpu_pack_q8(w, zeros.data());
        passed &= mt::cpu_decode_quantize(ctx, x, pack.get()) == nullptr;
        passed &= mt::cpu_decode_mul_mat(ctx, w, x, pack.get())->op == GGML_OP_MUL_MAT;
        passed &= mt::cpu_decode_mul_mat(ctx, w, x, nullptr)->op == GGML_OP_MUL_MAT;
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return !audit && passed ? 0 : 1;
}
