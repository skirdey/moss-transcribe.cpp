// Numerical parity and paired kernel timings against pinned eager CPU softmax.
#include "cpu_softmax.hpp"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    auto backend = ggml_backend_cpu_init();
    bool passed = true;
    for (int threads : {1, 16}) {
    ggml_backend_cpu_set_n_threads(backend, threads);
    for (int length : {1, 2, 7, 31, 256, 1024, 2048, 4095, 4096}) {
    for (int shape = 0; shape < 3; ++shape) {
        const int heads = shape == 0 ? 1 : shape == 1 ? 16 : 17;
        const int batches = shape == 2 ? 2 : 1;
        const int dim = shape == 2 ? 96 : 128;
        ggml_init_params params{2*1024*1024, nullptr, true};
        auto ctx = ggml_init(params);
        auto scores = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, length, 1, heads, batches);
        auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, dim, 1, heads, batches);
        ggml_set_input(scores); ggml_set_input(q);
        auto reference = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f/std::sqrt(float(dim)), 0.0f);
        auto parallel = mt::cpu_decode_softmax(ctx, scores, q);
        ggml_set_output(reference); ggml_set_output(parallel);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, reference); ggml_build_forward_expand(graph, parallel);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc, graph)) return 2;
        std::mt19937 random(42); std::normal_distribution<float> values(0, 12);
        std::vector<float> data(ggml_nelements(scores));
        for (auto& x : data) x = values(random);
        // Overflow/underflow-sensitive rows and vector tails, without an all -inf row.
        if (length > 1) {
            for (int row = 0; row < heads*batches; ++row) {
                if (row%3 == 0) data[row*length] = 10000.0f;
                if (row%3 == 1) data[row*length] = -INFINITY;
                if (row%3 == 2) data[row*length] = -10000.0f;
            }
        }
        ggml_backend_tensor_set(scores, data.data(), 0, data.size()*sizeof(float));
        ggml_backend_tensor_memset(q, 0, 0, ggml_nbytes(q));
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return 2;
        std::vector<float> a(data.size()), b(data.size());
        ggml_backend_tensor_get(reference, a.data(), 0, a.size()*sizeof(float));
        ggml_backend_tensor_get(parallel, b.data(), 0, b.size()*sizeof(float));
        int different = 0; float max_abs = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            different += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
            max_abs = std::max(max_abs, std::fabs(a[i]-b[i]));
            passed &= std::isfinite(a[i]) && std::isfinite(b[i]);
        }
        passed &= different == 0;
        double rt = 0, ft = 0;
        if (threads == 16 && heads == 16 && (length == 1 || length == 31 || length == 256 || length == 1024 || length == 2048)) {
            auto rg = ggml_new_graph(ctx); ggml_build_forward_expand(rg, reference);
            auto fg = ggml_new_graph(ctx); ggml_build_forward_expand(fg, parallel);
            std::vector<double> rtimes, ftimes;
            for (int round = 0; round < 6; ++round) for (int step = 0; step < 2; ++step) {
                bool opt = (round+step)%2;
                auto begin = std::chrono::steady_clock::now();
                for (int i = 0; i < 100; ++i)
                    if (ggml_backend_graph_compute(backend, opt ? fg : rg) != GGML_STATUS_SUCCESS) return 2;
                double us = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()*1e4;
                if (round) (opt ? ftimes : rtimes).push_back(us);
            }
            std::sort(rtimes.begin(), rtimes.end()); std::sort(ftimes.begin(), ftimes.end());
            rt = rtimes[2]; ft = ftimes[2];
        }
        std::printf("{\"threads\":%d,\"kvLength\":%d,\"heads\":%d,\"batches\":%d,\"headDim\":%d,\"bitwiseDifferent\":%d,\"maxAbsError\":%.9g,\"referenceUs\":%.3f,\"parallelUs\":%.3f,\"speedup\":%.4f}\n",
                    threads,length,heads,batches,dim,different,max_abs,rt,ft,ft ? rt/ft : 0);
        ggml_gallocr_free(alloc); ggml_free(ctx);
    }
    }
    }
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
