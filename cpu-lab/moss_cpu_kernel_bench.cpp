// Native kernel microbenchmark and numerical regression test.
// Link against the same native ggml libraries as the candidate.
#include "ggml.h"
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
    ggml_backend_cpu_set_n_threads(backend, 16);
    bool passed = true;
    for (auto shape : {std::pair<int,int>{3072,1}, {3073,1}, {3072,750}}) {
        const int n = shape.first * shape.second;
        ggml_init_params params{2*1024*1024,nullptr,true};
        auto ctx = ggml_init(params);
        auto a = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,shape.first,shape.second);
        auto b = ggml_new_tensor_2d(ctx,GGML_TYPE_F32,shape.first,shape.second);
        ggml_set_input(a); ggml_set_input(b);
        auto reference = ggml_mul(ctx,ggml_silu(ctx,a),b);
        auto fused = ggml_swiglu_split(ctx,a,b);
        ggml_set_output(reference); ggml_set_output(fused);
        auto graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,reference);
        ggml_build_forward_expand(graph,fused);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(alloc,graph)) return 2;
        std::mt19937 random(42);
        std::uniform_real_distribution<float> values(-20,20);
        std::vector<float> av(n),bv(n),rv(n),fv(n);
        for(int i=0;i<n;++i) { av[i]=values(random); bv[i]=values(random); }
        // Edge values exercise saturation, negative zero and tiny inputs.
        av[0]=0;av[1]=-0.f;av[2]=-100;av[3]=100;av[4]=1.e-20f;
        ggml_backend_tensor_set(a,av.data(),0,n*sizeof(float));
        ggml_backend_tensor_set(b,bv.data(),0,n*sizeof(float));
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 2;
        ggml_backend_tensor_get(reference,rv.data(),0,n*sizeof(float));
        ggml_backend_tensor_get(fused,fv.data(),0,n*sizeof(float));
        float max_abs=0;
        int differing=0;
        for(int i=0;i<n;++i) {
            max_abs=std::max(max_abs,std::fabs(rv[i]-fv[i]));
            if(std::memcmp(&rv[i],&fv[i],sizeof(float))) ++differing;
        }
        passed &= differing==0;
        auto rg = ggml_new_graph(ctx); ggml_build_forward_expand(rg,reference);
        auto fg = ggml_new_graph(ctx); ggml_build_forward_expand(fg,fused);
        int repeats = shape.second==1 ? 1000 : 50;
        std::vector<double> rtimes,ftimes;
        for(int round=0;round<6;++round) {
            // Alternate order and discard first pair as warmup.
            for(int step=0;step<2;++step) {
                bool is_fused=(step+round)%2;
                auto begin=std::chrono::steady_clock::now();
                for(int i=0;i<repeats;++i)
                    if(ggml_backend_graph_compute(backend,is_fused?fg:rg)!=GGML_STATUS_SUCCESS) return 2;
                double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()/repeats;
                if(round) (is_fused?ftimes:rtimes).push_back(elapsed);
            }
        }
        std::sort(rtimes.begin(),rtimes.end());std::sort(ftimes.begin(),ftimes.end());
        std::printf("{\"width\":%d,\"rows\":%d,\"bitwiseDifferent\":%d,\"maxAbsError\":%.9g,\"referenceUs\":%.3f,\"fusedUs\":%.3f,\"speedup\":%.4f}\n",
                    shape.first,shape.second,differing,max_abs,rtimes[2]*1e6,ftimes[2]*1e6,rtimes[2]/ftimes[2]);
        ggml_gallocr_free(alloc); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return passed ? 0 : 1;
}
