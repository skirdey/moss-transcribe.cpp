// Test/benchmark an appendable transposed F32 V cache with reference arithmetic.
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
    auto backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,16);
    bool passed=true;
    for(int length : {1,31,256,1024,2048}) {
        const int dim=128,heads=16,kv_heads=8,capacity=4096;
        ggml_init_params params{2*1024*1024,nullptr,true};
        auto ctx=ggml_init(params);
        auto v=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,kv_heads,length,1);
        auto weights=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,length,1,heads,1);
        auto cache=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,capacity,dim,kv_heads,1);
        ggml_set_input(v);ggml_set_input(weights);ggml_set_input(cache);
        auto vp=ggml_permute(ctx,v,0,2,1,3);
        auto vt=ggml_transpose(ctx,vp);
        if(!ggml_is_contiguous(vt)||ggml_is_transposed(vt)) vt=ggml_cont(ctx,vt);
        auto reference=ggml_mul_mat(ctx,vt,weights);
        auto cache_view=ggml_view_4d(ctx,cache,length,dim,kv_heads,1,cache->nb[1],cache->nb[2],cache->nb[3],0);
        auto cached=ggml_mul_mat(ctx,cache_view,weights);
        ggml_set_output(reference);ggml_set_output(cached);
        auto graph=ggml_new_graph(ctx);
        // Write a prefix, then append the last token through a separate view.
        if(length>1) {
            auto prefix=ggml_view_4d(ctx,v,dim,kv_heads,length-1,1,v->nb[1],v->nb[2],v->nb[3],0);
            auto dst=ggml_view_4d(ctx,cache,length-1,dim,kv_heads,1,cache->nb[1],cache->nb[2],cache->nb[3],0);
            ggml_build_forward_expand(graph,ggml_cpy(ctx,ggml_permute(ctx,prefix,1,2,0,3),dst));
        }
        auto last=ggml_view_4d(ctx,v,dim,kv_heads,1,1,v->nb[1],v->nb[2],v->nb[3],(length-1)*v->nb[2]);
        auto dst=ggml_view_4d(ctx,cache,1,dim,kv_heads,1,cache->nb[1],cache->nb[2],cache->nb[3],(length-1)*sizeof(float));
        auto append=ggml_cpy(ctx,ggml_permute(ctx,last,1,2,0,3),dst);
        ggml_build_forward_expand(graph,append);
        ggml_build_forward_expand(graph,reference);ggml_build_forward_expand(graph,cached);
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if(!ggml_gallocr_alloc_graph(alloc,graph)) return 2;
        std::mt19937 random(42);std::normal_distribution<float> values(0,1);
        for(auto tensor : {v,weights}) {
            std::vector<float> data(ggml_nelements(tensor));
            for(auto & value : data) value=values(random);
            ggml_backend_tensor_set(tensor,data.data(),0,data.size()*sizeof(float));
        }
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 2;
        std::vector<float> rv(dim*heads),fv(dim*heads);
        ggml_backend_tensor_get(reference,rv.data(),0,rv.size()*sizeof(float));
        ggml_backend_tensor_get(cached,fv.data(),0,fv.size()*sizeof(float));
        int different=0;float max_abs=0;
        for(size_t i=0;i<rv.size();++i) {
            different+=std::memcmp(&rv[i],&fv[i],sizeof(float))!=0;
            max_abs=std::max(max_abs,std::fabs(rv[i]-fv[i]));
        }
        passed &= different==0;
        auto rg=ggml_new_graph(ctx);ggml_build_forward_expand(rg,reference);
        auto fg=ggml_new_graph(ctx);ggml_build_forward_expand(fg,append);ggml_build_forward_expand(fg,cached);
        std::vector<double> rt,ft;
        for(int round=0;round<6;++round) for(int step=0;step<2;++step) {
            bool optimized=(round+step)%2;
            auto begin=std::chrono::steady_clock::now();
            for(int i=0;i<100;++i)
                if(ggml_backend_graph_compute(backend,optimized?fg:rg)!=GGML_STATUS_SUCCESS) return 2;
            auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()/100;
            if(round) (optimized?ft:rt).push_back(elapsed);
        }
        std::sort(rt.begin(),rt.end());std::sort(ft.begin(),ft.end());
        std::printf("{\"kvLength\":%d,\"bitwiseDifferent\":%d,\"maxAbsError\":%.9g,\"referenceUs\":%.3f,\"cachedUs\":%.3f,\"speedup\":%.4f}\n",
                    length,different,max_abs,rt[2]*1e6,ft[2]*1e6,rt[2]/ft[2]);
        ggml_gallocr_free(alloc);ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return passed?0:1;
}
