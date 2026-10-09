// F32 cached-attention kernel benchmark against MOSS's eager graph.
// Covers noncontiguous GQA KV views and short/long decoder histories.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    auto backend=ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend,16);
    bool passed=true;
    for(int length : {1,31,256,1024,2048}) {
        const int dim=128,heads=16,kv_heads=8;
        ggml_init_params params{2*1024*1024,nullptr,true};
        auto ctx=ggml_init(params);
        auto q=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,heads,1,1);
        auto k=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,kv_heads,length,1);
        auto v=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,dim,kv_heads,length,1);
        ggml_set_input(q);ggml_set_input(k);ggml_set_input(v);
        auto qp=ggml_permute(ctx,q,0,2,1,3);
        auto kp=ggml_permute(ctx,k,0,2,1,3);
        auto vp=ggml_permute(ctx,v,0,2,1,3);
        auto scores=ggml_mul_mat(ctx,kp,qp);
        ggml_mul_mat_set_prec(scores,GGML_PREC_F32);
        auto weights=ggml_soft_max_ext(ctx,scores,nullptr,1.f/std::sqrt(float(dim)),0.f);
        auto vt=ggml_transpose(ctx,vp);
        if(!ggml_is_contiguous(vt)||ggml_is_transposed(vt)) vt=ggml_cont(ctx,vt);
        auto reference=ggml_mul_mat(ctx,vt,weights);
        reference=ggml_cont_2d(ctx,ggml_permute(ctx,reference,0,2,1,3),dim*heads,1);
        auto fused=ggml_flash_attn_ext(ctx,qp,kp,vp,nullptr,1.f/std::sqrt(float(dim)),0.f,0.f);
        ggml_flash_attn_ext_set_prec(fused,GGML_PREC_F32);
        ggml_set_output(reference);ggml_set_output(fused);
        auto graph=ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,reference);ggml_build_forward_expand(graph,fused);
        auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if(!ggml_gallocr_alloc_graph(alloc,graph)) return 2;
        std::mt19937 random(42);std::normal_distribution<float> values(0,1);
        for(auto tensor : {q,k,v}) {
            std::vector<float> data(ggml_nelements(tensor));
            for(auto & value : data) value=values(random);
            ggml_backend_tensor_set(tensor,data.data(),0,data.size()*sizeof(float));
        }
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS) return 2;
        std::vector<float> rv(dim*heads),fv(dim*heads);
        ggml_backend_tensor_get(reference,rv.data(),0,rv.size()*sizeof(float));
        ggml_backend_tensor_get(fused,fv.data(),0,fv.size()*sizeof(float));
        float max_abs=0;double squared=0;
        for(size_t i=0;i<rv.size();++i) {
            float delta=std::fabs(rv[i]-fv[i]);
            max_abs=std::max(max_abs,delta);squared+=double(delta)*delta;
            passed &= std::isfinite(fv[i]) && delta<=1.e-5f;
        }
        auto rg=ggml_new_graph(ctx);ggml_build_forward_expand(rg,reference);
        auto fg=ggml_new_graph(ctx);ggml_build_forward_expand(fg,fused);
        std::vector<double> rt,ft;
        for(int round=0;round<6;++round) for(int step=0;step<2;++step) {
            bool is_fused=(round+step)%2;
            auto begin=std::chrono::steady_clock::now();
            for(int i=0;i<100;++i)
                if(ggml_backend_graph_compute(backend,is_fused?fg:rg)!=GGML_STATUS_SUCCESS) return 2;
            auto elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count()/100;
            if(round) (is_fused?ft:rt).push_back(elapsed);
        }
        std::sort(rt.begin(),rt.end());std::sort(ft.begin(),ft.end());
        std::printf("{\"kvLength\":%d,\"maxAbsError\":%.9g,\"rmsError\":%.9g,\"referenceUs\":%.3f,\"fusedUs\":%.3f,\"speedup\":%.4f}\n",
                    length,max_abs,std::sqrt(squared/rv.size()),rt[2]*1e6,ft[2]*1e6,rt[2]/ft[2]);
        ggml_gallocr_free(alloc);ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return passed?0:1;
}
