// MIT. Exercise the real Qwen layer builder, including its cache stores,
// causal prefill, RoPE, GQA, residuals and SwiGLU; no checkpoint/audio required.
#include "qwen3.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value,const char* why) { if(!value) throw std::runtime_error(why); }
struct Resources {
    ggml_context* ctx=nullptr;
    ggml_gallocr_t alloc=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ~Resources() {
        if(alloc)ggml_gallocr_free(alloc);
        if(buffer)ggml_backend_buffer_free(buffer);
        if(ctx)ggml_free(ctx);
    }
};
struct Weights {
    Resources r;
    mt::Qwen3Layer layer;
    std::vector<ggml_tensor*> tensors;
    std::vector<std::vector<unsigned char>> bytes;
    Weights(ggml_backend_t backend) {
        r.ctx=ggml_init({1024*1024,nullptr,true}); require(r.ctx,"weight context");
        auto norm=[&](ggml_tensor*& t,int n) {t=ggml_new_tensor_1d(r.ctx,GGML_TYPE_F32,n);tensors.push_back(t);};
        auto matrix=[&](ggml_tensor*& t,int k,int n) {t=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,k,n);tensors.push_back(t);};
        norm(layer.attn_norm,1024);norm(layer.q_norm,128);norm(layer.k_norm,128);norm(layer.ffn_norm,1024);
        matrix(layer.attn_q,1024,2048);matrix(layer.attn_k,1024,1024);matrix(layer.attn_v,1024,1024);
        matrix(layer.attn_o,2048,1024);matrix(layer.ffn_gate,1024,3072);matrix(layer.ffn_up,1024,3072);
        matrix(layer.ffn_down,3072,1024);
        r.buffer=ggml_backend_alloc_ctx_tensors(r.ctx,backend);require(r.buffer,"external weights");
        std::mt19937 random(20261010);std::normal_distribution<float> normal(0,.02f);
        const auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
        for(auto t:tensors) {
            bytes.emplace_back(ggml_nbytes(t));
            std::vector<float> row(t->ne[0]);
            if(t->type==GGML_TYPE_F32) {
                for(auto& x:row)x=1+normal(random);
                std::memcpy(bytes.back().data(),row.data(),bytes.back().size());
            } else for(int64_t i=0;i<t->ne[1];++i) {
                for(auto& x:row)x=normal(random);
                quant(row.data(),bytes.back().data()+i*t->nb[1],t->ne[0]);
            }
            ggml_backend_tensor_set(t,bytes.back().data(),0,bytes.back().size());
        }
    }
    void unchanged() const {
        for(size_t i=0;i<tensors.size();++i) {
            std::vector<unsigned char> actual(bytes[i].size());
            ggml_backend_tensor_get(tensors[i],actual.data(),0,actual.size());
            require(actual==bytes[i],"external weight overwritten");
        }
    }
};

void trial(ggml_backend_t backend,const Weights& w,int threads,int tokens,bool fused) {
    Resources graph_resources,cache_resources;
    graph_resources.ctx=ggml_init({16*1024*1024,nullptr,true});require(graph_resources.ctx,"graph context");
    cache_resources.ctx=ggml_init({1024*1024,nullptr,true});require(cache_resources.ctx,"cache context");
    auto ctx=graph_resources.ctx;
    const int past=tokens==1 ? 2 : 0, capacity=past+tokens+4;
    auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1024,tokens);
    auto pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,tokens);
    ggml_set_input(x);ggml_set_output(x);ggml_set_input(pos);ggml_set_output(pos);
    auto mask=tokens>1 ? ggml_new_tensor_2d(ctx,GGML_TYPE_F32,tokens,tokens) : nullptr;
    if(mask) {ggml_set_input(mask);ggml_set_output(mask);}
    std::array<ggml_tensor*,2> kc,vc,y;
    for(int v=0;v<2;++v) {
        kc[v]=ggml_new_tensor_4d(cache_resources.ctx,GGML_TYPE_F32,128,8,capacity,1);
        vc[v]=ggml_new_tensor_4d(cache_resources.ctx,GGML_TYPE_F32,capacity,128,8,1);
    }
    cache_resources.buffer=ggml_backend_alloc_ctx_tensors(cache_resources.ctx,backend);
    require(cache_resources.buffer,"persistent cache buffer");
    auto graph=ggml_new_graph_custom(ctx,2048,false);
    mt::Qwen3Hparams hp;
    hp.hidden=1024;hp.n_heads=16;hp.n_kv_heads=8;hp.head_dim=128;hp.intermediate=3072;
    for(int v=0;v<2;++v) {
        setenv("MTD_CPU_OPT",v ? (fused ? "32816" : "8240") : "48",1);
        auto out=mt::qwen3_layer_forward(ctx,x,pos,mask,nullptr,nullptr,w.layer,hp,graph,kc[v],vc[v],past);
        y[v]=out.y;ggml_set_output(y[v]);ggml_build_forward_expand(graph,y[v]);
    }
    int ordinary=0,shared=0,casts=0,fused_nodes=0,fused_consumers=0;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
        auto node=ggml_graph_node(graph,i);
        if(node->op==GGML_OP_CUSTOM && (node->src[1]==w.layer.attn_q || node->src[1]==w.layer.ffn_gate)) {
            ++fused_nodes;fused_consumers+=node->src[1]==w.layer.attn_q ? 3 : 2;
        }
        if(node->op!=GGML_OP_MUL_MAT)continue;
        const auto weight=node->src[0];
        if(weight!=w.layer.attn_q && weight!=w.layer.attn_k && weight!=w.layer.attn_v
            && weight!=w.layer.ffn_gate && weight!=w.layer.ffn_up)continue;
        if(node->src[1]->type==GGML_TYPE_F32)++ordinary;
        else {
            require(node->src[1]->type==GGML_TYPE_Q8_0,"projection type");++shared;
            casts+=node->src[1]->op==GGML_OP_CPY;
        }
    }
    if(fused)require(ordinary==(tokens==1 ? 5 : 10) && shared==0 && casts==0
        && fused_nodes==(tokens==1 ? 2 : 0) && fused_consumers==(tokens==1 ? 5 : 0),"actual fused layer routing/fallback");
    else require(ordinary==5 && shared==5 && casts==(tokens>=256 ? 5 : 0)
        && fused_nodes==0,"actual shared layer routing");
    graph_resources.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(graph_resources.alloc,graph),"layer allocation");
    std::vector<float> input(1024*tokens),causal(tokens*tokens),cache(128*8*capacity);
    std::vector<int32_t> positions(tokens);
    for(int i=0;i<tokens;++i)positions[i]=past+i;
    for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.031f);
    for(size_t i=0;i<cache.size();++i)cache[i]=std::cos(float(i)*.07f)*.1f;
    for(int q=0;q<tokens;++q)for(int k=0;k<tokens;++k)
        causal[q*tokens+k]=k>q ? -std::numeric_limits<float>::infinity() : 0;
    ggml_backend_tensor_set(pos,positions.data(),0,positions.size()*sizeof(int32_t));
    if(mask)ggml_backend_tensor_set(mask,causal.data(),0,causal.size()*sizeof(float));
    for(int update=0;update<2;++update) {
        if(update)for(auto& f:input)f*=-.5f;
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        for(int v=0;v<2;++v)for(auto t:{kc[v],vc[v]})ggml_backend_tensor_set(t,cache.data(),0,cache.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"layer compute");
        size_t different=0;
        for(auto pair:{std::array<ggml_tensor*,2>{y[0],y[1]},std::array<ggml_tensor*,2>{kc[0],kc[1]},std::array<ggml_tensor*,2>{vc[0],vc[1]}}) {
            std::vector<float> a(ggml_nelements(pair[0])),b(a.size());
            ggml_backend_tensor_get(pair[0],a.data(),0,a.size()*sizeof(float));
            ggml_backend_tensor_get(pair[1],b.data(),0,b.size()*sizeof(float));
            for(size_t i=0;i<a.size();++i) {
                require(std::isfinite(a[i]) && std::isfinite(b[i]),"finite layer/cache");
                different+=std::memcmp(&a[i],&b[i],sizeof(float))!=0;
            }
        }
        std::vector<float> unchanged(input.size());ggml_backend_tensor_get(x,unchanged.data(),0,unchanged.size()*sizeof(float));
        require(!std::memcmp(unchanged.data(),input.data(),input.size()*sizeof(float)),"layer input overwritten");w.unchanged();
        std::printf("{\"threads\":%d,\"tokens\":%d,\"pastTokens\":%d,\"update\":%d,\"sharedConsumers\":%d,\"castConsumers\":%d,\"fusedNodes\":%d,\"fusedConsumers\":%d,\"layerAndCacheBitDifferences\":%zu}\n",threads,tokens,past,update,shared,casts,fused_nodes,fused_consumers,different);
        std::fflush(stdout);require(different==0,"layer/cache bit identity");
    }
}
}

int main(int argc,char** argv) {
    try {
        const bool fused=argc==2 && !std::strcmp(argv[1],"--fused");
        require(argc==1 || fused,"usage: test_cpu_shared_layer [--fused]");
        setenv("MTD_DEVICE","cpu",1);setenv("MTD_THREADS","16",1);
        auto backend=mt::backend();require(ggml_backend_is_cpu(backend),"CPU backend");
        const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if(fused && (traits->nrows!=1 || traits->vec_dot_type!=GGML_TYPE_Q8_0))return 77;
        Weights weights(backend);
        for(int threads:{1,16}) {
            ggml_backend_cpu_set_n_threads(backend,threads);
            for(int tokens:{1,3,64,257})trial(backend,weights,threads,tokens,fused);
        }
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"test_cpu_shared_layer: %s\n",e.what());return 1;}
}
