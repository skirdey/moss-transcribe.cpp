// MIT. Exact parity and protected warm graph timing for decode-only fusion.
#include "cpu_projection.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
namespace {
void require(bool p,const char* why) {if(!p)throw std::runtime_error(why);}
struct Resources {
    ggml_context* ctx=nullptr;ggml_gallocr_t alloc=nullptr;ggml_backend_buffer_t buffer=nullptr;
    ~Resources(){if(alloc)ggml_gallocr_free(alloc);if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx);}
};
struct Shape {int k;std::vector<int> widths;};
void trial(ggml_backend_t backend,int threads,Shape s,const char* distribution,bool timing) {
    Resources leaves,ops;
    leaves.ctx=ggml_init({1024*1024,nullptr,true});ops.ctx=ggml_init({8*1024*1024,nullptr,true});
    require(leaves.ctx && ops.ctx,"contexts");
    auto x=ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_F32,s.k,1);
    std::vector<ggml_tensor*> weights;
    for(int n:s.widths)weights.push_back(ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_Q8_0,s.k,n));
    leaves.buffer=ggml_backend_alloc_ctx_tensors(leaves.ctx,backend);require(leaves.buffer,"external operands");
    std::vector<float> input(s.k),row(s.k);
    std::mt19937 random(20261010);std::normal_distribution<float> normal(0,1);
    for(auto& f:input)f=normal(random);
    if(std::strcmp(distribution,"normal")) {
        const float extreme[]={-1000.f,1000.f,-0.f,1e-12f,-1e-12f};
        for(int i=0;i<s.k;++i)input[i]=!std::strcmp(distribution,"zero") ? 0.f : extreme[i%5];
    }
    const auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    std::vector<std::vector<block_q8_0>> raw;
    for(auto w:weights) {
        raw.emplace_back(ggml_nbytes(w)/sizeof(block_q8_0));
        for(int64_t i=0;i<w->ne[1];++i) {
            for(auto& f:row)f=normal(random);
            quant(row.data(),raw.back().data()+i*s.k/32,s.k);
        }
        if(!std::strcmp(distribution,"extreme")) {
            const int8_t codes[]={-128,-127,0,127};
            for(size_t i=0;i<raw.back().size();++i) {
                raw.back()[i].d=ggml_fp32_to_fp16(1.f/(1+i%9));
                for(int j=0;j<32;++j)raw.back()[i].qs[j]=codes[(i+j)%4];
            }
        }
        ggml_backend_tensor_set(w,raw.back().data(),0,ggml_nbytes(w));
    }
    auto fused=mt::cpu_decode_q8_projections(ops.ctx,x,weights.data(),weights.size());
    require(fused && fused->op==GGML_OP_CUSTOM,"fusion selected");ggml_set_output(fused);
    int total=0;std::vector<ggml_tensor*> reference;
    auto rg=ggml_new_graph(ops.ctx),fg=ggml_new_graph(ops.ctx),all=ggml_new_graph(ops.ctx);
    for(auto w:weights) {
        total+=w->ne[1];auto y=ggml_mul_mat(ops.ctx,w,x);reference.push_back(y);ggml_set_output(y);
        ggml_build_forward_expand(rg,y);ggml_build_forward_expand(all,y);
    }
    require(fused->ne[0]==total && fused->src[0]==x,"concatenated output shape/input");
    for(size_t i=0;i<weights.size();++i)require(fused->src[i+1]==weights[i],"ordinary weight dependency");
    ggml_build_forward_expand(fg,fused);ggml_build_forward_expand(all,fused);
    ops.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(ops.alloc,all),"graph allocation");
    auto unchanged=[&] {
        std::vector<float> actual(s.k);ggml_backend_tensor_get(x,actual.data(),0,s.k*sizeof(float));
        require(!std::memcmp(actual.data(),input.data(),s.k*sizeof(float)),"input overwritten");
        for(size_t i=0;i<weights.size();++i) {
            std::vector<block_q8_0> w(raw[i].size());ggml_backend_tensor_get(weights[i],w.data(),0,ggml_nbytes(weights[i]));
            require(!std::memcmp(w.data(),raw[i].data(),ggml_nbytes(weights[i])),"weight overwritten");
        }
    };
    size_t different=0;
    auto parity=[&] {
        require(ggml_backend_graph_compute(backend,all)==GGML_STATUS_SUCCESS,"parity compute");unchanged();
        std::vector<float> joined(total);ggml_backend_tensor_get(fused,joined.data(),0,total*sizeof(float));
        size_t offset=0;
        for(auto y:reference) {
            std::vector<float> ref(ggml_nelements(y));ggml_backend_tensor_get(y,ref.data(),0,ref.size()*sizeof(float));
            for(size_t i=0;i<ref.size();++i) {
                require(std::isfinite(ref[i]) && std::isfinite(joined[offset+i]),"finite outputs");
                different+=std::memcmp(&ref[i],&joined[offset+i],sizeof(float))!=0;
            }
            offset+=ref.size();
        }
    };
    for(int pass=0;pass<2;++pass) {
        if(pass)for(auto& f:input)f*=-.5f;
        ggml_backend_tensor_set(x,input.data(),0,s.k*sizeof(float));parity();
    }
    std::array<double,2> us{};
    if(timing && !std::strcmp(distribution,"normal")) {
        std::array<std::vector<double>,2> samples;
        const int repeats=s.k==1024 ? 20 : 100;
        for(int round=0;round<6;++round)for(int step=0;step<2;++step) {
            const int v=(round+step)%2;const auto start=std::chrono::steady_clock::now();
            for(int i=0;i<repeats;++i)require(ggml_backend_graph_compute(backend,v ? fg : rg)==GGML_STATUS_SUCCESS,"timed compute");
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()*1e6/repeats;
            if(round)samples[v].push_back(elapsed);
        }
        for(int v=0;v<2;++v){std::sort(samples[v].begin(),samples[v].end());us[v]=samples[v][2];}
        parity();
    }
    auto bad=ggml_new_tensor_2d(ops.ctx,GGML_TYPE_F32,s.k,2);
    require(!mt::cpu_decode_q8_projections(ops.ctx,bad,weights.data(),weights.size()),"multirow fallback");
    require(!mt::cpu_decode_q8_projections(ops.ctx,x,weights.data(),1),"single consumer fallback");
    auto mixed=weights;mixed[0]=ggml_new_tensor_2d(ops.ctx,GGML_TYPE_F32,s.k,s.widths[0]);
    require(!mt::cpu_decode_q8_projections(ops.ctx,x,mixed.data(),mixed.size()),"F32 weight fallback");
    auto strided=weights;strided[0]=ggml_view_2d(ops.ctx,weights[0],s.k,s.widths[0]/2,weights[0]->nb[1]*2,0);
    require(!mt::cpu_decode_q8_projections(ops.ctx,x,strided.data(),strided.size()),"strided weight fallback");
    auto wide=ggml_new_tensor_2d(ops.ctx,GGML_TYPE_F32,4128,1);
    require(!mt::cpu_decode_q8_projections(ops.ctx,wide,weights.data(),weights.size()),"bounded worker scratch fallback");
    std::printf("{\"threads\":%d,\"K\":%d,\"outputRows\":%d,\"projections\":%zu,\"distribution\":\"%s\",\"inputUpdates\":2,\"floatBitDifferences\":%zu,\"referenceGraphUs\":%.3f,\"fusedGraphUs\":%.3f,\"speedup\":%.4f,\"extraWeightBytes\":0,\"fullInputLatency\":false}\n",threads,s.k,total,s.widths.size(),distribution,different,us[0],us[1],us[1] ? us[0]/us[1] : 0);
    std::fflush(stdout);require(!different,"exact pinned dot bits");
}
}
int main(int argc,char** argv) {
    try {
        const bool timing=argc==2 && !std::strcmp(argv[1],"--benchmark");
        require(argc==1 || timing,"usage: test_cpu_projection [--benchmark]");
        setenv("MTD_DEVICE","cpu",1);setenv("MTD_THREADS","16",1);
        auto backend=mt::backend();
        const auto* t=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if(t->nrows!=1 || t->vec_dot_type!=GGML_TYPE_Q8_0)return 77;
        for(int threads:timing ? std::vector<int>{1,8,16} : std::vector<int>{1,16}) {
            ggml_backend_cpu_set_n_threads(backend,threads);
            for(auto s:{Shape{32,{17,17,17}},Shape{96,{31,7,5}},Shape{1024,{2048,1024,1024}},Shape{1024,{3072,3072}}})
                for(const char* distribution:{"normal","zero","extreme"})trial(backend,threads,s,distribution,timing);
        }
        return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"projection probe: %s\n",e.what());return 1;}
}
