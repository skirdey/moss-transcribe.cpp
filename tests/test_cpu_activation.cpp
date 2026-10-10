// MIT. Compare shared conversion plus ordinary matmul with the pinned CPU
// path. --benchmark includes conversion and dispatch; weights are synthetic.
#include "cpu_activation.hpp"
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

namespace {
struct Shape { int k, n, m, projections; bool strided; int secondary_n = 0; };
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Resources {
    ggml_context* ctx = nullptr;
    ggml_gallocr_t alloc = nullptr;
    ~Resources() { if (alloc) ggml_gallocr_free(alloc); if (ctx) ggml_free(ctx); }
};
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end()); return values[values.size()/2];
}

bool trial(ggml_backend_t backend, int threads, Shape s, const char* distribution, bool timing) {
    Resources r;
    r.ctx=ggml_init({8*1024*1024,nullptr,true}); require(r.ctx,"context");
    const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    const int blocks=s.k/32, stride=s.k+(s.strided ? 32 : 0);
    auto source=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,stride,s.m);
    ggml_set_input(source);
    // INPUT controls initial allocation, not retention after its last use.
    // Keep reusable synthetic inputs alive across all variant graph executions,
    // as externally allocated model weights/input buffers would be.
    ggml_set_output(source);
    auto x=s.strided ? ggml_view_2d(r.ctx,source,s.k,s.m,stride*sizeof(float),0) : source;
    auto cast=ggml_cast(r.ctx,x,GGML_TYPE_Q8_0);
    auto shared=mt::cpu_shared_q8_activation(r.ctx,x); require(shared,"shared conversion");
    require(shared->op==GGML_OP_CUSTOM && shared->src[0]==x,"custom dependency");
    for (auto q:{cast,shared}) ggml_set_output(q);
    std::array<ggml_cgraph*,3> graphs;
    for (auto& g:graphs) g=ggml_new_graph(r.ctx);
    std::array<std::vector<ggml_tensor*>,3> outputs;
    std::vector<ggml_tensor*> weights;
    std::vector<std::vector<block_q8_0>> raw;
    std::mt19937 rng(20261009);std::normal_distribution<float> normal(0,1);
    std::vector<float> row(s.k),input(static_cast<size_t>(stride)*s.m,-9876.f);
    for(int i=0;i<s.m;++i)for(int j=0;j<s.k;++j) input[i*stride+j]=normal(rng);
    if(std::strcmp(distribution,"normal")) {
        const float extremes[]={-1000.f,1000.f,-0.f,1e-12f,-1e-12f};
        for(int i=0;i<s.m;++i)for(int j=0;j<s.k;++j)
            input[i*stride+j]=!std::strcmp(distribution,"zero") ? 0.f : extremes[(i+j)%5];
    }
    for(int p=0;p<s.projections;++p) {
        const int out_n=p && s.secondary_n ? s.secondary_n : s.n;
        auto w=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,s.k,out_n);ggml_set_input(w);
        ggml_set_output(w);
        weights.push_back(w);raw.emplace_back(static_cast<size_t>(blocks)*out_n);
        for(int i=0;i<out_n;++i) {
            for(auto& f:row)f=normal(rng);
            traits->from_float(row.data(),raw.back().data()+i*blocks,s.k);
        }
        if(!std::strcmp(distribution,"extreme")) {
            const int8_t codes[]={-128,-127,0,127};
            for(size_t b=0;b<raw.back().size();++b) {
                raw.back()[b].d=ggml_fp32_to_fp16(1.f/(1+(b+p)%9));
                for(int j=0;j<32;++j)raw.back()[b].qs[j]=codes[(b+j+p)%4];
            }
        }
        for(int v=0;v<3;++v) {
            auto y=ggml_mul_mat(r.ctx,w,v==0 ? x : v==1 ? cast : shared);
            require(y->op==GGML_OP_MUL_MAT,"ordinary matrix arithmetic");
            ggml_set_output(y);outputs[v].push_back(y);ggml_build_forward_expand(graphs[v],y);
        }
    }
    auto all=ggml_new_graph(r.ctx);
    for(auto& variant:outputs)for(auto y:variant)ggml_build_forward_expand(all,y);
    r.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(r.alloc && ggml_gallocr_alloc_graph(r.alloc,all),"graph allocation");
    for(int p=0;p<s.projections;++p)ggml_backend_tensor_set(weights[p],raw[p].data(),0,raw[p].size()*sizeof(block_q8_0));
    auto immutable_inputs=[&] {
        std::vector<float> copied_input(input.size());
        ggml_backend_tensor_get(source,copied_input.data(),0,copied_input.size()*sizeof(float));
        require(!std::memcmp(input.data(),copied_input.data(),input.size()*sizeof(float)),"input overwritten");
        for(int p=0;p<s.projections;++p) {
            std::vector<block_q8_0> copied_weight(raw[p].size());
            ggml_backend_tensor_get(weights[p],copied_weight.data(),0,copied_weight.size()*sizeof(block_q8_0));
            require(!std::memcmp(raw[p].data(),copied_weight.data(),copied_weight.size()*sizeof(block_q8_0)),"weight overwritten");
        }
    };
    size_t bit_diff_cast=0,bit_diff_shared=0,quant_diff=0;
    float max_abs=0;
    // Recompute after changing inputs to detect stale shared storage or a
    // missing dependency. Timing, when requested, uses this second input.
    for(int pass=0;pass<2;++pass) {
        if(pass)for(int i=0;i<s.m;++i)for(int j=0;j<s.k;++j)input[i*stride+j]*=-.5f;
        ggml_backend_tensor_set(source,input.data(),0,input.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend,all)==GGML_STATUS_SUCCESS,"compute");
        immutable_inputs();
        std::vector<block_q8_0> expected(static_cast<size_t>(blocks)*s.m),got(expected.size());
        for(int i=0;i<s.m;++i) {
            // The actual eager matmul splits each row's independent blocks
            // across workers; compare those exact conversion calls.
            for(int t=0;t<threads;++t) {
                const int first=t*blocks/threads,last=(t+1)*blocks/threads;
                traits->from_float(input.data()+i*stride+first*32,
                    expected.data()+i*blocks+first,(last-first)*32);
            }
        }
        for(auto q:{cast,shared}) {
            ggml_backend_tensor_get(q,got.data(),0,got.size()*sizeof(block_q8_0));
            const auto* a=reinterpret_cast<const unsigned char*>(expected.data());
            const auto* b=reinterpret_cast<const unsigned char*>(got.data());
            for(size_t i=0;i<got.size()*sizeof(block_q8_0);++i)quant_diff+=a[i]!=b[i];
        }
        for(int p=0;p<s.projections;++p) {
            std::vector<float> a(ggml_nelements(outputs[0][p])),b(a.size()),c(a.size());
            ggml_backend_tensor_get(outputs[0][p],a.data(),0,a.size()*sizeof(float));
            ggml_backend_tensor_get(outputs[1][p],b.data(),0,b.size()*sizeof(float));
            ggml_backend_tensor_get(outputs[2][p],c.data(),0,c.size()*sizeof(float));
            for(size_t i=0;i<a.size();++i) {
                if(!std::isfinite(a[i]) || !std::isfinite(b[i]) || !std::isfinite(c[i])) {
                    std::fprintf(stderr,"nonfinite: K=%d N=%d M=%d threads=%d pass=%d projection=%d index=%zu eager=%g cast=%g shared=%g\n",
                        s.k,s.n,s.m,threads,pass,p,i,a[i],b[i],c[i]);
                    throw std::runtime_error("finite output");
                }
                bit_diff_cast+=std::memcmp(&a[i],&b[i],sizeof(float))!=0;
                bit_diff_shared+=std::memcmp(&a[i],&c[i],sizeof(float))!=0;
                max_abs=std::max(max_abs,std::max(std::fabs(a[i]-b[i]),std::fabs(a[i]-c[i])));
            }
        }
    }
    std::array<double,3> us{};
    if(timing) {
        std::array<std::vector<double>,3> samples;
        const int repeats=s.m>=64 ? 1 : s.n*s.k<100000 ? 10 : 3;
        for(int round=0;round<6;++round)for(int step=0;step<3;++step) {
            const int v=(round+step)%3;
            auto start=std::chrono::steady_clock::now();
            for(int i=0;i<repeats;++i)require(ggml_backend_graph_compute(backend,graphs[v])==GGML_STATUS_SUCCESS,"timed compute");
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()*1e6/repeats;
            if(round)samples[v].push_back(elapsed);
        }
        for(int v=0;v<3;++v)us[v]=median(samples[v]);
        immutable_inputs();
    }
    std::printf("{\"threads\":%d,\"K\":%d,\"N\":%d,\"secondaryN\":%d,\"M\":%d,\"projections\":%d,\"stridedInput\":%s,\"distribution\":\"%s\",\"inputUpdates\":2,\"activationByteDifferences\":%zu,\"castFloatBitDifferences\":%zu,\"sharedFloatBitDifferences\":%zu,\"maxAbsError\":%.9g,\"referenceGraphUs\":%.3f,\"castGraphUs\":%.3f,\"sharedGraphUs\":%.3f,\"sharedSpeedup\":%.4f,\"extraWeightBytes\":0,\"fullInputLatency\":false}\n",
        threads,s.k,s.n,s.secondary_n,s.m,s.projections,s.strided ? "true":"false",distribution,quant_diff,
        bit_diff_cast,bit_diff_shared,max_abs,us[0],us[1],us[2],us[2] ? us[0]/us[2] : 0);
    std::fflush(stdout);
    return !quant_diff && !bit_diff_cast && !bit_diff_shared;
}

bool batch_conversion(ggml_backend_t backend) {
    Resources r;r.ctx=ggml_init({1024*1024,nullptr,true});require(r.ctx,"batch context");
    auto x=ggml_new_tensor_4d(r.ctx,GGML_TYPE_F32,96,3,2,2);ggml_set_input(x);
    auto q=mt::cpu_shared_q8_activation(r.ctx,x);ggml_set_output(q);
    auto graph=ggml_new_graph(r.ctx);ggml_build_forward_expand(graph,q);
    r.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ggml_gallocr_alloc_graph(r.alloc,graph),"batch allocation");
    std::vector<float> input(ggml_nelements(x));for(size_t i=0;i<input.size();++i)input[i]=std::sin(float(i));
    std::vector<block_q8_0> expected(36),actual(36);
    const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    for(int row=0;row<12;++row)traits->from_float(input.data()+row*96,expected.data()+row*3,96);
    ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"batch compute");
    ggml_backend_tensor_get(q,actual.data(),0,actual.size()*sizeof(block_q8_0));
    require(mt::cpu_shared_q8_activation(r.ctx,nullptr)==nullptr,"null rejection");
    require(mt::cpu_shared_q8_activation(r.ctx,ggml_new_tensor_1d(r.ctx,GGML_TYPE_F32,33))==nullptr,"unaligned rejection");
    require(mt::cpu_shared_q8_activation(r.ctx,ggml_new_tensor_1d(r.ctx,GGML_TYPE_F16,32))==nullptr,"type rejection");
    auto transposed=ggml_transpose(r.ctx,x);
    require(mt::cpu_shared_q8_activation(r.ctx,transposed)==nullptr,"scalar stride rejection");
    return !std::memcmp(expected.data(),actual.data(),actual.size()*sizeof(block_q8_0));
}
}

int main(int argc,char** argv) {
    try {
        const bool benchmark=argc==2 && !std::strcmp(argv[1],"--benchmark");
        if(argc>1 && !benchmark)throw std::runtime_error("usage: test_cpu_activation [--benchmark]");
        auto backend=ggml_backend_cpu_init();require(backend,"CPU backend");
        bool passed=true;
        std::vector<Shape> shapes={{32,17,1,3,false},{1024,1024,1,3,false},
            {1024,2048,1,3,false,1024},{1024,3072,3,2,true},{1280,1280,16,3,false}};
        if(benchmark) {
            shapes.push_back({1024,2048,64,3,false,1024});
            shapes.push_back({1024,3072,128,2,false});
            shapes.push_back({1024,1024,1500,3,false});
        }
        for(int threads:benchmark ? std::vector<int>{1,8,16} : std::vector<int>{1,16}) {
            ggml_backend_cpu_set_n_threads(backend,threads);
            passed &= batch_conversion(backend);
            for(auto shape:shapes)for(const char* distribution:{"normal","zero","extreme"})
                passed &= trial(backend,threads,shape,distribution,benchmark && !std::strcmp(distribution,"normal"));
        }
        ggml_backend_free(backend);return passed ? 0:1;
    } catch(const std::exception& e) {std::fprintf(stderr,"activation probe: %s\n",e.what());return 2;}
}
