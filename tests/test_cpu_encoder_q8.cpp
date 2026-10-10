// MIT. Protected graph operands, changed inputs, measured shapes and fallbacks.
#include "cpu_encoder_q8.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }
void env(const char* k,const char* v) {
#ifdef _WIN32
    _putenv_s(k,v);
#else
    setenv(k,v,1);
#endif
}
struct Resources {
    ggml_context* ctx=nullptr;
    ggml_backend_buffer_t buffer=nullptr;
    ggml_gallocr_t alloc=nullptr;
    ~Resources() { if (alloc)ggml_gallocr_free(alloc);if (buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx); }
};
void fallback(ggml_backend_t backend) {
    Resources r;r.ctx=ggml_init({2*1024*1024,nullptr,true});require(r.ctx,"context");
    mt::CpuEncoderQ8 owner;
    auto* w=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,1024,1024);
    auto* x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,1024,1500);
    require(!owner.mul_mat(nullptr,w,x) && !owner.mul_mat(r.ctx,nullptr,x)
        && !owner.mul_mat(r.ctx,w,nullptr),"null fallbacks");
    auto* short_x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,1024,64);
    auto* narrow_w=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,1024,128);
    auto* float_w=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,1024,1024);
    auto* q8_x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,1024,1500);
    auto* large_x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,1056,1500);
    auto* strided=ggml_view_2d(r.ctx,large_x,1024,1500,1056*sizeof(float),0);
    require(!owner.mul_mat(r.ctx,w,short_x) && !owner.mul_mat(r.ctx,narrow_w,x)
        && !owner.mul_mat(r.ctx,float_w,x) && !owner.mul_mat(r.ctx,w,q8_x)
        && !owner.mul_mat(r.ctx,w,strided),"shape/type/layout fallbacks");
    { mt::CpuThreadScope fewer(mt::CpuThreadPhase::Decode);
        require(mt::cpu_thread_count()==8 && !owner.mul_mat(r.ctx,w,x),"8-worker fallback"); }
    require(mt::cpu_thread_count()==16,"thread restore");
    if (!mt::CpuEncoderQ8::supported()) require(!owner.mul_mat(r.ctx,w,x),"unsupported ISA fallback");
    require(owner.nodes()==0 && owner.executions()==0 && owner.ok(),"fallback creates no contexts");
    (void)backend;
}
void trial(ggml_backend_t backend,int k,int n,const char* distribution) {
    constexpr int m=1500;const int blocks=k/32;
    std::mt19937 rng(20261010);std::normal_distribution<float> normal(0,1);
    std::vector<float> input(size_t(k)*m),row(k);
    std::vector<block_q8_0> weights(size_t(blocks)*n);
    const auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for(auto& f:input)f=normal(rng);
    const float edges[]={-1000.f,1000.f,-0.f,1e-12f,-1e-12f,.5f,-.5f};
    if (std::strcmp(distribution,"normal"))for(size_t i=0;i<input.size();++i)
        input[i]=!std::strcmp(distribution,"zero") ? 0.f : edges[i%7];
    for(int r=0;r<n;++r) { for(auto& f:row)f=normal(rng);quant(row.data(),weights.data()+size_t(r)*blocks,k); }
    if (!std::strcmp(distribution,"extreme") || !std::strcmp(distribution,"scales")) {
        const int8_t codes[]={-128,-127,0,127};
        const std::array<ggml_half,8> scales={0,0x8000,1,0x8001,0x0400,0x8400,0x7bff,0xfbff};
        for(size_t b=0;b<weights.size();++b) {
            weights[b].d=!std::strcmp(distribution,"scales") ? scales[b%8] : ggml_fp32_to_fp16(1.f/(1+b%9));
            for(int j=0;j<32;++j)weights[b].qs[j]=codes[(b+j)%4];
        }
    }
    const auto saved_weights=weights;
    Resources leaves,ops;mt::CpuEncoderQ8 owner;
    leaves.ctx=ggml_init({1024*1024,nullptr,true});ops.ctx=ggml_init({4*1024*1024,nullptr,true});
    require(leaves.ctx && ops.ctx,"contexts");
    auto* w=ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_Q8_0,k,n);
    auto* x=ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_F32,k,m);
    leaves.buffer=ggml_backend_alloc_ctx_tensors(leaves.ctx,backend);require(leaves.buffer,"external operands");
    auto* eager=ggml_mul_mat(ops.ctx,w,x);
    auto* custom=owner.mul_mat(ops.ctx,w,x);
    require(custom && custom->op==GGML_OP_CUSTOM && custom->src[0]==w && custom->src[1]==x,"actual route/dependencies");
    ggml_set_output(eager);ggml_set_output(custom);
    auto* all=ggml_new_graph(ops.ctx);ggml_build_forward_expand(all,eager);ggml_build_forward_expand(all,custom);
    ops.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ops.alloc && ggml_gallocr_alloc_graph(ops.alloc,all),"allocation");
    ggml_backend_tensor_set(w,weights.data(),0,ggml_nbytes(w));
    size_t differences=0;
    for(int pass=0;pass<2;++pass) {
        if(pass)for(auto& f:input)f*= -.5f;
        ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
        std::vector<float> a(size_t(n)*m,NAN),b(a.size(),NAN),actual_x(input.size());
        ggml_backend_tensor_set(eager,a.data(),0,ggml_nbytes(eager));
        ggml_backend_tensor_set(custom,b.data(),0,ggml_nbytes(custom));
        require(ggml_backend_graph_compute(backend,all)==GGML_STATUS_SUCCESS,"compute");
        require(owner.ok() && owner.nodes()==1 && owner.executions()==unsigned(pass+1)
            && owner.min_workers()==16 && owner.max_workers()==16,"actual workers and callback cleanup");
        ggml_backend_tensor_get(eager,a.data(),0,ggml_nbytes(eager));
        ggml_backend_tensor_get(custom,b.data(),0,ggml_nbytes(custom));
        for(size_t i=0;i<a.size();++i) { require(std::isfinite(a[i]) && std::isfinite(b[i]),"finite output");
            differences+=std::memcmp(&a[i],&b[i],sizeof(float))!=0; }
        ggml_backend_tensor_get(x,actual_x.data(),0,ggml_nbytes(x));
        std::vector<block_q8_0> actual_w(weights.size());ggml_backend_tensor_get(w,actual_w.data(),0,ggml_nbytes(w));
        require(!std::memcmp(input.data(),actual_x.data(),ggml_nbytes(x))
            && !std::memcmp(saved_weights.data(),actual_w.data(),ggml_nbytes(w))
            && !std::memcmp(weights.data(),saved_weights.data(),ggml_nbytes(w)),"immutable operands");
    }
    std::printf("{\"K\":%d,\"N\":%d,\"M\":1500,\"distribution\":\"%s\",\"inputUpdates\":2,\"floatBitDifferences\":%zu,\"nodes\":1,\"executions\":2,\"actualWorkers\":16,\"scratchCleaned\":true,\"operandsUnchanged\":true}\n",k,n,distribution,differences);
    require(!differences,"bit-exact ordinary F32 matmul");
}
}
int main() {
    try {
        env("MTD_DEVICE","cpu");env("MTD_THREADS","16");env("MTD_THREADS_DECODE","8");env("MTD_CPU_OPT","0");
        auto* backend=mt::backend();require(ggml_backend_is_cpu(backend),"CPU backend");fallback(backend);
        if(!mt::CpuEncoderQ8::supported()) { std::puts("Unsupported ISA: fallbacks verified");return 0; }
        for(auto s:std::vector<std::pair<int,int>>{{1024,1024},{1024,4096},{4096,1024}})
            for(const char* d:{"normal","zero","extreme","scales"})trial(backend,s.first,s.second,d);
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"encoder_q8: %s\n",e.what());return 1; }
}
