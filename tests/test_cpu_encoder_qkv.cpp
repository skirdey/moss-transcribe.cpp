// MIT. Three distinct projections, protected inputs, full bits and failure paths.
#include "cpu_encoder_q8.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
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
void require(bool b,const char* m) { if(!b)throw std::runtime_error(m); }
void env(const char* k,const char* v) {
#ifdef _WIN32
    _putenv_s(k,v);
#else
    setenv(k,v,1);
#endif
}
struct Resources {
    ggml_context* ctx=nullptr;ggml_backend_buffer_t buffer=nullptr;ggml_gallocr_t alloc=nullptr;
    ~Resources() { if(alloc)ggml_gallocr_free(alloc);if(buffer)ggml_backend_buffer_free(buffer);if(ctx)ggml_free(ctx); }
};
void fallbacks() {
    Resources r;r.ctx=ggml_init({2*1024*1024,nullptr,true});require(r.ctx,"fallback context");
    mt::CpuEncoderQ8 owner;
    auto* x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,1024,1500);
    auto* w=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,1024,1024);
    ggml_tensor* weights[]={w,w,w};
    require(!owner.qkv(nullptr,x,weights,3) && !owner.qkv(r.ctx,nullptr,weights,3)
        && !owner.qkv(r.ctx,x,nullptr,3) && !owner.qkv(r.ctx,x,weights,2)
        && !owner.qkv(r.ctx,x,weights,4),"null/count fallbacks");
    auto* short_x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,1024,64);
    auto* q8_x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,1024,1500);
    auto* large_x=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,1056,1500);
    auto* strided=ggml_view_2d(r.ctx,large_x,1024,1500,1056*sizeof(float),0);
    require(!owner.qkv(r.ctx,short_x,weights,3) && !owner.qkv(r.ctx,q8_x,weights,3)
        && !owner.qkv(r.ctx,strided,weights,3),"input shape/type/layout fallbacks");
    auto* float_w=ggml_new_tensor_2d(r.ctx,GGML_TYPE_F32,1024,1024);
    auto* narrow_w=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,1024,128);
    auto* large_w=ggml_new_tensor_2d(r.ctx,GGML_TYPE_Q8_0,1056,1024);
    auto* strided_w=ggml_view_2d(r.ctx,large_w,1024,1024,large_w->nb[1],0);
    for(auto* bad:{static_cast<ggml_tensor*>(nullptr),float_w,narrow_w,strided_w}) {
        weights[1]=bad;require(!owner.qkv(r.ctx,x,weights,3),"each weight validated");
    }
    weights[1]=w;
    { mt::CpuThreadScope fewer(mt::CpuThreadPhase::Decode);
        require(mt::cpu_thread_count()==8 && !owner.qkv(r.ctx,x,weights,3),"8-worker fallback"); }
    require(mt::cpu_thread_count()==16,"thread restoration");
    if(!mt::CpuEncoderQ8::supported())require(!owner.qkv(r.ctx,x,weights,3),"unsupported ISA fallback");
    require(owner.nodes()==0 && owner.consumers()==0 && owner.executions()==0 && owner.ok(),"no fallback contexts");
}

void trial(ggml_backend_t backend,const char* distribution,bool benchmark) {
    constexpr int k=1024,n=1024,m=1500,blocks=k/32;
    std::mt19937 rng(20261010);std::normal_distribution<float> normal(0,1);
    std::vector<float> input(size_t(k)*m),row(k);
    std::array<std::vector<block_q8_0>,3> weights;
    const auto quant=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
    for(auto& f:input)f=normal(rng);
    const float edges[]={-1000.f,1000.f,-0.f,1e-12f,-1e-12f,.5f,-.5f};
    if(std::strcmp(distribution,"normal"))for(size_t i=0;i<input.size();++i)
        input[i]=!std::strcmp(distribution,"zero") ? 0.f : edges[i%7];
    for(int group=0;group<3;++group) {
        weights[group].resize(size_t(blocks)*n);
        for(int r=0;r<n;++r) { for(auto& f:row)f=normal(rng);quant(row.data(),weights[group].data()+size_t(r)*blocks,k); }
        if(!std::strcmp(distribution,"extreme") || !std::strcmp(distribution,"scales")) {
            const int8_t codes[]={-128,-127,0,127};
            const std::array<ggml_half,8> scales={0,0x8000,1,0x8001,0x0400,0x8400,0x7bff,0xfbff};
            for(size_t b=0;b<weights[group].size();++b) {
                weights[group][b].d=!std::strcmp(distribution,"scales") ? scales[(b+group)%8] : ggml_fp32_to_fp16(1.f/(1+(b+group)%9));
                for(int j=0;j<32;++j)weights[group][b].qs[j]=codes[(b+j+group)%4];
            }
        }
    }
    const auto saved_weights=weights;
    Resources leaves,ops;mt::CpuEncoderQ8 single,grouped;
    leaves.ctx=ggml_init({1024*1024,nullptr,true});ops.ctx=ggml_init({8*1024*1024,nullptr,true});
    require(leaves.ctx && ops.ctx,"contexts");
    auto* x=ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_F32,k,m);
    ggml_tensor* w[3];for(int g=0;g<3;++g)w[g]=ggml_new_tensor_2d(leaves.ctx,GGML_TYPE_Q8_0,k,n);
    leaves.buffer=ggml_backend_alloc_ctx_tensors(leaves.ctx,backend);require(leaves.buffer,"external operands");
    auto* fused=grouped.qkv(ops.ctx,x,w,3);
    require(fused && fused->op==GGML_OP_CUSTOM && fused->ne[0]==n && fused->ne[1]==m
        && fused->ne[2]==3 && fused->ne[3]==1 && fused->src[0]==w[0] && fused->src[1]==x
        && fused->src[2]==w[1] && fused->src[3]==w[2],"group shape/dependencies");
    ggml_set_output(fused);
    ggml_tensor *eager[3],*old[3],*views[3];
    auto* all=ggml_new_graph(ops.ctx);
    std::array<ggml_cgraph*,3> graphs={ggml_new_graph(ops.ctx),ggml_new_graph(ops.ctx),ggml_new_graph(ops.ctx)};
    for(int g=0;g<3;++g) {
        eager[g]=ggml_mul_mat(ops.ctx,w[g],x);old[g]=single.mul_mat(ops.ctx,w[g],x);
        views[g]=ggml_view_2d(ops.ctx,fused,n,m,fused->nb[1],g*fused->nb[2]);
        require(old[g] && ggml_is_contiguous(views[g]) && views[g]->view_offs==g*size_t(n)*m*sizeof(float),"contiguous distinct slices");
        ggml_set_output(eager[g]);ggml_set_output(old[g]);ggml_set_output(views[g]);
        ggml_build_forward_expand(all,eager[g]);ggml_build_forward_expand(all,old[g]);ggml_build_forward_expand(all,views[g]);
        ggml_build_forward_expand(graphs[0],eager[g]);ggml_build_forward_expand(graphs[1],old[g]);ggml_build_forward_expand(graphs[2],views[g]);
    }
    ops.alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(ops.alloc && ggml_gallocr_alloc_graph(ops.alloc,all),"graph allocation");
    for(int g=0;g<3;++g)ggml_backend_tensor_set(w[g],weights[g].data(),0,ggml_nbytes(w[g]));
    auto check_operands=[&]() {
        std::vector<float> actual_x(input.size());ggml_backend_tensor_get(x,actual_x.data(),0,ggml_nbytes(x));
        require(!std::memcmp(input.data(),actual_x.data(),ggml_nbytes(x)),"input immutable");
        for(int g=0;g<3;++g) {
            std::vector<block_q8_0> actual(weights[g].size());ggml_backend_tensor_get(w[g],actual.data(),0,ggml_nbytes(w[g]));
            require(!std::memcmp(saved_weights[g].data(),actual.data(),ggml_nbytes(w[g]))
                && !std::memcmp(saved_weights[g].data(),weights[g].data(),ggml_nbytes(w[g])),"weights immutable");
        }
    };
    size_t old_differences=0,fused_differences=0;
    auto check_outputs=[&]() {
        std::vector<float> a(size_t(n)*m),b(a.size()),c(a.size());
        for(int g=0;g<3;++g) {
            ggml_backend_tensor_get(eager[g],a.data(),0,ggml_nbytes(eager[g]));
            ggml_backend_tensor_get(old[g],b.data(),0,ggml_nbytes(old[g]));
            ggml_backend_tensor_get(views[g],c.data(),0,ggml_nbytes(views[g]));
            for(size_t i=0;i<a.size();++i) {
                require(std::isfinite(a[i]) && std::isfinite(b[i]) && std::isfinite(c[i]),"finite output");
                old_differences+=std::memcmp(&a[i],&b[i],sizeof(float))!=0;
                fused_differences+=std::memcmp(&a[i],&c[i],sizeof(float))!=0;
            }
        }
        require(!old_differences && !fused_differences,"full raw float parity");check_operands();
    };
    std::vector<float> poison(size_t(n)*m,NAN);
    for(int pass=0;pass<2;++pass) {
        if(pass)for(auto& f:input)f*= -.5f;
        ggml_backend_tensor_set(x,input.data(),0,ggml_nbytes(x));
        for(int g=0;g<3;++g)for(auto* y:{eager[g],old[g],views[g]})ggml_backend_tensor_set(y,poison.data(),0,ggml_nbytes(y));
        require(ggml_backend_graph_compute(backend,all)==GGML_STATUS_SUCCESS,"compute");
        require(grouped.ok() && single.ok() && grouped.nodes()==1 && grouped.consumers()==3
            && grouped.executions()==unsigned(pass+1) && grouped.consumer_executions()==unsigned(3*(pass+1))
            && grouped.qkv_nodes()==1 && grouped.qkv_executions()==unsigned(pass+1)
            && single.nodes()==3 && single.consumers()==3 && single.executions()==unsigned(3*(pass+1))
            && single.qkv_nodes()==0 && grouped.min_workers()==16 && grouped.max_workers()==16,"callbacks/workers/cleanup");
        check_outputs();
    }
    std::printf("{\"K\":1024,\"N\":1024,\"M\":1500,\"distribution\":\"%s\",\"projections\":3,\"inputUpdates\":2,\"floatBitDifferences\":%zu,\"singleFloatBitDifferences\":%zu,\"nodes\":1,\"executions\":2,\"consumers\":3,\"consumerExecutions\":6,\"actualWorkers\":16,\"scratchCleaned\":true,\"operandsUnchanged\":true}\n",distribution,fused_differences,old_differences);
    if(benchmark) {
        std::array<std::vector<double>,3> samples;
        for(int round=0;round<7;++round) {
            std::array<int,3> order={round%3,(round+1)%3,(round+2)%3};
            if(round%2)std::reverse(order.begin(),order.end());
            for(int v:order) {
                const auto begin=std::chrono::steady_clock::now();
                require(ggml_backend_graph_compute(backend,graphs[v])==GGML_STATUS_SUCCESS,"timed graph");
                const double us=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count();
                if(round)samples[v].push_back(us);
            }
        }
        check_outputs();require(grouped.ok() && single.ok(),"timed cleanup");
        const char* names[]={"ordinaryF32","singleVnni","sharedQkvVnni"};
        for(int v=0;v<3;++v) {
            auto sorted=samples[v];std::sort(sorted.begin(),sorted.end());
            std::printf("{\"variant\":\"%s\",\"medianMicroseconds\":%.6f,\"samplesMicroseconds\":[",names[v],(sorted[2]+sorted[3])*.5);
            for(size_t i=0;i<samples[v].size();++i)std::printf("%s%.6f",i?",":"",samples[v][i]);
            std::puts("],\"quantizationPackingAllocationCleanupIncluded\":true,\"fullInputLatency\":false,\"operandsUnchanged\":true}");
        }
    }
    // The factory sees the supported cached setting, but GGML actually dispatches
    // eight workers. All three slices must be poisoned and owned scratch freed.
    ggml_backend_cpu_set_n_threads(backend,8);
    std::vector<float> zeros(size_t(n)*m*3,0.f),failed(zeros.size());
    ggml_backend_tensor_set(fused,zeros.data(),0,ggml_nbytes(fused));
    require(ggml_backend_graph_compute(backend,graphs[2])==GGML_STATUS_SUCCESS,"failure callback dispatch");
    ggml_backend_cpu_set_n_threads(backend,16);
    ggml_backend_tensor_get(fused,failed.data(),0,ggml_nbytes(fused));
    require(!grouped.ok() && grouped.min_workers()==8 && grouped.max_workers()==8
        && std::all_of(failed.begin(),failed.end(),[](float f){return std::isnan(f);}),"all failure slices poisoned");
    check_operands();
    std::puts("{\"failurePathActualWorkers\":8,\"failureOutputElements\":4608000,\"allFailureSlicesNaN\":true,\"operandsUnchanged\":true}");
    require(ggml_backend_graph_compute(backend,graphs[2])==GGML_STATUS_SUCCESS && grouped.ok(),"context recovers after worker failure");
    check_outputs();
    std::puts("{\"recoveryActualWorkers\":16,\"recoveredFloatBitDifferences\":0,\"scratchCleaned\":true,\"operandsUnchanged\":true}");
}
}
int main(int argc,char** argv) {
    try {
        const bool benchmark=argc==2 && !std::strcmp(argv[1],"--benchmark");
        require(argc==1 || benchmark,"usage: test_cpu_encoder_qkv [--benchmark]");
        env("MTD_DEVICE","cpu");env("MTD_THREADS","16");env("MTD_THREADS_DECODE","8");env("MTD_CPU_OPT","0");
        auto* backend=mt::backend();require(ggml_backend_is_cpu(backend),"CPU backend");fallbacks();
        if(!mt::CpuEncoderQ8::supported()){std::puts("Unsupported ISA: fallbacks verified");return 0;}
        if(benchmark)trial(backend,"normal",true);
        else for(const char* d:{"normal","zero","extreme","scales"})trial(backend,d,false);
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"encoder_qkv: %s\n",e.what());return 1;}
}
