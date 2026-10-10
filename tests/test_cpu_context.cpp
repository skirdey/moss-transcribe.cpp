// MIT. Actual native single-query matmuls are the oracle, including strided
// cache views, GQA, vector tails and deliberately poisoned future probabilities.
#include "cpu_context.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
void require(bool ok,const char* why) { if (!ok) throw std::runtime_error(why); }
void trial(int threads,int prefix,int queries,int hd,int heads,int kv_heads,bool poison) {
    ggml_backend_cpu_set_n_threads(mt::backend(),threads);
    auto* ctx=ggml_init({2*1024*1024,nullptr,true}); require(ctx,"context");
    const int kv=prefix+queries, capacity=kv+11;
    auto* full=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,capacity,hd,kv_heads,1);
    auto* p=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,kv,queries,heads,1);
    ggml_set_input(full); ggml_set_input(p);
    auto* v=ggml_view_4d(ctx,full,kv,hd,kv_heads,1,full->nb[1],full->nb[2],full->nb[3],0);
    auto* candidate=mt::cpu_causal_context(ctx,v,p); require(candidate,"eligible kernel");
    ggml_set_output(candidate);
    auto* graph=ggml_new_graph(ctx);
    std::vector<ggml_tensor*> reference;
    for (int q=0;q<queries;++q) {
        const int valid=prefix+q+1;
        auto* vr=ggml_view_4d(ctx,full,valid,hd,kv_heads,1,full->nb[1],full->nb[2],full->nb[3],0);
        auto* pr=ggml_view_4d(ctx,p,valid,1,heads,1,p->nb[1],p->nb[2],p->nb[3],q*p->nb[1]);
        auto* result=ggml_mul_mat(ctx,vr,pr); ggml_set_output(result);
        ggml_build_forward_expand(graph,result); reference.push_back(result);
    }
    ggml_build_forward_expand(graph,candidate);
    std::mt19937 random(20261010);std::uniform_real_distribution<float> value(-.2f,.2f),probability(0,.25f);
    std::vector<float> vv(ggml_nelements(full)),pp(ggml_nelements(p));
    for (auto& f:vv) f=value(random);
    // Poison inactive capacity; active future V slots belong to later queries.
    for (int h=0;h<kv_heads;++h) for (int f=0;f<hd;++f)
        for (int k=kv;k<capacity;++k) vv[(h*hd+f)*capacity+k]=std::numeric_limits<float>::quiet_NaN();
    for (int h=0;h<heads;++h) for (int q=0;q<queries;++q) for (int k=0;k<kv;++k)
        pp[(h*queries+q)*kv+k]=k<prefix+q+1 ? probability(random) :
            poison ? std::numeric_limits<float>::quiet_NaN() : 0.0f;
    const auto old_v=vv,old_p=pp;
    const auto counts=mt::cpu_causal_context_counts();
    require(mt::compute_graph_with_inputs(graph,[&]() {
        ggml_backend_tensor_set(full,vv.data(),0,vv.size()*sizeof(float));
        ggml_backend_tensor_set(p,pp.data(),0,pp.size()*sizeof(float));
    },true),"native graph compute");
    const auto after=mt::cpu_causal_context_counts();
    require(after.executed-counts.executed==1,"custom kernel actually executed");
    std::vector<float> result(ggml_nelements(candidate)),one(heads*hd);
    ggml_backend_tensor_get(candidate,result.data(),0,result.size()*sizeof(float));
    size_t bits=0;
    for (int q=0;q<queries;++q) {
        ggml_backend_tensor_get(reference[q],one.data(),0,one.size()*sizeof(float));
        for (int h=0;h<heads;++h) for (int f=0;f<hd;++f) {
            const auto& a=one[h*hd+f];const auto& b=result[(h*queries+q)*hd+f];
            bits+=std::memcmp(&a,&b,sizeof(float))!=0;
            require(std::isfinite(a)&&std::isfinite(b),"finite valid context");
        }
    }
    ggml_backend_tensor_get(full,vv.data(),0,vv.size()*sizeof(float));
    ggml_backend_tensor_get(p,pp.data(),0,pp.size()*sizeof(float));
    require(!std::memcmp(old_v.data(),vv.data(),vv.size()*sizeof(float)) &&
            !std::memcmp(old_p.data(),pp.data(),pp.size()*sizeof(float)),"all operands including poison unchanged");
    require(bits==0,"native single-query raw bits");
    ggml_tensor bad_v=*v,bad_p=*p;
    int rejected=0;
    auto reject=[&]() {require(!mt::cpu_causal_context(ctx,&bad_v,&bad_p),"unsupported layout falls back");++rejected;bad_v=*v;bad_p=*p;};
    bad_p.ne[1]=1;reject();bad_v.ne[0]=bad_p.ne[0]=queries;reject();bad_p.ne[3]=2;reject();
    bad_p.ne[2]=kv_heads+1;reject();bad_v.type=GGML_TYPE_F16;reject();
    bad_v.nb[0]=2*sizeof(float);reject();bad_v.nb[1]=sizeof(float);reject();
    bad_p.nb[1]+=sizeof(float);reject();
    bad_v.ne[0]=bad_p.ne[0]=int64_t(std::numeric_limits<int>::max())+1;reject();
    bad_v.nb[2]=std::numeric_limits<size_t>::max();reject();
    std::printf("{\"record\":\"causalContextControl\",\"threads\":%d,\"prefix\":%d,\"queries\":%d,"
        "\"headDim\":%d,\"queryHeads\":%d,\"kvHeads\":%d,\"futureProbabilitiesPoisoned\":%s,"
        "\"nativeMatmulBitDifferences\":%zu,\"unsupportedLayoutsRejected\":%d,\"operandsUnchanged\":true}\n",
        threads,prefix,queries,hd,heads,kv_heads,poison?"true":"false",bits,rejected);
    ggml_free(ctx);
}
}
int main() {
    try {
        for (const auto& item : {std::pair<const char*,const char*>{"MTD_DEVICE","cpu"},{"MTD_THREADS","16"}}) {
#ifdef _WIN32
            require(_putenv_s(item.first,item.second)==0,"CPU test environment");
#else
            require(setenv(item.first,item.second,1)==0,"CPU test environment");
#endif
        }
        require(std::string(mt::backend_name())=="CPU","CPU test backend");
        int count=0;
        for (int threads:{1,16}) for (int prefix:{1,31,32,33,63,64,65,127,128,129,511,512,513,1023,1024,1025})
            for (int queries:{2,4,8}) for (int shape:{0,1}) for (bool poison:{false,true}) {
                trial(threads,prefix,queries,shape?128:3,shape?16:4,shape?8:2,poison);++count;
            }
        std::printf("{\"record\":\"causalContextControlSummary\",\"cases\":%d,\"passed\":true,\"timingPerformed\":false}\n",count);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"causal context: %s\n",e.what());return 1; }
}
