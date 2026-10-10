// MIT. One graph operation for independent M=1 projections. Keep the exact
// pinned CPU quantizer/dot routine; no weight pack, allocation or new reductions.
#include "cpu_projection.hpp"
#include "backend.hpp"
#include "cpu_profile.hpp"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <array>
#include <limits>

namespace mt {
namespace {
constexpr int kMaxWidth=4096;
void projections(ggml_tensor* dst,int ith,int nth,void*) {
    const auto* x=dst->src[0];
    int64_t cursor=dst->ne[0]*ith/nth,end=dst->ne[0]*(ith+1)/nth;
    if(cursor==end)return;
    // The public custom-op callback exposes no worker-pool barrier/scratch.
    // Each active worker converts the small activation once into its own stack
    // storage. Include this redundant work in timing; no shared mutable state.
    alignas(64) std::array<block_q8_0,kMaxWidth/32> quant;
    const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    traits->from_float(static_cast<const float*>(x->data),quant.data(),x->ne[0]);
    int group=1;int64_t offset=0;
    while(cursor>=offset+dst->src[group]->ne[1])offset+=dst->src[group++]->ne[1];
    auto* output=static_cast<float*>(dst->data);
    while(cursor<end) {
        const auto* w=dst->src[group];
        const int64_t last=std::min(end,offset+w->ne[1]);
        for(;cursor<last;++cursor)traits->vec_dot(x->ne[0],output+cursor,0,
            static_cast<const char*>(w->data)+(cursor-offset)*w->nb[1],0,
            quant.data(),0,1);
        offset+=w->ne[1];++group;
    }
}
}
ggml_tensor* cpu_decode_q8_projections(ggml_context* ctx,ggml_tensor* x,
    ggml_tensor* const* weights,size_t count) {
    if(!x || !weights || count<2 || count>3 || x->type!=GGML_TYPE_F32
        || x->nb[0]!=sizeof(float) || x->ne[0]<=0 || x->ne[0]>kMaxWidth
        || x->ne[0]%32 || ggml_nrows(x)!=1 || !ggml_backend_is_cpu(backend()))return nullptr;
    const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    if(!traits->from_float || !traits->vec_dot || traits->vec_dot_type!=GGML_TYPE_Q8_0
        || traits->nrows!=1 || ggml_blck_size(GGML_TYPE_Q8_0)!=32)return nullptr;
    int64_t rows=0;
    ggml_tensor* args[4]={x};
    for(size_t i=0;i<count;++i) {
        auto w=weights[i];
        if(!w || w->type!=GGML_TYPE_Q8_0 || w->ne[0]!=x->ne[0]
            || w->ne[1]<=0 || w->ne[2]!=1 || w->ne[3]!=1 || !ggml_is_contiguous(w)
            || w->ne[1]>std::numeric_limits<int64_t>::max()-rows)return nullptr;
        rows+=w->ne[1];args[i+1]=w;
    }
    auto result=ggml_custom_4d(ctx,GGML_TYPE_F32,rows,1,1,1,args,count+1,
        projections,GGML_N_TASKS_MAX,nullptr);
    cpu_profile_record_fused_q8(count);
    return result;
}
}
