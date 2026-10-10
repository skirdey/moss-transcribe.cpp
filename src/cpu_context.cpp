// MIT. Preserve the pinned single-query F32 dot at each logical KV length.
#include "cpu_context.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#include "ggml-cpu/vec.h"
#include <algorithm>
#include <atomic>
#include <limits>

namespace mt {
namespace {
std::atomic<uint64_t> built{0}, executed{0};
void causal_context(ggml_tensor* dst, int ith, int nth, void*) {
    const auto* v = dst->src[0]; const auto* p = dst->src[1];
    const int64_t hd = dst->ne[0], queries = dst->ne[1];
    const int64_t past = p->ne[0]-queries, groups = p->ne[2]/v->ne[2];
    const int64_t elements = ggml_nelements(dst), per = elements/nth, tail = elements%nth;
    const int64_t begin = per*ith+std::min<int64_t>(ith,tail);
    const int64_t end = begin+per+(ith<tail);
    auto* out = static_cast<float*>(dst->data);
    for (int64_t i = begin; i < end; ++i) {
        const int64_t feature = i%hd, query = (i/hd)%queries, head = i/(hd*queries);
        const auto* vp = reinterpret_cast<const float*>(static_cast<const char*>(v->data)+
            feature*v->nb[1]+(head/groups)*v->nb[2]);
        const auto* pp = reinterpret_cast<const float*>(static_cast<const char*>(p->data)+
            query*p->nb[1]+head*p->nb[2]);
        // The common batch length can change GCC's FMA/non-FMA tail path even
        // with zero future probabilities. Exactly match the serial query length.
        ggml_vec_dot_f32(int(past+query+1),out+i,0,vp,0,pp,0,1);
    }
    if (ith==0) executed.fetch_add(1,std::memory_order_relaxed);
}
}
ggml_tensor* cpu_causal_context(ggml_context* ctx, ggml_tensor* v, ggml_tensor* p) {
    const auto imax = std::numeric_limits<int>::max();
    if (!ctx || !v || !p || v->type!=GGML_TYPE_F32 || p->type!=GGML_TYPE_F32 ||
        v->ne[3]!=1 || p->ne[3]!=1 || v->ne[0]!=p->ne[0] || p->ne[1]<2 ||
        p->ne[0]<=p->ne[1] || p->ne[0]>imax || v->ne[1]<=0 || v->ne[1]>imax ||
        v->ne[2]<=0 || v->ne[2]>imax || p->ne[2]<=0 || p->ne[2]>imax || p->ne[2]%v->ne[2] ||
        v->nb[0]!=sizeof(float) ||
        v->nb[1]<size_t(v->ne[0])*sizeof(float) ||
        size_t(v->ne[1])>std::numeric_limits<size_t>::max()/v->nb[1] ||
        v->nb[2]<size_t(v->ne[1])*v->nb[1] ||
        v->nb[2]>std::numeric_limits<size_t>::max()/size_t(v->ne[2]) ||
        p->ne[1]>std::numeric_limits<int64_t>::max()/v->ne[1] ||
        p->ne[2]>std::numeric_limits<int64_t>::max()/(v->ne[1]*p->ne[1]) ||
        uint64_t(v->ne[1]*p->ne[1]*p->ne[2])>std::numeric_limits<size_t>::max()/sizeof(float) ||
        uint64_t(p->ne[0]*p->ne[1])>std::numeric_limits<size_t>::max()/sizeof(float)/uint64_t(p->ne[2]) ||
        !ggml_is_contiguous(p) ||
        !ggml_backend_is_cpu(backend())) return nullptr;
    ggml_tensor* args[] = {v,p};
    auto* result = ggml_custom_4d(ctx,GGML_TYPE_F32,v->ne[1],p->ne[1],p->ne[2],1,
        args,2,causal_context,GGML_N_TASKS_MAX,nullptr);
    built.fetch_add(1,std::memory_order_relaxed);
    return result;
}
CpuContextCounts cpu_causal_context_counts() {
    return {built.load(std::memory_order_relaxed),executed.load(std::memory_order_relaxed)};
}
}
