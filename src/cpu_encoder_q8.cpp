// MIT. Selective encoder integration of the frozen exact F32 VNNI probe.
#include "cpu_encoder_q8.hpp"
#include "backend.hpp"
#include "cpu_profile.hpp"
#include "ggml-cpu.h"
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <new>

#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
    && defined(__F16C__) && !defined(__AVXVNNIINT8__)
#include <immintrin.h>
#define MT_ENCODER_Q8_NATIVE 1
#else
#define MT_ENCODER_Q8_NATIVE 0
#endif

namespace mt {
#if MT_ENCODER_Q8_NATIVE
namespace {
struct alignas(64) Panel {
    int8_t groups[8][16][4];
    ggml_half scales[16];
};
static_assert(sizeof(Panel) == 576, "Dense four-code activation panel ABI");

void pack_one(Panel& p, const block_q8_0* input, int blocks, int m, int t, int b) {
    // Worker-owned initialization includes unused columns and ABI padding.
    std::memset(&p, 0, sizeof(p));
    for (int c = 0; c < 16 && t * 16 + c < m; ++c) {
        const auto& x = input[size_t(t * 16 + c) * blocks + b];
        p.scales[c] = x.d;
        for (int g = 0; g < 8; ++g) std::memcpy(p.groups[g][c], x.qs + 4 * g, 4);
    }
}

// Two weight rows and sixteen input rows. Each VNNI instruction reduces only four
// codes, so all eight original floating accumulation chains remain separate.
// Outputs have the actual GGML layout y[input_row * N + weight_row].
inline float inline_half(ggml_half h, bool round_down) {
    // The pinned software converter subtracts equal positive values for half
    // +0. Under downward rounding that produces -0; F16C alone produces +0.
    if (h == 0 && round_down) return -0.0f;
    return _cvtsh_ss(h);
}

template<int R, bool InlineHalf = false>
__attribute__((noinline))
void tile(const block_q8_0* weights, const Panel* panels, int blocks,
          int n, int columns, float* output) {
    const bool round_down = InlineHalf && ((_mm_getcsr() & 0x6000u) == 0x2000u);
    __m512 acc[R][8];
    for (auto& row : acc) for (auto& a : row) a = _mm512_setzero_ps();
    const __mmask16 active = static_cast<__mmask16>((1u << columns) - 1);
    const auto offsets = _mm512_mullo_epi32(_mm512_setr_epi32(
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15), _mm512_set1_epi32(n));
    for (int b = 0; b < blocks; ++b) {
        const auto xs = _mm512_cvtph_ps(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(panels[b].scales)));
        __m512 scales[R];
        for (int r = 0; r < R; ++r) {
            const auto h = weights[r * blocks + b].d;
            const float converted = InlineHalf ? inline_half(h, round_down) : ggml_fp16_to_fp32(h);
            scales[r] = _mm512_mul_ps(xs, _mm512_set1_ps(converted));
        }
        #pragma GCC unroll 8
        for (int g = 0; g < 8; ++g) {
            const auto codes = _mm512_load_si512(panels[b].groups[g]);
            #pragma GCC unroll 2
            for (int r = 0; r < R; ++r) {
                int32_t raw;
                std::memcpy(&raw, weights[r * blocks + b].qs + 4 * g, sizeof(raw));
                const auto w = _mm512_set1_epi32(raw);
                // PSIGNB's wrapped negation is performed directly, including
                // literal -128. abs(-128) is unsigned 128 for DPBUSD.
                const auto zero = _mm512_setzero_si512();
                const auto negative = _mm512_cmp_epi8_mask(w, zero, _MM_CMPINT_LT);
                const auto signed_codes = _mm512_mask_sub_epi8(codes, negative, zero, codes);
                const auto sums = _mm512_dpbusd_epi32(zero, _mm512_abs_epi8(w), signed_codes);
                acc[r][g] = _mm512_fmadd_ps(scales[r], _mm512_cvtepi32_ps(sums), acc[r][g]);
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        const auto even = _mm512_add_ps(_mm512_add_ps(acc[r][4], acc[r][0]),
                                      _mm512_add_ps(acc[r][6], acc[r][2]));
        const auto odd = _mm512_add_ps(_mm512_add_ps(acc[r][5], acc[r][1]),
                                     _mm512_add_ps(acc[r][7], acc[r][3]));
        _mm512_mask_i32scatter_ps(output + r, active, offsets,
                                 _mm512_add_ps(even, odd), sizeof(float));
    }
}

struct WorkerBarrier {
    std::atomic<int> arrived{0};
    std::atomic<unsigned> generation{0};
    void wait(int nth) {
        const unsigned observed = generation.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) == nth - 1) {
            arrived.store(0, std::memory_order_relaxed);
            generation.fetch_add(1, std::memory_order_release);
        } else {
            while (generation.load(std::memory_order_acquire) == observed) _mm_pause();
        }
    }
};
}
#endif
struct CpuEncoderQ8::Context {
    unsigned long long calls=0;
    int workers=0, groups=1;
    bool failed=false;
#if MT_ENCODER_Q8_NATIVE
    WorkerBarrier barrier;
    block_q8_0* quant=nullptr;
    Panel* panels=nullptr;
    ~Context() { delete[] quant; delete[] panels; }
    static void compute(ggml_tensor* dst,int ith,int nth,void* userdata) {
        auto& s=*static_cast<Context*>(userdata);
        const auto* w=dst->src[0]; const auto* x=dst->src[1];
        const int k=int(w->ne[0]),n=int(w->ne[1]),m=int(x->ne[1]);
        const int blocks=k/32,weight_tiles=(n+1)/2,input_tiles=(m+15)/16;
        const size_t count=size_t(input_tiles)*blocks;
        if (ith==0) {
            ++s.calls; s.workers=nth;
            s.quant=new (std::nothrow) block_q8_0[size_t(m)*blocks];
            s.panels=new (std::nothrow) Panel[count];
            s.failed=!s.quant || !s.panels || nth!=16;
        }
        s.barrier.wait(nth);
        if (s.failed) {
            // Report an invalid graph result, never return uninitialized data.
            auto* out=static_cast<float*>(dst->data);
            for (size_t i=ith;i<size_t(n)*m*s.groups;i+=nth) out[i]=NAN;
            s.barrier.wait(nth);
            if (ith==0) { delete[] s.quant; delete[] s.panels;
                s.quant=nullptr;s.panels=nullptr; }
            return;
        }
        const auto from_float=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
        const auto* input=static_cast<const float*>(x->data);
        for (int c=ith;c<m;c+=nth) from_float(input+size_t(c)*k,s.quant+size_t(c)*blocks,k);
        s.barrier.wait(nth);
        for (size_t i=ith;i<count;i+=nth)
            pack_one(s.panels[i],s.quant,blocks,m,int(i/blocks),int(i%blocks));
        s.barrier.wait(nth);
        const int total_tiles=weight_tiles*s.groups;
        const int q=total_tiles/nth,remainder=total_tiles%nth;
        const int first=ith*q+std::min(ith,remainder),last=first+q+(ith<remainder);
        for (int mt=0;mt<input_tiles;++mt) for (int flat=first;flat<last;++flat) {
            const int group=flat/weight_tiles,wt=flat%weight_tiles;
            const auto* group_weight=dst->src[group ? group+1 : 0];
            const auto* weight=static_cast<const block_q8_0*>(group_weight->data)+size_t(wt*2)*blocks;
            const auto* panels=s.panels+size_t(mt)*blocks;
            auto* output=static_cast<float*>(dst->data)+size_t(group)*n*m+size_t(mt*16)*n+wt*2;
            const int columns=std::min(16,m-mt*16);
            if (wt*2+1<n) tile<2,true>(weight,panels,blocks,n,columns,output);
            else tile<1,true>(weight,panels,blocks,n,columns,output);
        }
        s.barrier.wait(nth);
        if (ith==0) { delete[] s.quant; delete[] s.panels;
            s.quant=nullptr;s.panels=nullptr; }
        // GGML's node barrier/join also includes worker-zero cleanup.
    }
#endif
};
CpuEncoderQ8::CpuEncoderQ8()=default;
CpuEncoderQ8::~CpuEncoderQ8()=default;
bool CpuEncoderQ8::supported() {
#if MT_ENCODER_Q8_NATIVE
    ggml_cpu_init();
    const auto* t=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    return ggml_cpu_has_avx512_vnni() && ggml_cpu_has_avx512() && ggml_cpu_has_f16c()
        && t->from_float && t->vec_dot && t->vec_dot_type==GGML_TYPE_Q8_0 && t->nrows==1;
#else
    return false;
#endif
}
ggml_tensor* CpuEncoderQ8::qkv(ggml_context* ctx,ggml_tensor* x,ggml_tensor* const* weights,size_t count) {
#if MT_ENCODER_Q8_NATIVE
    if (!ctx || !x || !weights || count!=3 || x->type!=GGML_TYPE_F32
        || !ggml_is_contiguous(x) || x->ne[0]!=1024 || x->ne[1]!=1500
        || x->ne[2]!=1 || x->ne[3]!=1 || !ggml_backend_is_cpu(backend())
        || cpu_thread_count()!=16 || !supported())return nullptr;
    for(size_t i=0;i<count;++i) {
        const auto* w=weights[i];
        if(!w || w->type!=GGML_TYPE_Q8_0 || w->ne[0]!=1024 || w->ne[1]!=1024
            || w->ne[2]!=1 || w->ne[3]!=1 || !ggml_is_contiguous(w))return nullptr;
    }
    auto context=std::make_unique<Context>();context->groups=3;
    auto* data=context.get();contexts_.push_back(std::move(context));
    ggml_tensor* args[]={weights[0],x,weights[1],weights[2]};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,1024,1500,3,1,args,4,
        Context::compute,GGML_N_TASKS_MAX,data);
#else
    (void)ctx;(void)x;(void)weights;(void)count;return nullptr;
#endif
}
ggml_tensor* CpuEncoderQ8::mul_mat(ggml_context* ctx,ggml_tensor* w,ggml_tensor* x) {
#if MT_ENCODER_Q8_NATIVE
    if (!ctx || !w || !x || w->type!=GGML_TYPE_Q8_0 || x->type!=GGML_TYPE_F32
        || w->ne[0]!=x->ne[0] || !ggml_is_contiguous(w) || !ggml_is_contiguous(x)
        || w->ne[2]!=1 || w->ne[3]!=1 || x->ne[2]!=1 || x->ne[3]!=1 || x->ne[1]!=1500
        || !((w->ne[0]==1024 && (w->ne[1]==1024 || w->ne[1]==4096))
             || (w->ne[0]==4096 && w->ne[1]==1024))
        || !ggml_backend_is_cpu(backend()) || cpu_thread_count()!=16 || !supported()) return nullptr;
    auto context=std::make_unique<Context>();
    auto* data=context.get();
    contexts_.push_back(std::move(context));
    ggml_tensor* args[]={w,x};
    return ggml_custom_4d(ctx,GGML_TYPE_F32,w->ne[1],x->ne[1],1,1,args,2,
        Context::compute,GGML_N_TASKS_MAX,data);
#else
    (void)ctx;(void)w;(void)x; return nullptr;
#endif
}
bool CpuEncoderQ8::ok() const {
    for (const auto& s:contexts_) {
        if (s->failed || !s->calls || s->workers!=16) return false;
#if MT_ENCODER_Q8_NATIVE
        if (s->quant || s->panels) return false;
#endif
    }
    return true;
}
unsigned long long CpuEncoderQ8::executions() const {
    unsigned long long count=0;for (const auto& s:contexts_)count+=s->calls;return count;
}
unsigned long long CpuEncoderQ8::consumers() const {
    unsigned long long count=0;for(const auto& s:contexts_)count+=s->groups;return count;
}
unsigned long long CpuEncoderQ8::consumer_executions() const {
    unsigned long long count=0;for(const auto& s:contexts_)count+=s->calls*s->groups;return count;
}
unsigned long long CpuEncoderQ8::qkv_nodes() const {
    unsigned long long count=0;for(const auto& s:contexts_)count+=s->groups==3;return count;
}
unsigned long long CpuEncoderQ8::qkv_executions() const {
    unsigned long long count=0;for(const auto& s:contexts_)if(s->groups==3)count+=s->calls;return count;
}
int CpuEncoderQ8::min_workers() const {
    int result=0;for (const auto& s:contexts_) if (s->workers)
        result=result ? std::min(result,s->workers) : s->workers;return result;
}
int CpuEncoderQ8::max_workers() const {
    int result=0;for (const auto& s:contexts_)result=std::max(result,s->workers);return result;
}
void CpuEncoderQ8::record_profile() const {
    unsigned long long failures=0;
    for (const auto& s:contexts_)failures+=s->failed || !s->calls || s->workers!=16;
    cpu_profile_record_encoder_q8(nodes(),executions(),failures,min_workers(),max_workers(),
        consumers(),consumer_executions(),qkv_nodes(),qkv_executions());
}
}
