// MIT research probe. Sparse AMX columns keep the pinned x86 Q8 dot's eight
// independent four-code groups; original row-major Q8 weight bytes are loaded
// directly. This is an owned-vector probe, not a MOSS/full-input speed claim.
#include "ggml.h"
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

#if defined(__linux__) && defined(__AMX_INT8__) && defined(__AMX_TILE__) \
    && defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512VL__) \
    && defined(__F16C__) && defined(_OPENMP) && !defined(__AVXVNNIINT8__)
#include <immintrin.h>
#include <omp.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {
void require(bool ok,const char* why) {if(!ok)throw std::runtime_error(why);}
struct alignas(64) Panel {int8_t bytes[8][32]{};};
struct alignas(64) Config {
    uint8_t palette=1,start=0,reserved[14]{};
    uint16_t columns[16]{};
    uint8_t rows[16]{};
};
static_assert(sizeof(Config)==64 && sizeof(Panel)==256,"AMX ABI/panel size");

std::vector<Panel> panels(const std::vector<block_q8_0>& x) {
    std::vector<Panel> result(x.size());
    // Output column g receives only activation group g. Other columns are zero,
    // so AMX yields eight separate integer sums rather than one 32-code sum.
    for(size_t b=0;b<x.size();++b)for(int g=0;g<8;++g)
        std::memcpy(result[b].bytes[g]+g*4,x[b].qs+g*4,4);
    return result;
}

void dot16(const block_q8_0* w,int blocks,const block_q8_0* x,
           const Panel* input,float* output,int lanes) {
    Config config;
    config.columns[0]=32;config.rows[0]=lanes;
    config.columns[1]=32;config.rows[1]=8;
    config.columns[2]=32;config.rows[2]=lanes;
    _tile_loadconfig(&config);
    __m512 acc[8];for(auto& a:acc)a=_mm512_setzero_ps();
    const __mmask16 active=static_cast<__mmask16>((1u<<lanes)-1);
    const __m512i offsets=_mm512_setr_epi32(0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120);
    alignas(64) int32_t sums[16][8];
    alignas(64) ggml_half scales[16]{};
    for(int b=0;b<blocks;++b) {
        // Tile A reads original codes after the two-byte scale. Its row stride
        // is the complete original Q8 weight row; no weight pack or copy.
        _tile_loadd(0,w[b].qs,blocks*sizeof(block_q8_0));
        _tile_loadd(1,input[b].bytes,32);
        _tile_zero(2);
        _tile_dpbssd(2,0,1);
        _tile_stored(2,sums,32);
        for(int r=0;r<lanes;++r)scales[r]=w[r*blocks+b].d;
        // This pinned VNNI reference uses PSIGNB, where negating -128 wraps.
        // Direct signed AMX differs only for activation -128 with weight<0.
        // Keep those literal-byte edge cases exact. Normal F32 Q8 conversion
        // does not usually emit -128; include the check/correction in timing.
        uint32_t minus128=0;
        for(int j=0;j<32;++j)if(x[b].qs[j]==-128)minus128|=uint32_t(1)<<j;
        if(minus128)for(int r=0;r<lanes;++r)for(int j=0;j<32;++j)
            if((minus128&(uint32_t(1)<<j)) && w[r*blocks+b].qs[j]<0)
                sums[r][j/4]+=256*int(w[r*blocks+b].qs[j]);
        const __m512 scale=_mm512_mul_ps(
            _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(scales))),
            _mm512_set1_ps(ggml_fp16_to_fp32(x[b].d)));
        for(int g=0;g<8;++g) {
            const auto index=_mm512_add_epi32(offsets,_mm512_set1_epi32(g));
            const auto partial=_mm512_mask_i32gather_epi32(_mm512_setzero_si512(),active,index,sums,4);
            acc[g]=_mm512_fmadd_ps(scale,_mm512_cvtepi32_ps(partial),acc[g]);
        }
    }
    // Same FMA chains and horizontal-add tree as pinned x86/quants.c.
    const auto even=_mm512_add_ps(_mm512_add_ps(acc[4],acc[0]),_mm512_add_ps(acc[6],acc[2]));
    const auto odd=_mm512_add_ps(_mm512_add_ps(acc[5],acc[1]),_mm512_add_ps(acc[7],acc[3]));
    _mm512_mask_storeu_ps(output,active,_mm512_add_ps(even,odd));
    _tile_release();
}

void trial(int threads,int k,int n,const char* distribution,bool timing) {
    const int blocks=k/32,tiles=(n+15)/16;
    const std::array<ggml_half,8> scale_edges={0x0000,0x8000,0x0001,0x8001,0x0400,0x8400,0x7bff,0xfbff};
    const auto* traits=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    std::mt19937 random(20261010);std::normal_distribution<float> normal(0,1);
    std::vector<float> source(k),row(k),reference(n),candidate(n);
    std::vector<block_q8_0> weights(size_t(blocks)*n),x(blocks);
    for(int r=0;r<n;++r) {
        for(auto& f:row)f=normal(random);
        traits->from_float(row.data(),weights.data()+r*blocks,k);
    }
    for(auto& f:source)f=normal(random);
    if(std::strcmp(distribution,"normal")) {
        const int8_t codes[]={-128,-127,0,127};
        const bool zero=!std::strcmp(distribution,"zero");
        for(size_t b=0;b<weights.size();++b) {
            weights[b].d=ggml_fp32_to_fp16(zero ? 0.f : 1.f/(1+b%9));
            if(!std::strcmp(distribution,"scales"))weights[b].d=scale_edges[b%scale_edges.size()];
            for(int j=0;j<32;++j)weights[b].qs[j]=zero ? 0 : codes[(b+j)%4];
        }
    }
    const auto saved_weights=weights;
    auto compute=[&](bool amx) {
        if(amx) {
            // Panel construction/allocation is inside the timed operation.
            // Shared here in the standalone probe; a graph callback's private
            // per-worker conversion/panels need a separate measurement.
            auto expanded=panels(x);
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(int t=0;t<tiles;++t)dot16(weights.data()+t*16*blocks,blocks,
                x.data(),expanded.data(),candidate.data()+t*16,std::min(16,n-t*16));
        } else {
            #pragma omp parallel for num_threads(threads) schedule(static)
            for(int r=0;r<n;++r)traits->vec_dot(k,reference.data()+r,0,
                weights.data()+r*blocks,0,x.data(),0,1);
        }
    };
    size_t differences=0;
    std::vector<float> saved_source;
    std::vector<block_q8_0> saved_x;
    auto parity=[&] {
        compute(false);compute(true);
        require(!std::memcmp(weights.data(),saved_weights.data(),weights.size()*sizeof(block_q8_0)),"weight bytes changed");
        require(!std::memcmp(x.data(),saved_x.data(),x.size()*sizeof(block_q8_0)),"activation bytes changed");
        require(source==saved_source,"source input changed");
        for(int r=0;r<n;++r) {
            require(std::isfinite(reference[r]) && std::isfinite(candidate[r]),"finite output");
            differences+=std::memcmp(&reference[r],&candidate[r],sizeof(float))!=0;
        }
    };
    for(int pass=0;pass<2;++pass) {
        if(pass)for(auto& f:source)f*=-.5f;
        traits->from_float(source.data(),x.data(),k);
        if(std::strcmp(distribution,"normal")) {
            const int8_t codes[]={-128,-127,0,127};
            const bool zero=!std::strcmp(distribution,"zero");
            for(int b=0;b<blocks;++b) {
                x[b].d=ggml_fp32_to_fp16(zero ? 0.f : (pass ? 1.f : -1.f)/(1+b%5));
                if(!std::strcmp(distribution,"scales"))x[b].d=scale_edges[(b*3+pass)%scale_edges.size()];
                for(int j=0;j<32;++j)x[b].qs[j]=zero ? 0 : codes[(b*3+j+pass)%4];
            }
        }
        saved_source=source;saved_x=x;parity();
    }
    std::array<double,2> us{};
    if(timing && !std::strcmp(distribution,"normal")) {
        std::array<std::vector<double>,2> samples;
        const int repeats=k>=1024 ? 5 : 20;
        for(int round=0;round<6;++round)for(int step=0;step<2;++step) {
            const int v=(round+step)%2;const auto start=std::chrono::steady_clock::now();
            for(int repeat=0;repeat<repeats;++repeat)compute(v);
            const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()*1e6/repeats;
            if(round)samples[v].push_back(elapsed);
        }
        for(int v=0;v<2;++v){std::sort(samples[v].begin(),samples[v].end());us[v]=samples[v][2];}
        parity();
    }
    std::printf("{\"threads\":%d,\"K\":%d,\"N\":%d,\"M\":1,\"distribution\":\"%s\",\"inputUpdates\":2,\"floatBitDifferences\":%zu,\"ordinaryWeightBytes\":%zu,\"packedWeightBytes\":0,\"activationPanelBytes\":%zu,\"panelConstructionIncluded\":true,\"inputQuantizationExcluded\":true,\"referenceUs\":%.3f,\"amxUs\":%.3f,\"speedup\":%.4f,\"fullInputLatency\":false}\n",threads,k,n,distribution,differences,weights.size()*sizeof(block_q8_0),size_t(blocks)*sizeof(Panel),us[0],us[1],us[1] ? us[0]/us[1] : 0);
    std::fflush(stdout);require(!differences,"pinned Q8 float bits changed");
}
}

int main(int argc,char** argv) {
    try {
        const bool timing=argc==2 && !std::strcmp(argv[1],"--benchmark");
        require(argc==1 || timing,"usage: moss_amx_q8_lanes [--benchmark]");
        ggml_cpu_init();
        const auto* t=ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
        if(!ggml_cpu_has_avx512_vnni() || !__builtin_cpu_supports("amx-int8")
            || t->nrows!=1 || t->vec_dot_type!=GGML_TYPE_Q8_0)return 77;
        // Linux grants this permission per process; no privileged kernel change.
        constexpr int get_support=0x1021,request_permission=0x1023,tile_data=18;
        uint64_t features=0;
        if(syscall(SYS_arch_prctl,get_support,&features) || !(features&(uint64_t(1)<<tile_data))
            || syscall(SYS_arch_prctl,request_permission,tile_data))return 77;
        for(int threads:timing ? std::vector<int>{1,8,16} : std::vector<int>{1,16})
            for(int k:{32,96,1024,3072})for(int n:{17,1024,3072})
                for(const char* d:{"normal","zero","extreme","scales"})trial(threads,k,n,d,timing);
        return 0;
    } catch(const std::exception& e){std::fprintf(stderr,"moss_amx_q8_lanes: %s\n",e.what());return 1;}
}
#else
int main(){std::fprintf(stderr,"Requires Linux AMX INT8/TILE and pinned AVX512 VNNI/F16C/OpenMP; no portable fallback timing\n");return 77;}
#endif
