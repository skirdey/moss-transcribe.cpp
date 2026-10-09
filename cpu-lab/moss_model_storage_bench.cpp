// Full-checkpoint byte/promotion check; loader timings are diagnostics only.
// No inference, token output or weights are written. Backend initialization is
// shared by both loaders: use the fresh-process full-model gate for latency.
#include "model_loader.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void env(const char* key,const char* value) {
#ifdef _WIN32
    _putenv_s(key,value);
#else
    setenv(key,value,1);
#endif
}
static bool equal(const mt::ModelLoader& a,const mt::ModelLoader& b,size_t& bytes) {
    if(a.tensor_names()!=b.tensor_names())return false;
    bytes=0;
    for(const auto& name:a.tensor_names()) {
        auto x=a.tensor(name);auto y=b.tensor(name);
        if(!x || !y || x->type!=y->type || std::memcmp(x->ne,y->ne,sizeof(x->ne)) ||
            std::memcmp(x->nb,y->nb,sizeof(x->nb)) || std::memcmp(x->data,y->data,ggml_nbytes(x)))return false;
        bytes+=ggml_nbytes(x);
    }
    return true;
}
int main(int argc,char** argv) {
    if(argc!=2) {std::fprintf(stderr,"usage: moss_model_storage_bench model.gguf\n");return 2;}
    env("MTD_DEVICE","cpu");env("MTD_CPU_OPT","0");
    auto start=std::chrono::steady_clock::now();
    mt::ModelLoader copied;if(!copied.load(argv[1]))return 1;
    auto copied_end=std::chrono::steady_clock::now();
    env("MTD_CPU_OPT","4096");
    auto mapped_start=std::chrono::steady_clock::now();
    mt::ModelLoader mapped;if(!mapped.load(argv[1]))return 1;
    auto mapped_end=std::chrono::steady_clock::now();
    if(!mapped.cpu_mapped())return 77;
    size_t raw_bytes=0,promoted_bytes=0;
    const bool raw_equal=equal(copied,mapped,raw_bytes);
    copied.promote_small_f16_to_f32();mapped.promote_small_f16_to_f32();
    const bool promoted_equal=equal(copied,mapped,promoted_bytes);
    std::printf("{\"protocol\":\"Full tensor-byte and promotion check, same process/backend, shared/warm file cache. Loader time excludes lazy page faults until comparison; no inference or full-input latency claim.\",\"tensors\":%zu,\"rawBytesCompared\":%zu,\"postPromotionBytesCompared\":%zu,\"rawByteIdentity\":%s,\"postPromotionByteIdentity\":%s,\"mappedActive\":true,\"copiedLoadSeconds\":%.9f,\"mappedLoadSeconds\":%.9f}\n",
        copied.tensor_names().size(),raw_bytes,promoted_bytes,raw_equal?"true":"false",promoted_equal?"true":"false",
        std::chrono::duration<double>(copied_end-start).count(),std::chrono::duration<double>(mapped_end-mapped_start).count());
    return raw_equal && promoted_equal ? 0 : 1;
}
