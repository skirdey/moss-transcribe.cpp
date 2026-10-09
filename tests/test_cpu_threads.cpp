// Verify the *actual ggml worker count*, nested restoration and startup bounds.
#include "backend.hpp"
#include "ggml-cpu.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

static void env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}
static void observe(ggml_tensor* out, const ggml_tensor* in, int ith, int nth, void* data) {
    static_cast<std::atomic<int>*>(data)->store(nth);
    if (ith == 0) std::memcpy(out->data, in->data, ggml_nbytes(out));
}
static bool workers(int expected) {
    std::atomic<int> count{0};
    auto ctx=ggml_init({2*1024*1024,nullptr,true});
    auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,4);ggml_set_input(x);
    auto y=ggml_map_custom1(ctx,x,observe,GGML_N_TASKS_MAX,&count);ggml_set_output(y);
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    const float data[]={1,2,3,4};
    bool ok=mt::compute_graph_with_inputs(graph,[&]{ggml_backend_tensor_set(x,data,0,sizeof(data));});
    float output[4];ggml_backend_tensor_get(y,output,0,sizeof(output));
    ok &= count.load()==expected && mt::cpu_thread_count()==expected && !std::memcmp(output,data,sizeof(data));
    ggml_free(ctx);return ok;
}
int main() {
    env("MTD_DEVICE","cpu");env("MTD_THREADS","16");env("MTD_CPU_OPT","1");env("MTD_PROFILE","0");
    env("MTD_THREADS_DECODE","8");env("MTD_THREADS_LOGITS","4");
    env("MTD_THREADS_WHISPER","500");env("MTD_THREADS_ADAPTOR","0");env("MTD_THREADS_PREFILL","invalid");
    if (!ggml_backend_is_cpu(mt::backend())) return 77;
    bool ok=workers(16);
    {
        mt::CpuThreadScope decode(mt::CpuThreadPhase::Decode);ok &= workers(8);
        { mt::CpuThreadScope logits(mt::CpuThreadPhase::Logits);ok &= workers(4); }
        ok &= workers(8);
    }
    ok &= workers(16);
    try { mt::CpuThreadScope decode(mt::CpuThreadPhase::Decode);throw std::runtime_error("test"); }
    catch(const std::runtime_error&) { ok &= workers(16); }
    for (auto phase : {mt::CpuThreadPhase::Whisper,mt::CpuThreadPhase::Adaptor,mt::CpuThreadPhase::Prefill}) {
        mt::CpuThreadScope scope(phase);ok &= workers(16);
    }
    return ok ? 0 : 1;
}
