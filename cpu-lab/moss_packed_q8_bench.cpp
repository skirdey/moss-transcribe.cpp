// Research probe: identical Q8 weight bytes, eager versus packed AMX/VNNI math.
// Float-bit differences are reported and make this strict numerical gate fail.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    auto backend = ggml_backend_cpu_init();
    auto dev = ggml_backend_get_device(backend);
    auto get_extra = reinterpret_cast<ggml_backend_dev_get_extra_bufts_t>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_dev_get_extra_bufts"));
    ggml_backend_buffer_type_t packed_type = nullptr;
    if (get_extra) for (auto* p = get_extra(dev); p && *p; ++p)
        if (std::strcmp(ggml_backend_buft_name(*p), "AMX") == 0) packed_type = *p;
    if (!packed_type) { std::fprintf(stderr,"No AMX packed buffer\n"); return 77; }
    bool exact = true;
    for (int threads : {1,16}) for (int k : {1024,3072}) for (int m : {1,16,375}) {
        const int n = 1024;
        ggml_backend_cpu_set_n_threads(backend, threads);
        ggml_init_params params{4*1024*1024,nullptr,true};
        auto wc = ggml_init(params), pc = ggml_init(params), gc = ggml_init(params);
        auto w = ggml_new_tensor_2d(wc, GGML_TYPE_Q8_0,k,n);
        auto p = ggml_new_tensor_2d(pc, GGML_TYPE_Q8_0,k,n);
        auto wb = ggml_backend_alloc_ctx_tensors_from_buft(wc,ggml_backend_get_default_buffer_type(backend));
        auto pb = ggml_backend_alloc_ctx_tensors_from_buft(pc,packed_type);
        auto x = ggml_new_tensor_2d(gc,GGML_TYPE_F32,k,m); ggml_set_input(x);
        auto y = ggml_mul_mat(gc,w,x), z = ggml_mul_mat(gc,p,x);
        ggml_mul_mat_set_prec(y,GGML_PREC_F32); ggml_mul_mat_set_prec(z,GGML_PREC_F32);
        ggml_set_output(y); ggml_set_output(z);
        auto both = ggml_new_graph(gc); ggml_build_forward_expand(both,y); ggml_build_forward_expand(both,z);
        auto alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!wb || !pb || !ggml_gallocr_alloc_graph(alloc,both)) return 2;
        std::mt19937 random(410); std::normal_distribution<float> normal(0,1);
        std::vector<float> weights(k*n), input(k*m);
        for (auto& value:weights) value=normal(random);
        for (auto& value:input) value=normal(random);
        std::vector<unsigned char> quant(ggml_nbytes(w));
        ggml_quantize_chunk(GGML_TYPE_Q8_0,weights.data(),quant.data(),0,n,k,nullptr);
        ggml_backend_tensor_set(w,quant.data(),0,quant.size());
        ggml_backend_tensor_set(p,quant.data(),0,quant.size());
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        if (ggml_backend_graph_compute(backend,both)!=GGML_STATUS_SUCCESS) return 2;
        std::vector<float> a(n*m),b(n*m);
        ggml_backend_tensor_get(y,a.data(),0,a.size()*sizeof(float));
        ggml_backend_tensor_get(z,b.data(),0,b.size()*sizeof(float));
        size_t different=0; float max_abs=0; double sq=0;
        for (size_t i=0;i<a.size();++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return 2;
            different+=std::memcmp(&a[i],&b[i],4)!=0;
            max_abs=std::max(max_abs,std::fabs(a[i]-b[i])); sq+=double(a[i]-b[i])*(a[i]-b[i]);
        }
        exact &= different==0;
        auto rg=ggml_new_graph(gc), pg=ggml_new_graph(gc);
        ggml_build_forward_expand(rg,y); ggml_build_forward_expand(pg,z);
        std::vector<double> rt,pt;
        for (int round=0;round<6;++round) for (int step=0;step<2;++step) {
            bool pack=(round+step)%2;
            auto start=std::chrono::steady_clock::now();
            for (int i=0;i<3;++i) if(ggml_backend_graph_compute(backend,pack?pg:rg)!=GGML_STATUS_SUCCESS) return 2;
            double us=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()*1e6/3;
            if(round) (pack?pt:rt).push_back(us);
        }
        std::sort(rt.begin(),rt.end());std::sort(pt.begin(),pt.end());
        std::printf("{\"threads\":%d,\"K\":%d,\"N\":%d,\"M\":%d,\"identicalInputWeightBytes\":true,\"floatBitDifferences\":%zu,\"maxAbsError\":%.9g,\"rmsError\":%.9g,\"eagerUs\":%.3f,\"packedUs\":%.3f,\"speedup\":%.4f}\n",threads,k,n,m,different,max_abs,std::sqrt(sq/a.size()),rt[2],pt[2],rt[2]/pt[2]);
        ggml_gallocr_free(alloc);ggml_backend_buffer_free(wb);ggml_backend_buffer_free(pb);
        ggml_free(wc);ggml_free(pc);ggml_free(gc);
    }
    ggml_backend_free(backend);
    return exact?0:1;
}
