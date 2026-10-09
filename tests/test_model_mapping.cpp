// File-backed CPU weights: byte identity, graph use, promotion, inode lifetime,
// bounds rejection and copied fallback for non-natural GGUF alignment.
#include "model_loader.hpp"
#include "backend.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef _WIN32
#include <unistd.h>

namespace {
struct TempFile {
    std::string path;
    TempFile() {
        char name[]="/tmp/moss-model-map-XXXXXX";
        int fd=mkstemp(name);
        if (fd<0) throw std::runtime_error("mkstemp");
        close(fd); path=name;
    }
    ~TempFile() { std::remove(path.c_str()); }
};
void setting(const char* opt) { setenv("MTD_CPU_OPT",opt,1); }
void require(bool value,const char* message) {
    if (!value) throw std::runtime_error(message);
}
void write_model(const std::string& path,uint32_t alignment=32,int padding=0) {
    auto ctx=ggml_init({65536,nullptr,false}); require(ctx,"context");
    auto f=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,32,4);ggml_set_name(f,"f32.weight");
    for (int i=0;i<128;++i) static_cast<float*>(f->data)[i]=std::sin(float(i))*.7f;
    auto q=ggml_new_tensor_2d(ctx,GGML_TYPE_Q8_0,32,3);ggml_set_name(q,"q8.weight");
    std::vector<float> values(96);for(int i=0;i<96;++i)values[i]=std::cos(float(i))*.2f;
    require(ggml_quantize_chunk(GGML_TYPE_Q8_0,values.data(),q->data,0,3,32,nullptr)==ggml_nbytes(q),"quantize");
    auto h=ggml_new_tensor_1d(ctx,GGML_TYPE_F16,7);ggml_set_name(h,"small.norm");
    for(int i=0;i<7;++i)static_cast<ggml_fp16_t*>(h->data)[i]=ggml_fp32_to_fp16(float(i)-2.5f);
    auto gguf=gguf_init_empty();require(gguf,"gguf");
    gguf_set_val_u32(gguf,"general.alignment",alignment);
    gguf_set_val_u32(gguf,"mtd.text.hidden",32);
    gguf_set_val_str(gguf,"test.padding",std::string(padding,'x').c_str());
    if (alignment != 32) {
        // Setting general.alignment changes metadata, not the writer context's
        // cached alignment. Reopen the tensorless header so offsets and payload
        // padding are actually generated using the requested alignment.
        require(gguf_write_to_file(gguf,path.c_str(),true),"write alignment header");
        gguf_free(gguf);
        gguf=gguf_init_from_file(path.c_str(),{true,nullptr});require(gguf,"alignment header");
    }
    gguf_add_tensor(gguf,f);gguf_add_tensor(gguf,q);gguf_add_tensor(gguf,h);
    require(gguf_write_to_file(gguf,path.c_str(),false),"write");
    gguf_free(gguf);ggml_free(ctx);
}
struct Extent { size_t data, end; };
Extent extent(const std::string& path) {
    auto g=gguf_init_from_file(path.c_str(),{true,nullptr}); require(g,"header");
    const auto data=gguf_get_data_offset(g);size_t end=data;
    for(int64_t i=0;i<gguf_get_n_tensors(g);++i)
        end=std::max(end,data+gguf_get_tensor_offset(g,i)+gguf_get_tensor_size(g,i));
    gguf_free(g);return {data,end};
}
void compare(const mt::ModelLoader& a,const mt::ModelLoader& b) {
    require(a.tensor_names()==b.tensor_names(),"names");
    require(a.config().text_hidden==32 && b.config().text_hidden==32,"metadata");
    for(const auto& name:a.tensor_names()) {
        auto x=a.tensor(name);auto y=b.tensor(name);
        require(x && y && x->type==y->type,"type");
        require(!std::memcmp(x->ne,y->ne,sizeof(x->ne)),"shape");
        require(!std::memcmp(x->nb,y->nb,sizeof(x->nb)),"strides");
        require(!std::memcmp(x->data,y->data,ggml_nbytes(x)),"weight bytes");
    }
}
std::vector<float> projection(const mt::ModelLoader& m,const char* name) {
    auto ctx=ggml_init({1024*1024,nullptr,true});require(ctx,"graph context");
    auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,32);ggml_set_input(x);
    auto y=ggml_mul_mat(ctx,m.tensor(name),x);ggml_set_output(y);
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
    float input[32];for(int i=0;i<32;++i)input[i]=float(i)/32-.5f;
    require(mt::compute_graph_with_inputs(graph,[&]{ggml_backend_tensor_set(x,input,0,sizeof(input));}),"compute");
    std::vector<float> out(ggml_nelements(y));ggml_backend_tensor_get(y,out.data(),0,out.size()*sizeof(float));
    ggml_free(ctx);return out;
}
void run() {
    setenv("MTD_DEVICE","cpu",1);setenv("MTD_THREADS","2",1);setenv("OMP_DYNAMIC","FALSE",1);
    TempFile file;write_model(file.path);
    setting("0");mt::ModelLoader copied;require(copied.load(file.path),"copied load");
    require(!copied.cpu_mapped(),"default storage");
    setting("4096");mt::ModelLoader mapped;require(mapped.load(file.path),"mapped load");
    require(mapped.cpu_mapped(),"mapping active");compare(copied,mapped);
    for(auto name:{"f32.weight","q8.weight"}) {
        auto a=projection(copied,name);auto b=projection(mapped,name);
        require(a.size()==b.size() && !std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),"graph bits");
    }
    copied.promote_small_f16_to_f32();mapped.promote_small_f16_to_f32();compare(copied,mapped);
    require(mapped.tensor("small.norm")->type==GGML_TYPE_F32,"promoted type");
    require(!mapped.load(file.path),"double load");
    // The descriptor is already closed. Unlinking the name must not invalidate
    // the borrowed pages or prevent their later owner-controlled unmapping.
    require(std::remove(file.path.c_str())==0,"unlink");compare(copied,mapped);
    auto a=projection(copied,"q8.weight");auto b=projection(mapped,"q8.weight");
    require(!std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),"unlinked graph");

    TempFile truncated;write_model(truncated.path);auto e=extent(truncated.path);
    require(truncate(truncated.path.c_str(),static_cast<off_t>(e.end-1))==0,"truncate");
    mt::ModelLoader bad;require(!bad.load(truncated.path),"truncated rejection");
    require(!bad.cpu_mapped(),"invalid map not attached");
    for(const auto& name:bad.tensor_names())require(!bad.tensor(name)->data,"validate all before attach");

    TempFile unusual;bool selected=false;
    for(int padding=0;padding<16;++padding) {
        write_model(unusual.path,1,padding);
        if(extent(unusual.path).data % alignof(std::max_align_t)) { selected=true;break; }
    }
    require(selected,"misaligned fixture");
    setting("0");mt::ModelLoader normal;require(normal.load(unusual.path),"unaligned copied");
    setting("4096");mt::ModelLoader fallback;require(fallback.load(unusual.path),"unaligned fallback");
    require(!fallback.cpu_mapped(),"fallback storage");compare(normal,fallback);
}
}
int main() {
    try {run();std::puts("mapped CPU weights: bytes, graph, promotion, lifetime and bounds pass");return 0;}
    catch(const std::exception& e) {std::fprintf(stderr,"mapping test: %s\n",e.what());return 1;}
}
#else
int main() { return 77; }
#endif
