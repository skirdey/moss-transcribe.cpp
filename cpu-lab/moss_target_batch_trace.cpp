// MIT. Model-backed first-divergence trace; no timing or production routing.
#include "backend.hpp"
#include "generate.hpp"
#include "qwen3_decoder.hpp"
#include "ggml-impl.h" // Pinned custom-op field layout; exclude padding/pointers from hashing.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
void env(const char* key, const char* value) {
#ifdef _WIN32
    require(_putenv_s(key,value) == 0, "environment update");
#else
    require(setenv(key,value,1) == 0, "environment update");
#endif
}
using Bits = std::vector<uint32_t>;
enum Stage { Norm,Q,K,V,QNorm,KNorm,QRope,KRope,Scores,Probabilities,
             Context,AttentionProjection,Residual,FfnNorm,Gate,Up,Activation,Down,LayerOutput,StageCount };
constexpr const char* names[] = {"attentionNorm","queryProjection","keyProjection","valueProjection",
    "queryNorm","keyNorm","queryRope","keyRope","scores","probabilities","attentionContext",
    "attentionProjection","attentionResidual","ffnNorm","ffnGate","ffnUp","activation","ffnDown","layerOutput"};
struct Delta {
    size_t bits = 0, nonfinite = 0;
    double max_abs = 0;
    void add(const std::vector<float>& a, const std::vector<float>& b) {
        require(a.size() == b.size() && !a.empty(), "float comparison shape");
        for (size_t i = 0; i < a.size(); ++i) {
            bits += std::memcmp(&a[i],&b[i],sizeof(float)) != 0;
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) ++nonfinite;
            else max_abs = std::max(max_abs,std::abs(double(a[i])-double(b[i])));
        }
    }
};
struct Capture {
    uint64_t graph_fingerprint = 0;
    int nodes = 0, tokens = 0, kv = 0, heads = 0;
    std::vector<std::array<std::vector<float>,StageCount>> layers;
    std::vector<uint8_t> graph_metadata;
    std::vector<uint8_t> metadata_without_leaf_names;
    std::vector<std::string> leaf_names;
    std::vector<int> leaf_cache_roles;
    std::vector<ggml_custom2_op_t> callback_identities;
    std::vector<std::array<uint8_t,GGML_MAX_OP_PARAMS>> raw_custom_parameters;
};
size_t differences(const Bits& a, const Bits& b) {
    require(a.size() == b.size(), "cache shape");
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
    return count;
}
std::vector<float> row(const std::vector<float>& x, int i, int width) {
    require(i >= 0 && (size_t)(i+1)*width <= x.size(), "row bounds");
    return {x.begin()+(size_t)i*width,x.begin()+(size_t)(i+1)*width};
}
void identify_graph(ggml_cgraph* graph, Capture* identity,
                    const std::unordered_map<const ggml_tensor*,int>* cache_roles = nullptr) {
    identity->graph_metadata.clear(); identity->callback_identities.clear();
    identity->metadata_without_leaf_names.clear(); identity->leaf_names.clear(); identity->leaf_cache_roles.clear();
    identity->raw_custom_parameters.clear();
    uint64_t hash = 14695981039346656037ull;
    auto bytes = [&](const void* ptr, size_t size) {
        const auto* p = static_cast<const uint8_t*>(ptr);
        for (size_t i = 0; i < size; ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
        identity->graph_metadata.insert(identity->graph_metadata.end(),p,p+size);
        identity->metadata_without_leaf_names.insert(identity->metadata_without_leaf_names.end(),p,p+size);
    };
    auto label_bytes = [&](const void* ptr, size_t size) {
        const auto* p = static_cast<const uint8_t*>(ptr);
        for (size_t i = 0; i < size; ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
        identity->graph_metadata.insert(identity->graph_metadata.end(),p,p+size);
    };
    std::unordered_map<const ggml_tensor*,int> index;
    const int count = ggml_graph_n_nodes(graph);
    for (int i = 0; i < count; ++i) index[ggml_graph_node(graph,i)] = i;
    for (int i = 0; i < count; ++i) {
        const auto* t = ggml_graph_node(graph,i);
        bytes(&t->op,sizeof(t->op)); bytes(&t->type,sizeof(t->type));
        bytes(t->ne,sizeof(t->ne)); bytes(t->nb,sizeof(t->nb));
        if (t->op == GGML_OP_MAP_CUSTOM2) {
            ggml_map_custom2_op_params params;
            std::memcpy(&params,t->op_params,sizeof(params));
            require(params.fun && !params.userdata,"supported null-userdata custom callback");
            identity->callback_identities.push_back(params.fun);
            // The ggml constructor copies a C struct containing padding. Hash
            // only semantic fields, and compare function identities separately.
            bytes(&params.n_tasks,sizeof(params.n_tasks));
            std::array<uint8_t,GGML_MAX_OP_PARAMS> raw{};
            std::memcpy(raw.data(),t->op_params,raw.size());
            identity->raw_custom_parameters.push_back(raw);
        } else {
            require(t->op != GGML_OP_MAP_CUSTOM1 && t->op != GGML_OP_MAP_CUSTOM3 &&
                    t->op != GGML_OP_CUSTOM,"unsupported custom metadata");
            bytes(t->op_params,sizeof(t->op_params));
        }
        for (const auto* src : t->src) {
            const int source = !src ? -2 : index.count(src) ? index.at(src) : -1;
            bytes(&source,sizeof(source));
            if (src && source == -1) {
                bytes(src->ne,sizeof(src->ne)); bytes(&src->type,sizeof(src->type));
                const char* name = ggml_get_name(src); const size_t size = std::strlen(name);
                label_bytes(&size,sizeof(size)); label_bytes(name,size);
                identity->leaf_names.emplace_back(name);
                identity->leaf_cache_roles.push_back(cache_roles && cache_roles->count(src) ? cache_roles->at(src) : -1);
            }
        }
    }
    identity->graph_fingerprint = hash;
}
bool same_graph(const Capture& a, const Capture& b) {
    return a.nodes == b.nodes && a.graph_metadata == b.graph_metadata &&
           a.callback_identities == b.callback_identities;
}
void require_same_graph(const Capture& a, const Capture& b, const char* message) {
    if (same_graph(a,b)) return;
    size_t cache_labels = 0, other_labels = 0;
    const bool roles_equal = a.leaf_cache_roles == b.leaf_cache_roles;
    if (a.leaf_names.size() == b.leaf_names.size()) {
        for (size_t i = 0; i < a.leaf_names.size(); ++i) {
            if (a.leaf_names[i] == b.leaf_names[i]) continue;
            if (roles_equal && a.leaf_cache_roles[i] >= 0) ++cache_labels;
            else ++other_labels;
        }
    }
    std::printf("{\"record\":\"graphMismatch\",\"nodesExact\":%s,\"metadataWithoutLeafNamesExact\":%s,"
                "\"callbackIdentitiesExact\":%s,\"leafCountsExact\":%s,\"leafCacheRolesExact\":%s,"
                "\"cacheLeafNameDifferences\":%zu,\"otherLeafNameDifferences\":%zu}\n",
                a.nodes==b.nodes?"true":"false",a.metadata_without_leaf_names==b.metadata_without_leaf_names?"true":"false",
                a.callback_identities==b.callback_identities?"true":"false",a.leaf_names.size()==b.leaf_names.size()?"true":"false",
                roles_equal?"true":"false",cache_labels,other_labels);
    std::fflush(stdout);
    require(false,message);
}
size_t raw_custom_differences(const Capture& a, const Capture& b) {
    require(a.raw_custom_parameters.size() == b.raw_custom_parameters.size(),"same custom node count");
    size_t count = 0;
    for (size_t i = 0; i < a.raw_custom_parameters.size(); ++i)
        for (size_t j = 0; j < GGML_MAX_OP_PARAMS; ++j)
            count += a.raw_custom_parameters[i][j] != b.raw_custom_parameters[i][j];
    return count;
}
void metadata_noop(ggml_tensor*,const ggml_tensor*,const ggml_tensor*,int,int,void*) {}
void metadata_other_noop(ggml_tensor*,const ggml_tensor*,const ggml_tensor*,int,int,void*) {}
int audit_metadata() {
    auto ctx = mt::make_ctx(ggml_tensor_overhead()*32+ggml_graph_overhead_custom(32,false)*8+4096,true);
    require(bool(ctx),"metadata context");
    auto* a = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,16,1);
    auto* b = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,128,1);
    ggml_set_name(a,"metadata-scores"); ggml_set_name(b,"metadata-query");
    auto* x = ggml_map_custom2(ctx.get(),a,b,metadata_noop,16,nullptr);
    auto* y = ggml_map_custom2(ctx.get(),a,b,metadata_noop,16,nullptr);
    auto graph = [&](ggml_tensor* t) {
        auto* g = ggml_new_graph_custom(ctx.get(),32,false); ggml_build_forward_expand(g,t); return g;
    };
    auto* gx = graph(x); auto* gy = graph(y);
    auto identity = [&](ggml_cgraph* g) { Capture r; r.nodes = ggml_graph_n_nodes(g); identify_graph(g,&r); return r; };
    const auto baseline = identity(gx);
    ggml_map_custom2_op_params params; std::memcpy(&params,y->op_params,sizeof(params));
    std::memset(y->op_params,0xa5,sizeof(y->op_params));
    std::memcpy(reinterpret_cast<uint8_t*>(y->op_params)+offsetof(ggml_map_custom2_op_params,fun),&params.fun,sizeof(params.fun));
    std::memcpy(reinterpret_cast<uint8_t*>(y->op_params)+offsetof(ggml_map_custom2_op_params,n_tasks),&params.n_tasks,sizeof(params.n_tasks));
    std::memcpy(reinterpret_cast<uint8_t*>(y->op_params)+offsetof(ggml_map_custom2_op_params,userdata),&params.userdata,sizeof(params.userdata));
    const auto padded = identity(gy);
    require(same_graph(baseline,padded) && raw_custom_differences(baseline,padded)>0,"padding cannot change semantics");
    std::printf("{\"record\":\"metadataControl\",\"paddingIgnored\":true,\"ignoredRawBytes\":%zu}\n",raw_custom_differences(baseline,padded));
    params.n_tasks = 7; std::memcpy(y->op_params,&params,sizeof(params));
    require(!same_graph(baseline,identity(gy)),"task-count difference rejected");
    params.n_tasks = 16; params.fun = metadata_other_noop; std::memcpy(y->op_params,&params,sizeof(params));
    require(!same_graph(baseline,identity(gy)),"callback-identity difference rejected");
    params.fun = metadata_noop; params.userdata = y; std::memcpy(y->op_params,&params,sizeof(params));
    bool rejected = false; try { (void)identity(gy); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected,"non-null userdata rejected");
    params.userdata = nullptr; std::memcpy(y->op_params,&params,sizeof(params));
    const auto stride = y->nb[1]; y->nb[1] += sizeof(float);
    require(!same_graph(baseline,identity(gy)),"stride difference rejected"); y->nb[1] = stride;
    ++y->ne[1]; require(!same_graph(baseline,identity(gy)),"shape difference rejected"); --y->ne[1];
    auto* w = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,128,16);
    ggml_set_name(w,"metadata-weight");
    auto* m1 = ggml_mul_mat(ctx.get(),w,b); auto* m2 = ggml_mul_mat(ctx.get(),w,b);
    ggml_mul_mat_set_prec(m2,GGML_PREC_F32);
    require(!same_graph(identity(graph(m1)),identity(graph(m2))),"precision difference rejected");
    std::printf("{\"record\":\"metadataControl\",\"taskCountRejected\":true,\"callbackIdentityRejected\":true,"
                "\"userdataRejected\":true,\"strideRejected\":true,\"shapeRejected\":true,\"precisionRejected\":true}\n");
    return 0;
}
const ggml_tensor* storage(const ggml_tensor* t) {
    while (t && t->view_src) t = t->view_src;
    return t;
}
}

namespace mt {
class CpuTargetBatchAudit {
public:
    static void copy_state(const Qwen3Decoder& from, Qwen3Decoder& to) {
        require(from.max_seq_ == to.max_seq_ && from.hp_.n_layers == to.hp_.n_layers,
                "state clone dimensions");
        for (int l = 0; l < from.hp_.n_layers; ++l) {
            copy_tensor(from.k_cache_[l],to.k_cache_[l]);
            copy_tensor(from.v_cache_[l],to.v_cache_[l]);
        }
        to.past_len_ = from.past_len_;
    }
    static Bits cache(const Qwen3Decoder& d, int positions) {
        require(positions >= 0 && positions <= d.past_len_, "active cache positions");
        Bits out;
        const int hd = d.hp_.head_dim, heads = d.hp_.n_kv_heads;
        for (int l = 0; l < d.hp_.n_layers; ++l) {
            for (int kind = 0; kind < 2; ++kind) {
                auto* t = kind ? d.v_cache_[l] : d.k_cache_[l];
                require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[3] == 1,
                        "contiguous F32 cache");
                Bits raw(ggml_nbytes(t)/sizeof(uint32_t));
                ggml_backend_tensor_get(t,raw.data(),0,ggml_nbytes(t));
                if (!kind) {
                    require(t->ne[0] == hd && t->ne[1] == heads && t->ne[2] == d.max_seq_, "K layout");
                    out.insert(out.end(),raw.begin(),raw.begin()+(size_t)positions*hd*heads);
                } else {
                    require(t->ne[0] == d.max_seq_ && t->ne[1] == hd && t->ne[2] == heads,"opt48 V layout");
                    for (int p = 0; p < positions; ++p)
                        for (int h = 0; h < heads; ++h)
                            for (int f = 0; f < hd; ++f)
                                out.push_back(raw[((size_t)h*hd+f)*d.max_seq_+p]);
                }
            }
        }
        return out;
    }
    static bool append(Qwen3Decoder& d, const std::vector<float>& x, int tokens,
                       bool capture, Capture* result, std::vector<float>* hidden) {
        require(result && hidden && tokens > 0 && x.size() == (size_t)tokens*d.hp_.hidden &&
                tokens <= d.max_seq_-d.past_len_, "append dimensions");
        result->tokens = tokens; result->kv = d.past_len_+tokens; result->heads = d.hp_.n_heads;
        std::vector<std::array<ggml_tensor*,StageCount>> tensors;
        std::unordered_map<const ggml_tensor*,int> cache_roles;
        for (int l = 0; l < d.hp_.n_layers; ++l) {
            cache_roles[d.k_cache_[l]] = 2*l; cache_roles[d.v_cache_[l]] = 2*l+1;
        }
        Qwen3Decoder::AuditHook hook = [&](ggml_cgraph* graph, bool before) {
            if (before) {
                result->nodes = ggml_graph_n_nodes(graph);
                identify_graph(graph,result,&cache_roles);
                if (!capture) return;
                tensors = discover(d,graph);
                for (const auto& layer : tensors)
                    for (auto* t : layer) {
                        require(t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t), "capture tensor layout");
                        ggml_set_output(t);
                    }
            } else if (capture) {
                result->layers.resize(tensors.size());
                for (size_t l = 0; l < tensors.size(); ++l)
                    for (int s = 0; s < StageCount; ++s) {
                        auto* t = tensors[l][s]; auto& values = result->layers[l][s];
                        values.resize(ggml_nelements(t));
                        require(values.size()*sizeof(float) == ggml_nbytes(t), "contiguous capture bytes");
                        ggml_backend_tensor_get(t,values.data(),0,ggml_nbytes(t));
                    }
            }
        };
        return d.run(x,tokens,hidden,&hook);
    }
private:
    static void copy_tensor(const ggml_tensor* from, ggml_tensor* to) {
        require(from->type == to->type && ggml_are_same_shape(from,to) &&
                ggml_nbytes(from) == ggml_nbytes(to), "cache clone shape");
        std::vector<uint8_t> raw(ggml_nbytes(from)), check(raw.size());
        ggml_backend_tensor_get(from,raw.data(),0,raw.size());
        ggml_backend_tensor_set(to,raw.data(),0,raw.size());
        ggml_backend_tensor_get(to,check.data(),0,check.size());
        require(raw == check, "cache clone byte readback");
    }
    static std::vector<std::array<ggml_tensor*,StageCount>> discover(const Qwen3Decoder& d, ggml_cgraph* graph) {
        std::vector<std::array<ggml_tensor*,StageCount>> result(d.hp_.n_layers);
        auto unique = [&](auto predicate) {
            ggml_tensor* found = nullptr;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto* t = ggml_graph_node(graph,i);
                if (!predicate(t)) continue;
                require(!found, "unique stage node"); found = t;
            }
            require(found, "stage node found"); return found;
        };
        for (int l = 0; l < d.hp_.n_layers; ++l) {
            const auto& w = d.layers_[l]; auto& p = result[l];
            auto matmul = [&](ggml_tensor* weight) {
                return unique([&](ggml_tensor* t) { return t->op == GGML_OP_MUL_MAT && t->src[0] == weight; });
            };
            p[Q] = matmul(w.attn_q); p[K] = matmul(w.attn_k); p[V] = matmul(w.attn_v);
            p[Norm] = p[Q]->src[1];
            p[Scores] = unique([&](ggml_tensor* t) { return t->op == GGML_OP_MUL_MAT && storage(t->src[0]) == d.k_cache_[l]; });
            require(p[Scores]->src[1]->op == GGML_OP_PERMUTE, "query permutation");
            p[QRope] = p[Scores]->src[1]->src[0];
            auto* k_store = unique([&](ggml_tensor* t) { return t->op == GGML_OP_CPY && storage(t->src[1]) == d.k_cache_[l]; });
            p[KRope] = k_store->src[0];
            require(p[QRope]->op == GGML_OP_ROPE && p[KRope]->op == GGML_OP_ROPE, "RoPE stages");
            p[QNorm] = p[QRope]->src[0]; p[KNorm] = p[KRope]->src[0];
            p[Probabilities] = unique([&](ggml_tensor* t) {
                return (t->op == GGML_OP_SOFT_MAX || t->op == GGML_OP_MAP_CUSTOM2) && t->src[0] == p[Scores];
            });
            p[AttentionProjection] = matmul(w.attn_o); p[Context] = p[AttentionProjection]->src[1];
            p[Gate] = matmul(w.ffn_gate); p[Up] = matmul(w.ffn_up); p[Down] = matmul(w.ffn_down);
            p[FfnNorm] = p[Gate]->src[1]; p[Activation] = p[Down]->src[1];
            require(p[FfnNorm]->src[0]->op == GGML_OP_RMS_NORM, "FFN norm input");
            p[Residual] = p[FfnNorm]->src[0]->src[0];
            p[LayerOutput] = unique([&](ggml_tensor* t) {
                return t->op == GGML_OP_ADD && t->src[0] == p[Residual] && t->src[1] == p[Down];
            });
        }
        return result;
    }
};
}

namespace {
std::vector<float> stage_row(const Capture& c, int layer, Stage stage, int query, int valid) {
    const auto& x = c.layers[layer][stage];
    if (stage != Scores && stage != Probabilities) {
        require(x.size()%c.tokens == 0, "token-major stage rows");
        return row(x,query,int(x.size()/c.tokens));
    }
    require(valid > 0 && valid <= c.kv && x.size() == (size_t)c.kv*c.tokens*c.heads, "attention stage rows");
    std::vector<float> out;
    out.reserve((size_t)valid*c.heads);
    for (int head = 0; head < c.heads; ++head) {
        const auto begin = x.begin()+((size_t)head*c.tokens+query)*c.kv;
        out.insert(out.end(),begin,begin+valid);
    }
    return out;
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && !std::strcmp(argv[1],"--audit-metadata")) return audit_metadata();
        require(argc == 2, "usage: moss_target_batch_trace MODEL");
        env("MTD_DEVICE","cpu"); env("MTD_CPU_OPT","48"); env("MTD_THREADS","16");
        env("OMP_NUM_THREADS","16"); env("OMP_DYNAMIC","FALSE");
        for (const char* key : {"MTD_THREADS_WHISPER","MTD_THREADS_ADAPTOR","MTD_THREADS_PREFILL","MTD_THREADS_DECODE","MTD_THREADS_LOGITS"}) env(key,"16");
        mt::ModelLoader model; require(model.load(argv[1]), "model load"); model.promote_small_f16_to_f32();
        require(std::string(mt::backend_name()) == "CPU" && mt::cpu_thread_count() == 16, "CPU budget");
        const auto& cfg = model.config(); const int hidden = cfg.text_hidden, max_seq = 1088;
        require(hidden > 0 && cfg.text_vocab > 2 && cfg.text_layers > 0, "model dimensions");
        struct Weight { ggml_tensor* t; std::vector<uint8_t> bytes; };
        std::vector<Weight> weights; size_t weight_bytes = 0;
        for (const auto& name : model.tensor_names()) {
            auto* t = model.tensor(name); require(t, "weight tensor");
            Weight w{t,std::vector<uint8_t>(ggml_nbytes(t))};
            ggml_backend_tensor_get(t,w.bytes.data(),0,w.bytes.size()); weight_bytes += w.bytes.size(); weights.push_back(std::move(w));
        }
        struct Case { int mode, prefix, count; };
        std::vector<Case> cases;
        for (const Case& c : {Case{0,32,4},{0,32,8},{0,512,8},{0,1023,8},
                             {1,512,8},{1,1024,4},{1,1024,8},{1,1025,8}}) cases.push_back(c);
        for (const Case& c : {Case{0,32,1},{0,32,2},{0,512,1},{0,512,2},{0,1023,1},{0,1023,2},
                             {1,512,1},{1,512,2},{1,1024,1},{1,1024,2},{1,1025,1},{1,1025,2},
                             {0,31,4},{0,31,8},{0,129,8},{0,1024,4},{1,32,4},{1,1023,8}}) cases.push_back(c);
        size_t complete = 0, capture_changes = 0, quiet_drift_cases = 0, stage_records = 0;
        size_t future_nonzero = 0, nonfinite = 0;
        for (const auto& c : cases) {
            uint32_t seed = 0x12345678u;
            auto next = [&]() { seed = seed*1664525u+1013904223u; return seed; };
            std::vector<float> prefix_x;
            if (!c.mode) {
                prefix_x.resize((size_t)c.prefix*hidden);
                for (auto& f : prefix_x) f = (int(next() >> 8)-8388608)*(0.2f/8388608.f);
            } else {
                std::vector<int32_t> ids(c.prefix); for (auto& id : ids) id = next()%cfg.text_vocab;
                require(mt::embed_rows_f32(model.tensor("token_embd.weight"),ids.data(),c.prefix,hidden,&prefix_x), "prefix embeddings");
            }
            std::vector<int32_t> ids(c.count); for (int i = 0; i < c.count; ++i) ids[i] = (int64_t(i)*104729+41)%cfg.text_vocab;
            std::vector<float> append_x;
            require(mt::embed_rows_f32(model.tensor("token_embd.weight"),ids.data(),c.count,hidden,&append_x), "append embeddings");
            const auto saved_prefix = prefix_x, saved_append = append_x;
            mt::Qwen3Decoder serial,batch,traced_serial,traced_batch;
            for (auto* d : {&serial,&batch,&traced_serial,&traced_batch}) require(d->load(model,max_seq), "decoder load");
            std::vector<float> prefix_hidden;
            require(serial.prefill(prefix_x,c.prefix,&prefix_hidden), "prefix prefill");
            for (auto* d : {&batch,&traced_serial,&traced_batch}) mt::CpuTargetBatchAudit::copy_state(serial,*d);
            const auto initial = mt::CpuTargetBatchAudit::cache(serial,c.prefix);
            for (auto* d : {&batch,&traced_serial,&traced_batch}) require(initial == mt::CpuTargetBatchAudit::cache(*d,c.prefix), "copied prefix state");
            std::vector<float> quiet_serial,quiet_batch,captured_serial,captured_batch;
            std::vector<Capture> quiet_steps(c.count), captured_steps(c.count);
            size_t ignored_metadata_bytes = 0;
            for (int i = 0; i < c.count; ++i) {
                std::vector<float> h; require(mt::CpuTargetBatchAudit::append(serial,row(append_x,i,hidden),1,false,&quiet_steps[i],&h), "quiet serial append");
                quiet_serial.insert(quiet_serial.end(),h.begin(),h.end());
            }
            Capture quiet_group,captured_group;
            require(mt::CpuTargetBatchAudit::append(batch,append_x,c.count,false,&quiet_group,&quiet_batch), "quiet batch append");
            for (int i = 0; i < c.count; ++i) {
                std::vector<float> h; require(mt::CpuTargetBatchAudit::append(traced_serial,row(append_x,i,hidden),1,true,&captured_steps[i],&h), "captured serial append");
                require_same_graph(quiet_steps[i],captured_steps[i],"same serial semantic graph/callback identities");
                ignored_metadata_bytes += raw_custom_differences(quiet_steps[i],captured_steps[i]);
                captured_serial.insert(captured_serial.end(),h.begin(),h.end());
            }
            require(mt::CpuTargetBatchAudit::append(traced_batch,append_x,c.count,true,&captured_group,&captured_batch), "captured batch append");
            require_same_graph(quiet_group,captured_group,"same batch semantic graph/callback identities");
            ignored_metadata_bytes += raw_custom_differences(quiet_group,captured_group);
            Delta quiet_delta,serial_capture,batch_capture;
            quiet_delta.add(quiet_serial,quiet_batch); serial_capture.add(quiet_serial,captured_serial); batch_capture.add(quiet_batch,captured_batch);
            const auto serial_cache = mt::CpuTargetBatchAudit::cache(serial,c.prefix+c.count);
            const auto batch_cache = mt::CpuTargetBatchAudit::cache(batch,c.prefix+c.count);
            const size_t quiet_kv = differences(serial_cache,batch_cache);
            const size_t serial_capture_kv = differences(serial_cache,mt::CpuTargetBatchAudit::cache(traced_serial,c.prefix+c.count));
            const size_t batch_capture_kv = differences(batch_cache,mt::CpuTargetBatchAudit::cache(traced_batch,c.prefix+c.count));
            const bool unchanged_capture = !serial_capture.bits && !batch_capture.bits && !serial_capture_kv && !batch_capture_kv;
            capture_changes += !unchanged_capture; quiet_drift_cases += quiet_delta.bits || quiet_kv;
            nonfinite += quiet_delta.nonfinite+serial_capture.nonfinite+batch_capture.nonfinite;
            int first_layer = -1, first_stage = -1;
            size_t case_future_nonzero = 0;
            for (int l = 0; l < cfg.text_layers; ++l) {
                for (int s = 0; s < StageCount; ++s) {
                    Delta delta; size_t elements = 0;
                    for (int i = 0; i < c.count; ++i) {
                        auto a = stage_row(captured_steps[i],l,Stage(s),0,c.prefix+i+1);
                        auto b = stage_row(captured_group,l,Stage(s),i,c.prefix+i+1);
                        elements += a.size(); delta.add(a,b);
                    }
                    if (delta.bits && first_layer == -1) { first_layer = l; first_stage = s; }
                    nonfinite += delta.nonfinite; ++stage_records;
                    std::printf("{\"record\":\"stage\",\"mode\":%d,\"prefix\":%d,\"appendTokens\":%d,\"layer\":%d,\"stage\":%d,"
                        "\"stageName\":\"%s\",\"elements\":%zu,\"floatBitDifferences\":%zu,\"maxAbs\":%.17g,\"nonfiniteElements\":%zu}\n",
                        c.mode,c.prefix,c.count,l,s,names[s],elements,delta.bits,delta.max_abs,delta.nonfinite);
                }
                const auto& p = captured_group.layers[l][Probabilities];
                for (int i = 0; i < c.count; ++i)
                    for (int h = 0; h < captured_group.heads; ++h)
                        for (int k = c.prefix+i+1; k < captured_group.kv; ++k) {
                            const float value = p[((size_t)h*c.count+i)*captured_group.kv+k];
                            case_future_nonzero += value != 0.0f || !std::isfinite(value);
                        }
            }
            future_nonzero += case_future_nonzero;
            for (auto* d : {&serial,&batch,&traced_serial,&traced_batch}) {
                require(d->past_len() == c.prefix+c.count && initial == mt::CpuTargetBatchAudit::cache(*d,c.prefix), "positions and immutable prefix");
            }
            require(!std::memcmp(prefix_x.data(),saved_prefix.data(),prefix_x.size()*sizeof(float)) &&
                    !std::memcmp(append_x.data(),saved_append.data(),append_x.size()*sizeof(float)), "immutable inputs");
            ++complete;
            std::printf("{\"record\":\"case\",\"mode\":%d,\"prefix\":%d,\"appendTokens\":%d,\"firstDifferingLayer\":%d,\"firstDifferingStage\":%d,"
                "\"quietHiddenBitDifferences\":%zu,\"quietKvBitDifferences\":%zu,\"serialCaptureHiddenBitDifferences\":%zu,\"batchCaptureHiddenBitDifferences\":%zu,"
                "\"serialCaptureKvBitDifferences\":%zu,\"batchCaptureKvBitDifferences\":%zu,\"capturePreservesQuietResults\":%s,"
                "\"batchGraphFingerprint\":\"%016llx\",\"batchGraphNodes\":%d,\"futureProbabilityNonzeroElements\":%zu,"
                "\"ignoredCustomMetadataByteDifferences\":%zu,\"semanticGraphMetadataExact\":true,\"callbackIdentitiesExact\":true,"
                "\"prefixAndInputsUnchanged\":true,\"positionStateExact\":true,\"timingPerformed\":false}\n",
                c.mode,c.prefix,c.count,first_layer,first_stage,quiet_delta.bits,quiet_kv,serial_capture.bits,batch_capture.bits,
                serial_capture_kv,batch_capture_kv,unchanged_capture?"true":"false",
                (unsigned long long)quiet_group.graph_fingerprint,quiet_group.nodes,case_future_nonzero,ignored_metadata_bytes);
            std::fflush(stdout);
        }
        size_t changed_weights = 0;
        for (const auto& w : weights) {
            std::vector<uint8_t> actual(w.bytes.size()); ggml_backend_tensor_get(w.t,actual.data(),0,actual.size());
            for (size_t i = 0; i < actual.size(); ++i) changed_weights += actual[i] != w.bytes[i];
        }
        const bool valid = complete == 26 && !capture_changes && !future_nonzero && !nonfinite && !changed_weights;
        std::printf("{\"record\":\"summary\",\"cases\":%zu,\"stageRecords\":%zu,\"quietDriftCases\":%zu,\"captureChangedCases\":%zu,"
            "\"futureProbabilityNonzeroElements\":%zu,\"nonfiniteElements\":%zu,\"modelWeightBytesChecked\":%zu,\"modelWeightByteChanges\":%zu,"
            "\"traceAttributionEligible\":%s,\"timingPerformed\":false,\"productionPromoted\":false}\n",
            complete,stage_records,quiet_drift_cases,capture_changes,future_nonzero,nonfinite,weight_bytes,changed_weights,valid?"true":"false");
        return valid ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr,"target batch trace: %s\n",e.what()); return 2;
    }
}
