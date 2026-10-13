// MIT. Protected attention capture on the actual audio/past-only draft boundary.
// No raw model/audio/token values are printed; no timing or production routing.
#include "backend.hpp"
#include "audio_io.hpp"
#include "audio_encoder.hpp"
#include "audio_span.hpp"
#include "tokenizer.hpp"
#include "cpu_speculative.hpp"
#include "cpu_context.hpp"
#include "generate.hpp"
#include "qwen3_decoder.hpp"
#include "ggml-impl.h" // Pinned custom-op field layout; exclude padding/pointers from hashing.
#include "ggml-cpu/vec.h" // Replay the very same F32 dot from the pinned CPU library.

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
bool inspect_context_operations = false;
enum Stage { Norm,Q,K,V,QNorm,KNorm,QRope,KRope,Scores,Probabilities,
             Context,AttentionProjection,Residual,FfnNorm,Gate,Up,Activation,Down,LayerOutput,StageCount };
constexpr const char* names[] = {"attentionNorm","queryProjection","keyProjection","valueProjection",
    "queryNorm","keyNorm","queryRope","keyRope","scores","probabilities","attentionContext",
    "attentionProjection","attentionResidual","ffnNorm","ffnGate","ffnUp","activation","ffnDown","layerOutput"};
struct Delta {
    size_t bits = 0, nonfinite = 0;
    double max_abs = 0;
    void one(const float& a, const float& b) {
        bits += std::memcmp(&a,&b,sizeof(float)) != 0;
        if (!std::isfinite(a) || !std::isfinite(b)) ++nonfinite;
        else max_abs = std::max(max_abs,std::abs(double(a)-double(b)));
    }
    void add(const std::vector<float>& a, const std::vector<float>& b) {
        require(a.size() == b.size() && !a.empty(), "float comparison shape");
        for (size_t i = 0; i < a.size(); ++i) {
            one(a[i],b[i]);
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
    std::vector<ggml_custom_op_t> generic_callback_identities;
    std::vector<std::array<uint8_t,GGML_MAX_OP_PARAMS>> raw_custom_parameters;
    struct Attention {
        int head_dim = 0, kv_heads = 0;
        std::vector<float> keys;    // [position, KV head, feature]
        std::vector<float> values;  // [KV head, feature, position]
        std::vector<float> raw_context; // Actual matmul: [head, query, feature]
    };
    std::vector<Attention> attention;
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
std::vector<float> raw_context_row(const Capture& c, int layer, int query) {
    require(layer >= 0 && (size_t)layer < c.attention.size() && query >= 0 && query < c.tokens,
            "raw context row bounds");
    const auto& a = c.attention[layer];
    require(a.head_dim > 0 && c.heads > 0 && a.raw_context.size() == (size_t)c.heads*c.tokens*a.head_dim,
            "raw context layout");
    std::vector<float> result;
    for (int head = 0; head < c.heads; ++head) {
        const auto begin = a.raw_context.begin()+((size_t)head*c.tokens+query)*a.head_dim;
        result.insert(result.end(),begin,begin+a.head_dim);
    }
    return result;
}
int kv_head(const Capture& c, const Capture::Attention& a, int head) {
    require(a.kv_heads > 0 && c.heads % a.kv_heads == 0 && head >= 0 && head < c.heads,
            "GQA head mapping");
    return head/(c.heads/a.kv_heads);
}
std::vector<float> replay_context_row(const Capture& c, int layer, int query, int length) {
    require(layer >= 0 && (size_t)layer < c.attention.size() && query >= 0 && query < c.tokens &&
            length > 0 && length <= c.kv, "replay row bounds");
    const auto& a = c.attention[layer];
    require(a.head_dim > 0 && a.values.size() == (size_t)a.kv_heads*a.head_dim*c.kv &&
            c.layers[layer][Probabilities].size() == (size_t)c.heads*c.tokens*c.kv,"replay operand layout");
    std::vector<float> result((size_t)c.heads*a.head_dim);
    for (int h = 0; h < c.heads; ++h) {
        const auto* p = c.layers[layer][Probabilities].data()+((size_t)h*c.tokens+query)*c.kv;
        const int k = kv_head(c,a,h);
        for (int f = 0; f < a.head_dim; ++f) {
            const auto* v = a.values.data()+((size_t)k*a.head_dim+f)*c.kv;
            ggml_vec_dot_f32(length,&result[(size_t)h*a.head_dim+f],0,v,0,p,0,1);
        }
    }
    return result;
}
void identify_graph(ggml_cgraph* graph, Capture* identity,
                    const std::unordered_map<const ggml_tensor*,int>* cache_roles = nullptr) {
    identity->graph_metadata.clear(); identity->callback_identities.clear(); identity->generic_callback_identities.clear();
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
        } else if (t->op == GGML_OP_CUSTOM) {
            ggml_custom_op_params params;
            std::memcpy(&params,t->op_params,sizeof(params));
            require(params.fun && !params.userdata,"supported null-userdata generic callback");
            identity->generic_callback_identities.push_back(params.fun);
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
           a.callback_identities == b.callback_identities && a.generic_callback_identities == b.generic_callback_identities && a.leaf_cache_roles == b.leaf_cache_roles;
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
                (a.callback_identities==b.callback_identities && a.generic_callback_identities==b.generic_callback_identities)?"true":"false",a.leaf_names.size()==b.leaf_names.size()?"true":"false",
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
    static bool append_plain(Qwen3Decoder& d,const std::vector<float>& x,int tokens,std::vector<float>* hidden) {
        return hidden && tokens>0 && x.size()==size_t(tokens)*d.hp_.hidden && tokens<=d.max_seq_-d.past_len_ && d.run(x,tokens,hidden);
    }
    static bool append(Qwen3Decoder& d, const std::vector<float>& x, int tokens,
                       bool capture, Capture* result, std::vector<float>* hidden) {
        require(result && hidden && tokens > 0 && x.size() == (size_t)tokens*d.hp_.hidden &&
                tokens <= d.max_seq_-d.past_len_, "append dimensions");
        result->tokens = tokens; result->kv = d.past_len_+tokens; result->heads = d.hp_.n_heads;
        std::vector<std::array<ggml_tensor*,StageCount>> tensors;
        std::vector<ggml_tensor*> raw_context;
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
                if (inspect_context_operations) {
                    for (size_t l = 0; l < tensors.size(); ++l) {
                        auto* ctx = tensors[l][Context];
                        require(ctx->op == GGML_OP_CONT && ctx->src[0]->op == GGML_OP_PERMUTE,
                                "context contiguous copy and permutation");
                        auto* mat = ctx->src[0]->src[0];
                        require(mat && (mat->op == GGML_OP_MUL_MAT || mat->op == GGML_OP_CUSTOM) && mat->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous(mat) && storage(mat->src[0]) == d.v_cache_[l] &&
                                mat->src[1] == tensors[l][Probabilities] && mat->ne[0] == d.hp_.head_dim &&
                                mat->ne[1] == tokens && mat->ne[2] == d.hp_.n_heads && mat->ne[3] == 1,
                                "actual value matmul layout and operands");
                        raw_context.push_back(mat); ggml_set_output(mat);
                    }
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
                if (inspect_context_operations) {
                    result->attention.resize(tensors.size());
                    for (size_t l = 0; l < tensors.size(); ++l) {
                        auto& a = result->attention[l]; a.head_dim = d.hp_.head_dim; a.kv_heads = d.hp_.n_kv_heads;
                        auto* mat = raw_context[l]; a.raw_context.resize(ggml_nelements(mat));
                        ggml_backend_tensor_get(mat,a.raw_context.data(),0,ggml_nbytes(mat));
                        const int kv = result->kv, hd = a.head_dim, heads = a.kv_heads;
                        auto* k = d.k_cache_[l]; auto* v = d.v_cache_[l];
                        require(ggml_is_contiguous(k) && ggml_is_contiguous(v) && k->type == GGML_TYPE_F32 &&
                                v->type == GGML_TYPE_F32 && k->ne[0] == hd && k->ne[1] == heads &&
                                k->ne[2] == d.max_seq_ && v->ne[0] == d.max_seq_ && v->ne[1] == hd &&
                                v->ne[2] == heads,"actual cache operand layout");
                        a.keys.resize((size_t)kv*hd*heads);
                        ggml_backend_tensor_get(k,a.keys.data(),0,a.keys.size()*sizeof(float));
                        std::vector<float> cache_values(ggml_nelements(v));
                        ggml_backend_tensor_get(v,cache_values.data(),0,ggml_nbytes(v));
                        a.values.reserve((size_t)kv*hd*heads);
                        for (int h = 0; h < heads; ++h)
                            for (int f = 0; f < hd; ++f) {
                                auto begin = cache_values.begin()+((size_t)h*hd+f)*d.max_seq_;
                                a.values.insert(a.values.end(),begin,begin+kv);
                            }
                    }
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
        // GGML gives unnamed leaves labels when their first graph is expanded.
        // A prefilling state and a freshly loaded clone have different histories.
        // Copy this descriptive metadata too, keeping label equality checked.
        ggml_set_name(to,ggml_get_name(from));
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

namespace {
void generic_metadata_noop(ggml_tensor*,int,int,void*) {}
void generic_metadata_other(ggml_tensor*,int,int,void*) {}
int audit_generic_metadata() {
    auto ctx=mt::make_ctx(1024*1024,true);require(bool(ctx),"metadata context");
    auto* a=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,32,8,2,1);
    auto* b=ggml_new_tensor_4d(ctx.get(),GGML_TYPE_F32,32,5,4,1);ggml_tensor* args[]={a,b};
    auto make=[&](int tasks,ggml_custom_op_t callback,void* data) {
        return ggml_custom_4d(ctx.get(),GGML_TYPE_F32,8,5,4,1,args,2,callback,tasks,data);
    };
    auto identity=[&](ggml_tensor* t) {
        auto* graph=ggml_new_graph_custom(ctx.get(),64,false);ggml_build_forward_expand(graph,t);
        Capture c;c.nodes=ggml_graph_n_nodes(graph);identify_graph(graph,&c);return c;
    };
    auto reference=identity(make(16,generic_metadata_noop,nullptr));
    require(same_graph(reference,identity(make(16,generic_metadata_noop,nullptr))),"generic semantic equality");
    require(!same_graph(reference,identity(make(8,generic_metadata_noop,nullptr))),"generic task difference rejected");
    require(!same_graph(reference,identity(make(16,generic_metadata_other,nullptr))),"generic callback difference rejected");
    int marker=0;bool rejected=false;try{(void)identity(make(16,generic_metadata_noop,&marker));}catch(const std::runtime_error&){rejected=true;}
    require(rejected,"generic userdata rejected");
    auto* changed=make(16,generic_metadata_noop,nullptr);changed->nb[1]+=sizeof(float);
    require(!same_graph(reference,identity(changed)),"generic stride difference rejected");
    changed=make(16,generic_metadata_noop,nullptr);changed->ne[1]=4;
    require(!same_graph(reference,identity(changed)),"generic shape difference rejected");
    std::printf("{\"record\":\"genericMetadataControl\",\"semanticEqualityExact\":true,\"taskCountRejected\":true,\"callbackIdentityRejected\":true,\"userdataRejected\":true,\"strideRejected\":true,\"shapeRejected\":true}\n");return 0;
}
std::vector<float> replay_softmax(const Capture& c,int layer,int query,int length,bool mask_future,int valid,int hd) {
    require(length>0 && length<=c.kv && valid<=length,"softmax replay bounds");
    std::vector<float> result(size_t(c.heads)*valid),work(length),output(length);
    const float scale=1.f/std::sqrt(float(hd));
    for(int head=0;head<c.heads;++head) {
        auto* raw=c.layers[layer][Scores].data()+(size_t(head)*c.tokens+query)*c.kv;
        ggml_vec_cpy_f32(length,work.data(),raw);ggml_vec_scale_f32(length,work.data(),scale);
        if(mask_future)for(int i=0;i<length;++i)work[i]+=i<valid?0.f:-INFINITY;
        float maximum=-INFINITY;ggml_vec_max_f32(length,&maximum,work.data());
        auto sum=ggml_vec_soft_max_f32(length,output.data(),work.data(),maximum);
        require(sum>0,"positive softmax replay sum");sum=1.0/sum;ggml_vec_scale_f32(length,output.data(),sum);
        std::copy(output.begin(),output.begin()+valid,result.begin()+size_t(head)*valid);
    }
    return result;
}
struct Weight {ggml_tensor* tensor;std::vector<uint8_t> bytes;};
}
int main(int argc,char** argv) {
    try {
        if(argc==2 && !std::strcmp(argv[1],"--audit-generic-metadata"))return audit_generic_metadata();
        require(argc==3,"usage: moss_audio_attention_trace MODEL PRIVATE_AUDIO");
        env("MTD_DEVICE","cpu");env("MTD_CPU_OPT","48");env("MTD_THREADS","16");env("OMP_NUM_THREADS","16");env("OMP_DYNAMIC","FALSE");env("MTD_TRACE_TOKENS","0");env("MTD_LOOP_GUARD","1");env("MTD_REPETITION_PENALTY","1.0");
        for(const char* name:{"MTD_THREADS_WHISPER","MTD_THREADS_ADAPTOR","MTD_THREADS_PREFILL","MTD_THREADS_DECODE","MTD_THREADS_LOGITS"})env(name,"16");
        mt::ModelLoader model;require(model.load(argv[1]),"model load");model.promote_small_f16_to_f32();
        require(std::string(mt::backend_name())=="CPU" && mt::cpu_thread_count()==16,"CPU budget");
        const auto& cfg=model.config();const int hidden=cfg.text_hidden,capacity=4971;
        std::vector<Weight> weights;size_t weight_bytes=0;
        for(const auto& name:model.tensor_names()) {auto* tensor=model.tensor(name);Weight w{tensor,std::vector<uint8_t>(ggml_nbytes(tensor))};ggml_backend_tensor_get(tensor,w.bytes.data(),0,w.bytes.size());weight_bytes+=w.bytes.size();weights.push_back(std::move(w));}
        mt::Audio audio;require(mt::load_audio_16k_mono(argv[2],audio),"audio load");mt::AudioEncoder encoder(model);int n_audio=0;
        auto audio_x=encoder.encode(audio.samples,n_audio,hidden);require(!audio_x.empty(),"audio encode");mt::Tokenizer tok;require(tok.load(model),"tokenizer load");
        auto input_ids=mt::build_input_ids(tok,cfg,cfg.default_prompt,n_audio);auto fused=mt::fuse_embeds(model,input_ids,audio_x,n_audio,hidden,cfg.audio_token_id);
        require(input_ids.size()==859 && !fused.empty(),"known actual audio prefix");
        mt::Qwen3Decoder prime;require(prime.load(model,capacity),"prime decoder");
        auto history=mt::greedy_generate(prime,model,fused,859,129,cfg.eos_token_id);
        require(history.size()==129 && prime.past_len()==987,"known seed excluded from cache");
        auto proposal=mt::cpu_ngram_draft(history,4);require(proposal.size()==4,"past-only five-input fixture");
        std::vector<int32_t> inputs{history.back()};inputs.insert(inputs.end(),proposal.begin(),proposal.end());
        std::vector<float> x;require(mt::embed_rows_f32(model.tensor("token_embd.weight"),inputs.data(),5,hidden,&x),"target inputs");
        const auto saved_x=x,saved_fused=fused;const int past=987,count=5;const auto prefix=mt::CpuTargetBatchAudit::cache(prime,past);
        mt::Qwen3Decoder plain_serial,plain_batch,quiet_serial,quiet_batch,captured_serial,captured_batch;
        for(auto* d:{&plain_serial,&plain_batch,&quiet_serial,&quiet_batch,&captured_serial,&captured_batch}) {require(d->load(model,capacity),"owned decoder");mt::CpuTargetBatchAudit::copy_state(prime,*d);require(prefix==mt::CpuTargetBatchAudit::cache(*d,past),"clone active cache");}
        std::vector<float> ps,pb,qs,qb,cs,cb;std::vector<Capture> quiet_steps(count),captured_steps(count);Capture quiet_group,captured_group;
        env("MTD_CPU_OPT","48");
        for(int q=0;q<count;++q){std::vector<float> h;require(mt::CpuTargetBatchAudit::append_plain(plain_serial,row(x,q,hidden),1,&h),"null-hook serial");ps.insert(ps.end(),h.begin(),h.end());}
        env("MTD_CPU_OPT","262192");const auto before_context=mt::cpu_causal_context_counts();require(mt::CpuTargetBatchAudit::append_plain(plain_batch,x,count,&pb),"null-hook batch");
        env("MTD_CPU_OPT","48");
        for(int q=0;q<count;++q){std::vector<float> h;require(mt::CpuTargetBatchAudit::append(quiet_serial,row(x,q,hidden),1,false,&quiet_steps[q],&h),"metadata serial");qs.insert(qs.end(),h.begin(),h.end());}
        env("MTD_CPU_OPT","262192");require(mt::CpuTargetBatchAudit::append(quiet_batch,x,count,false,&quiet_group,&qb),"metadata batch");
        inspect_context_operations=true;env("MTD_CPU_OPT","48");size_t padding_bytes=0;
        for(int q=0;q<count;++q){std::vector<float> h;require(mt::CpuTargetBatchAudit::append(captured_serial,row(x,q,hidden),1,true,&captured_steps[q],&h),"protected serial");require_same_graph(quiet_steps[q],captured_steps[q],"serial semantic graph");padding_bytes+=raw_custom_differences(quiet_steps[q],captured_steps[q]);cs.insert(cs.end(),h.begin(),h.end());}
        env("MTD_CPU_OPT","262192");require(mt::CpuTargetBatchAudit::append(captured_batch,x,count,true,&captured_group,&cb),"protected batch");require_same_graph(quiet_group,captured_group,"batch semantic graph");padding_bytes+=raw_custom_differences(quiet_group,captured_group);
        const auto after_context=mt::cpu_causal_context_counts();require(after_context.built-before_context.built==84 && after_context.executed-before_context.executed==84,"actual three-batch context dispatch");
        Delta serial_observer,batch_observer,serial_capture,batch_capture,drift,logit_drift;
        serial_observer.add(ps,qs);batch_observer.add(pb,qb);serial_capture.add(ps,cs);batch_capture.add(pb,cb);drift.add(ps,pb);
        auto serial_cache=mt::CpuTargetBatchAudit::cache(plain_serial,past+count),batch_cache=mt::CpuTargetBatchAudit::cache(plain_batch,past+count);
        size_t observer_kv=differences(serial_cache,mt::CpuTargetBatchAudit::cache(quiet_serial,past+count))+differences(batch_cache,mt::CpuTargetBatchAudit::cache(quiet_batch,past+count));
        size_t capture_kv=differences(serial_cache,mt::CpuTargetBatchAudit::cache(captured_serial,past+count))+differences(batch_cache,mt::CpuTargetBatchAudit::cache(captured_batch,past+count));
        env("MTD_CPU_OPT","48");size_t logit_capture_bits=0;int first_hidden_query=-1;
        for(int q=0;q<count;++q) {
            Delta h;h.add(row(ps,q,hidden),row(pb,q,hidden));if(h.bits && first_hidden_query<0)first_hidden_query=q;
            auto a=plain_serial.logits_from_hidden(row(ps,q,hidden)),b=plain_batch.logits_from_hidden(row(pb,q,hidden));logit_drift.add(a,b);
            for(auto pair:{std::pair<mt::Qwen3Decoder*,const std::vector<float>*>{&quiet_serial,&qs},{&captured_serial,&cs}}){Delta d;d.add(a,pair.first->logits_from_hidden(row(*pair.second,q,hidden)));logit_capture_bits+=d.bits;}
            for(auto pair:{std::pair<mt::Qwen3Decoder*,const std::vector<float>*>{&quiet_batch,&qb},{&captured_batch,&cb}}){Delta d;d.add(b,pair.first->logits_from_hidden(row(*pair.second,q,hidden)));logit_capture_bits+=d.bits;}
        }
        const bool observer_exact=!serial_observer.bits&&!batch_observer.bits&&!serial_capture.bits&&!batch_capture.bits&&!observer_kv&&!capture_kv&&!logit_capture_bits;
        require(observer_exact,"observer must preserve actual null-hook hidden/logit/cache results");require(first_hidden_query==3,"same real-audio hidden boundary");
        size_t stages=0,replay_failures=0,replay_operand_changes=0,nonfinite=0,future=0;int first_layer=-1,first_stage=-1,first_query=-1;
        for(int layer=0;layer<cfg.text_layers;++layer) {
            for(int stage=0;stage<StageCount;++stage)for(int q=0;q<count;++q) {
                auto a=stage_row(captured_steps[q],layer,Stage(stage),0,past+q+1),b=stage_row(captured_group,layer,Stage(stage),q,past+q+1);Delta d;d.add(a,b);nonfinite+=d.nonfinite;++stages;
                if(d.bits && first_layer<0){first_layer=layer;first_stage=stage;first_query=q;}
                std::printf("{\"record\":\"audioStage\",\"layer\":%d,\"stage\":%d,\"stageName\":\"%s\",\"query\":%d,\"validLength\":%d,\"elements\":%zu,\"floatBitDifferences\":%zu,\"maxAbs\":%.17g,\"nonfiniteElements\":%zu}\n",layer,stage,names[stage],q,past+q+1,a.size(),d.bits,d.max_abs,d.nonfinite);
            }
            const auto saved_scores=captured_group.layers[layer][Scores],saved_probabilities=captured_group.layers[layer][Probabilities],saved_values=captured_group.attention[layer].values;
            for(int q=0;q<count;++q) {
                const int valid=past+q+1,hd=captured_group.attention[layer].head_dim;const auto& sc=captured_steps[q];
                const auto saved_serial_scores=sc.layers[layer][Scores],saved_serial_probabilities=sc.layers[layer][Probabilities],saved_serial_values=sc.attention[layer].values;
                auto a=stage_row(sc,layer,Probabilities,0,valid),b=stage_row(captured_group,layer,Probabilities,q,valid);
                auto sr=replay_softmax(sc,layer,0,valid,false,valid,hd),br=replay_softmax(captured_group,layer,q,past+count,true,valid,hd),vr=replay_softmax(captured_group,layer,q,valid,false,valid,hd);
                Delta serial_replay,batch_replay,valid_serial,common_valid,context_serial,context_batch;
                serial_replay.add(a,sr);batch_replay.add(b,br);valid_serial.add(a,vr);common_valid.add(br,vr);
                context_serial.add(raw_context_row(sc,layer,0),replay_context_row(sc,layer,0,valid));context_batch.add(raw_context_row(captured_group,layer,q),replay_context_row(captured_group,layer,q,valid));
                for(const auto* d:{&serial_replay,&batch_replay,&valid_serial,&common_valid,&context_serial,&context_batch})nonfinite+=d->nonfinite;
                replay_failures+=bool(serial_replay.bits||batch_replay.bits||context_serial.bits||context_batch.bits);
                std::printf("{\"record\":\"audioAttentionReplay\",\"layer\":%d,\"query\":%d,\"validLength\":%d,\"commonLength\":%d,\"serialSoftmaxReplayBitDifferences\":%zu,\"batchSoftmaxCommonReplayBitDifferences\":%zu,\"batchSoftmaxValidReplayVsSerialBitDifferences\":%zu,\"batchCommonVsValidSoftmaxBitDifferences\":%zu,\"batchCommonVsValidSoftmaxMaxAbs\":%.17g,\"serialContextReplayBitDifferences\":%zu,\"batchValidContextReplayBitDifferences\":%zu}\n",layer,q,valid,past+count,serial_replay.bits,batch_replay.bits,valid_serial.bits,common_valid.bits,common_valid.max_abs,context_serial.bits,context_batch.bits);
                replay_operand_changes+=std::memcmp(saved_serial_scores.data(),sc.layers[layer][Scores].data(),saved_serial_scores.size()*sizeof(float))!=0;
                replay_operand_changes+=std::memcmp(saved_serial_probabilities.data(),sc.layers[layer][Probabilities].data(),saved_serial_probabilities.size()*sizeof(float))!=0;
                replay_operand_changes+=std::memcmp(saved_serial_values.data(),sc.attention[layer].values.data(),saved_serial_values.size()*sizeof(float))!=0;
                const auto& probabilities=captured_group.layers[layer][Probabilities];for(int head=0;head<captured_group.heads;++head)for(int k=valid;k<past+count;++k){float v=probabilities[(size_t(head)*count+q)*(past+count)+k];future+=v!=0.f || !std::isfinite(v);}
            }
            replay_operand_changes+=std::memcmp(saved_scores.data(),captured_group.layers[layer][Scores].data(),saved_scores.size()*sizeof(float))!=0;
            replay_operand_changes+=std::memcmp(saved_probabilities.data(),captured_group.layers[layer][Probabilities].data(),saved_probabilities.size()*sizeof(float))!=0;
            replay_operand_changes+=std::memcmp(saved_values.data(),captured_group.attention[layer].values.data(),saved_values.size()*sizeof(float))!=0;
        }
        require(!replay_operand_changes,"replay operands immutable");
        for(auto* d:{&plain_serial,&plain_batch,&quiet_serial,&quiet_batch,&captured_serial,&captured_batch})require(d->past_len()==past+count && prefix==mt::CpuTargetBatchAudit::cache(*d,past),"prefix/positions unchanged");
        require(!std::memcmp(x.data(),saved_x.data(),x.size()*sizeof(float)) && !std::memcmp(fused.data(),saved_fused.data(),fused.size()*sizeof(float)),"input bytes unchanged");
        size_t weight_changes=0;for(const auto& w:weights){std::vector<uint8_t> actual(w.bytes.size());ggml_backend_tensor_get(w.tensor,actual.data(),0,actual.size());for(size_t i=0;i<actual.size();++i)weight_changes+=actual[i]!=w.bytes[i];}
        const bool eligible=observer_exact && stages==size_t(cfg.text_layers)*StageCount*count && !replay_failures && !replay_operand_changes && !nonfinite && !future && !weight_changes;
        std::printf("{\"record\":\"audioAttentionSummary\",\"audioPrefixTokens\":859,\"cacheCapacity\":4971,\"generatedCommittedBefore\":128,\"past\":987,\"appendTokens\":5,\"contextOpsBuilt\":84,\"contextOpsExecuted\":84,\"firstHiddenDifferenceQuery\":%d,\"firstDifferingLayer\":%d,\"firstDifferingStage\":%d,\"firstDifferingQuery\":%d,\"stageRecords\":%zu,\"attentionReplayRecords\":%d,\"quietHiddenBitDifferences\":%zu,\"quietLogitBitDifferences\":%zu,\"quietKvBitDifferences\":%zu,\"nullHookVsMetadataObserverHiddenBitDifferences\":%zu,\"nullHookVsProtectedCaptureHiddenBitDifferences\":%zu,\"observerKvBitDifferences\":%zu,\"captureKvBitDifferences\":%zu,\"observerAndCaptureLogitBitDifferences\":%zu,\"semanticGraphMetadataExact\":true,\"callbackIdentitiesExact\":true,\"ignoredRawCustomParameterByteDifferences\":%zu,\"replayControlFailures\":%zu,\"replayOperandChanges\":%zu,\"futureProbabilityNonzeroElements\":%zu,\"nonfiniteElements\":%zu,\"prefixAndInputsUnchanged\":true,\"positionStateExact\":true,\"modelWeightBytesChecked\":%zu,\"modelWeightByteChanges\":%zu,\"attributionEligible\":%s,\"timingPerformed\":false,\"productionPromoted\":false}\n",first_hidden_query,first_layer,first_stage,first_query,stages,cfg.text_layers*count,drift.bits,logit_drift.bits,differences(serial_cache,batch_cache),serial_observer.bits+batch_observer.bits,serial_capture.bits+batch_capture.bits,observer_kv,capture_kv,logit_capture_bits,padding_bytes,replay_failures,replay_operand_changes,future,nonfinite,weight_bytes,weight_changes,eligible?"true":"false");
        return eligible?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"audio attention trace: %s\n",e.what());return 2;}
}
