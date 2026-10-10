// MIT. Paired batch append plus full-vocabulary logits. Numeric output only.
// This executable owns all decoder states and serializes every backend call.
#include "backend.hpp"
#include "cpu_context.hpp"
#include "generate.hpp"
#include "qwen3_decoder.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void env(const char* name, const char* value) {
#ifdef _WIN32
    require(_putenv_s(name, value) == 0, "environment update");
#else
    require(setenv(name, value, 1) == 0, "environment update");
#endif
}
using Bits = std::vector<uint32_t>;
size_t differences(const Bits& a, const Bits& b) {
    require(a.size() == b.size(), "cache comparison size");
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
    return count;
}
bool same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size()*sizeof(float));
}
struct FloatDiff {
    size_t bits = 0, nonfinite = 0;
    double max_abs = 0;
    void add(const std::vector<float>& a, const std::vector<float>& b) {
        require(a.size() == b.size() && !a.empty(), "float comparison size");
        for (size_t i = 0; i < a.size(); ++i) {
            bits += std::memcmp(&a[i], &b[i], sizeof(float)) != 0;
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) ++nonfinite;
            else max_abs = std::max(max_abs, std::abs(double(a[i])-double(b[i])));
        }
    }
};
int argmax(const std::vector<float>& x) {
    require(!x.empty(), "nonempty logits");
    return int(std::max_element(x.begin(), x.end()) - x.begin());
}
std::vector<float> row(const std::vector<float>& x, int index, int hidden) {
    require(index >= 0 && (size_t)(index+1)*hidden <= x.size(), "embedding row bounds");
    return {x.begin()+(size_t)index*hidden, x.begin()+(size_t)(index+1)*hidden};
}
struct WeightSnapshot {
    ggml_tensor* tensor;
    std::vector<uint8_t> bytes;
};
}

namespace mt {
// Only this standalone executable defines the decoder's friend accessor.
class CpuTargetBatchAudit {
public:
    static bool append(Qwen3Decoder& d, const std::vector<float>& x, int tokens,
                       std::vector<float>* output) {
        if (!output || tokens <= 0 || x.size() != (size_t)tokens*d.hp_.hidden ||
            tokens > d.max_seq_-d.past_len_) return false;
        return d.run(x, tokens, output);
    }
    static bool rewind(Qwen3Decoder& d, int position) {
        if (position < 0 || position > d.past_len_) return false;
        d.past_len_ = position;
        return true;
    }
    // Canonical order: layer, K/V, position, KV head, head feature. Only active
    // positions are compared. Inactive bytes are copied as integers, not floats.
    static Bits cache(const Qwen3Decoder& d, int positions) {
        require(positions >= 0 && positions <= d.past_len_, "active cache bounds");
        Bits output;
        const size_t width = (size_t)d.hp_.head_dim*d.hp_.n_kv_heads;
        output.reserve(2*d.hp_.n_layers*(size_t)positions*width);
        for (int l = 0; l < d.hp_.n_layers; ++l) {
            for (int kind = 0; kind < 2; ++kind) {
                ggml_tensor* t = kind ? d.v_cache_[l] : d.k_cache_[l];
                validate(d, t, kind);
                Bits raw(ggml_nbytes(t)/sizeof(uint32_t));
                ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
                if (!kind || t->ne[0] == d.hp_.head_dim) {
                    output.insert(output.end(), raw.begin(), raw.begin()+(size_t)positions*width);
                } else {
                    for (int p = 0; p < positions; ++p)
                        for (int h = 0; h < d.hp_.n_kv_heads; ++h)
                            for (int f = 0; f < d.hp_.head_dim; ++f)
                                output.push_back(raw[((size_t)h*d.hp_.head_dim+f)*d.max_seq_+p]);
                }
            }
        }
        return output;
    }
    static size_t poison_discarded_suffix(Qwen3Decoder& d) {
        size_t count = 0;
        const size_t width = (size_t)d.hp_.head_dim*d.hp_.n_kv_heads;
        constexpr uint32_t quiet_nan = 0x7fc00000u;
        for (int l = 0; l < d.hp_.n_layers; ++l) {
            for (int kind = 0; kind < 2; ++kind) {
                ggml_tensor* t = kind ? d.v_cache_[l] : d.k_cache_[l];
                validate(d, t, kind);
                Bits raw(ggml_nbytes(t)/sizeof(uint32_t));
                ggml_backend_tensor_get(t, raw.data(), 0, ggml_nbytes(t));
                if (!kind || t->ne[0] == d.hp_.head_dim) {
                    std::fill(raw.begin()+(size_t)d.past_len_*width, raw.end(), quiet_nan);
                } else {
                    for (size_t r = 0; r < width; ++r)
                        std::fill(raw.begin()+r*d.max_seq_+d.past_len_,
                                  raw.begin()+(r+1)*d.max_seq_, quiet_nan);
                }
                count += (size_t)(d.max_seq_-d.past_len_)*width;
                ggml_backend_tensor_set(t, raw.data(), 0, ggml_nbytes(t));
                Bits check(raw.size());
                ggml_backend_tensor_get(t, check.data(), 0, ggml_nbytes(t));
                require(raw == check, "discarded suffix poison readback");
            }
        }
        return count;
    }
    struct State { int past; std::vector<Bits> raw; };
    static State save(const Qwen3Decoder& d) {
        State saved{d.past_len_,{}};
        for (int l=0;l<d.hp_.n_layers;++l) for (int kind=0;kind<2;++kind) {
            auto* t=kind?d.v_cache_[l]:d.k_cache_[l]; validate(d,t,kind);
            Bits raw(ggml_nbytes(t)/sizeof(uint32_t));
            ggml_backend_tensor_get(t,raw.data(),0,ggml_nbytes(t)); saved.raw.push_back(std::move(raw));
        }
        return saved;
    }
    static void restore(Qwen3Decoder& d,const State& saved) {
        require(saved.raw.size()==size_t(2*d.hp_.n_layers),"complete saved state");
        size_t index=0;
        for(int l=0;l<d.hp_.n_layers;++l) for(int kind=0;kind<2;++kind) {
            auto* t=kind?d.v_cache_[l]:d.k_cache_[l]; validate(d,t,kind);
            const auto& raw=saved.raw[index++]; require(raw.size()*sizeof(uint32_t)==ggml_nbytes(t),"state bytes");
            ggml_backend_tensor_set(t,raw.data(),0,ggml_nbytes(t));
        }
        d.past_len_=saved.past;
    }
private:
    static void validate(const Qwen3Decoder& d, const ggml_tensor* t, int kind) {
        require(t && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[3] == 1,
                "F32 contiguous cache");
        const bool ordinary = t->ne[0] == d.hp_.head_dim &&
            t->ne[1] == d.hp_.n_kv_heads && t->ne[2] == d.max_seq_;
        const bool transposed = kind && t->ne[0] == d.max_seq_ &&
            t->ne[1] == d.hp_.head_dim && t->ne[2] == d.hp_.n_kv_heads;
        require(ordinary || transposed, "cache layout");
        require(ggml_nbytes(t) == (size_t)d.max_seq_*d.hp_.head_dim*d.hp_.n_kv_heads*sizeof(float),
                "cache byte size");
    }
};
}

int main(int argc,char** argv) {
    try {
        require(argc==2,"usage: moss_causal_context_bench MODEL");
        env("MTD_DEVICE","cpu"); env("MTD_CPU_OPT","48"); env("MTD_THREADS","16");
        env("OMP_NUM_THREADS","16"); env("OMP_DYNAMIC","FALSE");
        for (const char* name : {"MTD_THREADS_WHISPER","MTD_THREADS_ADAPTOR",
             "MTD_THREADS_PREFILL","MTD_THREADS_DECODE","MTD_THREADS_LOGITS"}) env(name,"16");
        mt::ModelLoader model; require(model.load(argv[1]),"model load"); model.promote_small_f16_to_f32();
        require(std::string(mt::backend_name())=="CPU" && mt::cpu_thread_count()==16,"CPU thread budget");
        const auto& cfg=model.config(); const int hidden=cfg.text_hidden,vocab=cfg.text_vocab;
        require(hidden>0 && vocab>2,"model dimensions");
        std::vector<WeightSnapshot> weights; size_t model_bytes=0;
        for(const auto& name:model.tensor_names()) {
            auto* t=model.tensor(name); require(t,"model tensor"); WeightSnapshot w{t,std::vector<uint8_t>(ggml_nbytes(t))};
            ggml_backend_tensor_get(t,w.bytes.data(),0,w.bytes.size()); model_bytes+=w.bytes.size(); weights.push_back(std::move(w));
        }
        std::printf("{\"record\":\"configuration\",\"serialCpuOpt\":48,\"batchCpuOpt\":262192,"
            "\"threads\":16,\"pairsPerFixture\":5,\"warmupPairsPerFixture\":1,\"maxSequence\":1088,"
            "\"scope\":\"appendPlusEveryFullVocabularyLogit\",\"prefixAndStateRestoreTimed\":false,"
            "\"modelLoadTimed\":false,\"draftAndRollbackTimed\":false,\"fullInputLatencyMeasured\":false}\n");
        size_t fixtures=0,samples=0,exact=0;
        using Clock=std::chrono::steady_clock;
        for(int prefix:{32,128,512,1024}) for(int count:{2,4,8}) {
            uint32_t seed=0x12345678u; std::vector<int32_t> ids(prefix);
            for(auto& id:ids) {seed=seed*1664525u+1013904223u;id=int32_t(seed%vocab);}
            std::vector<float> prime_x; require(mt::embed_rows_f32(model.tensor("token_embd.weight"),ids.data(),prefix,hidden,&prime_x),"prefix embeddings");
            std::vector<int32_t> tail_ids(count);for(int i=0;i<count;++i)tail_ids[i]=(int64_t(i)*104729+41)%vocab;
            std::vector<float> append;require(mt::embed_rows_f32(model.tensor("token_embd.weight"),tail_ids.data(),count,hidden,&append),"append embeddings");
            const auto saved_prime_x=prime_x,saved_append=append;
            mt::Qwen3Decoder serial,batch; require(serial.load(model,1088) && batch.load(model,1088),"owned states");
            std::vector<float> prime_s,prime_b;require(serial.prefill(prime_x,prefix,&prime_s) && batch.prefill(prime_x,prefix,&prime_b),"prefix prefill");
            FloatDiff prime;prime.add(prime_s,prime_b);require(!prime.bits && !prime.nonfinite,"prefix exact");
            const auto prefix_bits=mt::CpuTargetBatchAudit::cache(serial,prefix);
            require(prefix_bits==mt::CpuTargetBatchAudit::cache(batch,prefix),"prefix cache exact");
            require(mt::CpuTargetBatchAudit::poison_discarded_suffix(serial)>0 && mt::CpuTargetBatchAudit::poison_discarded_suffix(batch)>0,"poison inactive capacity");
            const auto serial_start=mt::CpuTargetBatchAudit::save(serial),batch_start=mt::CpuTargetBatchAudit::save(batch);
            require(serial_start.raw==batch_start.raw && serial_start.past==prefix && batch_start.past==prefix,"complete start snapshot");
            for(int pair=-1;pair<5;++pair) {
                mt::CpuTargetBatchAudit::restore(serial,serial_start);mt::CpuTargetBatchAudit::restore(batch,batch_start);
                require(mt::CpuTargetBatchAudit::save(serial).raw==serial_start.raw && mt::CpuTargetBatchAudit::save(batch).raw==batch_start.raw,"complete restore readback");
                std::vector<float> hs,hb,ls,lb; double serial_seconds=0,batch_seconds=0;
                const bool batch_first=(pair>=0 && pair%2==1);
                auto run=[&](bool is_batch) {
                    auto& d=is_batch?batch:serial;auto& h=is_batch?hb:hs;auto& logits=is_batch?lb:ls;
                    env("MTD_CPU_OPT",is_batch?"262192":"48");const auto before=mt::cpu_causal_context_counts();
                    const auto start=Clock::now();
                    if(is_batch)require(mt::CpuTargetBatchAudit::append(d,append,count,&h),"batch append");
                    else for(int i=0;i<count;++i) {auto x=d.decode_one(row(append,i,hidden));require(x.size()==size_t(hidden),"serial shape");h.insert(h.end(),x.begin(),x.end());}
                    for(int i=0;i<count;++i) {auto x=d.logits_from_hidden(row(h,i,hidden));require(x.size()==size_t(vocab),"every full vocabulary");logits.insert(logits.end(),x.begin(),x.end());}
                    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
                    const auto after=mt::cpu_causal_context_counts();const uint64_t expected=is_batch?cfg.text_layers:0;
                    require(after.built-before.built==expected && after.executed-before.executed==expected,"actual context dispatch");
                    (is_batch?batch_seconds:serial_seconds)=elapsed;env("MTD_CPU_OPT","48");
                };
                run(batch_first);run(!batch_first);
                FloatDiff hdiff,ldiff;hdiff.add(hs,hb);ldiff.add(ls,lb);
                const size_t kv=differences(mt::CpuTargetBatchAudit::cache(serial,prefix+count),mt::CpuTargetBatchAudit::cache(batch,prefix+count));
                const size_t changed_prefix=differences(prefix_bits,mt::CpuTargetBatchAudit::cache(serial,prefix))+
                    differences(prefix_bits,mt::CpuTargetBatchAudit::cache(batch,prefix));
                require(!changed_prefix && same(prime_x,saved_prime_x) && same(append,saved_append),"immutable prefix and inputs");
                const bool pass=!hdiff.bits && !ldiff.bits && !kv && !hdiff.nonfinite && !ldiff.nonfinite && serial.past_len()==prefix+count && batch.past_len()==prefix+count;
                std::printf("{\"record\":\"%s\",\"prefix\":%d,\"appendTokens\":%d,\"pair\":%d,\"batchFirst\":%s,"
                    "\"serialSeconds\":%.17g,\"batchSeconds\":%.17g,\"speedup\":%.17g,\"hiddenBitDifferences\":%zu,"
                    "\"logitBitDifferences\":%zu,\"activeKvBitDifferences\":%zu,\"prefixKvBitDifferences\":%zu,"
                    "\"nonfiniteElements\":%zu,\"hiddenElements\":%zu,\"logitElements\":%zu,"
                    "\"contextOpsBuilt\":%d,\"contextOpsExecuted\":%d,\"restoredAllCacheBytes\":true,\"exact\":%s}\n",
                    pair<0?"warmup":"sample",prefix,count,pair,batch_first?"true":"false",serial_seconds,batch_seconds,serial_seconds/batch_seconds,
                    hdiff.bits,ldiff.bits,kv,changed_prefix,hdiff.nonfinite+ldiff.nonfinite,hs.size(),ls.size(),cfg.text_layers,cfg.text_layers,pass?"true":"false");
                std::fflush(stdout);require(pass,"timed state arithmetic drift");if(pair>=0){++samples;++exact;}
            }
            ++fixtures;
        }
        size_t weight_changes=0;
        for(const auto& w:weights) {std::vector<uint8_t> actual(w.bytes.size());ggml_backend_tensor_get(w.tensor,actual.data(),0,actual.size());
            for(size_t i=0;i<actual.size();++i)weight_changes+=actual[i]!=w.bytes[i];}
        const bool passed=fixtures==12 && samples==60 && exact==60 && !weight_changes;
        std::printf("{\"record\":\"summary\",\"fixtures\":%zu,\"warmupPairs\":12,\"measuredPairs\":%zu,"
            "\"exactMeasuredPairs\":%zu,\"modelWeightBytesChecked\":%zu,\"modelWeightByteChanges\":%zu,"
            "\"parityGatePassed\":%s,\"fullInputLatencyMeasured\":false,\"productionPromoted\":false,\"tenfoldAchieved\":false}\n",
            fixtures,samples,exact,model_bytes,weight_changes,passed?"true":"false");return passed?0:1;
    } catch(const std::exception& e) {std::fprintf(stderr,"causal context timing: %s\n",e.what());return 2;}
}
