// MIT. Actual target/rollback controls for strict speculative generation.
// This executable owns all decoder states and serializes every backend call.
#include "backend.hpp"
#include "cpu_context.hpp"
#include "generate.hpp"
#include "cpu_speculative.hpp"
#include "qwen3_decoder.hpp"

#include <algorithm>
#include <cmath>
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
        require(argc==2,"usage: moss_speculative_gate MODEL");
        env("MTD_DEVICE","cpu");env("MTD_CPU_OPT","48");env("MTD_THREADS","16");env("OMP_NUM_THREADS","16");env("OMP_DYNAMIC","FALSE");env("MTD_TRACE_TOKENS","0");
        for(const char* name:{"MTD_THREADS_WHISPER","MTD_THREADS_ADAPTOR","MTD_THREADS_PREFILL","MTD_THREADS_DECODE","MTD_THREADS_LOGITS"})env(name,"16");
        mt::ModelLoader model;require(model.load(argv[1]),"model load");model.promote_small_f16_to_f32();
        require(std::string(mt::backend_name())=="CPU" && mt::cpu_thread_count()==16,"CPU budget");
        const auto& cfg=model.config();const int hidden=cfg.text_hidden,vocab=cfg.text_vocab;
        std::vector<WeightSnapshot> weights;size_t model_bytes=0;
        for(const auto& name:model.tensor_names()) {
            auto* t=model.tensor(name);WeightSnapshot w{t,std::vector<uint8_t>(ggml_nbytes(t))};
            ggml_backend_tensor_get(t,w.bytes.data(),0,w.bytes.size());model_bytes+=w.bytes.size();weights.push_back(std::move(w));
        }
        size_t cases=0,exact_cases=0,all_hidden_bits=0,all_logit_bits=0,all_kv_bits=0,all_token_bits=0,eos_after_batch=0;
        for(int prefix:{31,128,513,1024})for(const char* penalty:{"1.0","1.1","1.5"}) {
            env("MTD_REPETITION_PENALTY",penalty);env("MTD_CPU_OPT","48");
            std::vector<int32_t> prime_ids(prefix);uint32_t seed=0x12345678u;
            for(auto& id:prime_ids){seed=seed*1664525u+1013904223u;id=int32_t(seed%vocab);}
            std::vector<float> prime_x;require(mt::embed_rows_f32(model.tensor("token_embd.weight"),prime_ids.data(),prefix,hidden,&prime_x),"prime rows");
            const auto saved_prime=prime_x;mt::Qwen3Decoder gold_state;
            require(gold_state.load(model,1088),"gold state");
            const auto gold=mt::greedy_generate(gold_state,model,prime_x,prefix,10,-1);
            require(gold.size()==10,"complete controlled gold");
            int distinct=-1;for(int i=1;i<5;++i)if(std::find(gold.begin(),gold.begin()+i,gold[i])==gold.begin()+i){distinct=i;break;}
            // Five rejection/all-accepted lengths, three token limits, then
            // first-token EOS and EOS inside the first accepted batch.
            for(int scenario=0;scenario<10;++scenario) {
                if(scenario==9 && distinct<0)continue;
                const int forced_accept=scenario<5?scenario:4;
                const int limit=scenario==5?1:scenario==6?2:scenario==7?5:10;
                const int eos=scenario==8?gold[0]:scenario==9?gold[distinct]:-1;
                mt::Qwen3Decoder serial,candidate;env("MTD_CPU_OPT","48");
                require(serial.load(model,1088)&&candidate.load(model,1088),"owned generation states");
                std::vector<float> serial_prime;require(serial.prefill(prime_x,prefix,&serial_prime),"reference prefill");
                const auto prefix_kv=mt::CpuTargetBatchAudit::cache(serial,prefix);
                const auto initial_raw=serial.logits_from_hidden(row(serial_prime,prefix-1,hidden));
                std::vector<float> branch_hidden;int branch_start=prefix;
                size_t hidden_bits=0,logit_bits=0,kv_bits=0,nonfinite=0,appends=0,batches=0,poisoned=0,commits=0;
                bool first_draft=true,stopped=false;size_t expected_ops=0;
                mt::CpuSpeculativeHooks hooks;
                hooks.draft=[&](const std::vector<int32_t>& ids,int budget) {
                    if(!first_draft)return std::vector<int32_t>{};first_draft=false;
                    const int count=std::min({4,budget,int(gold.size()-ids.size())});
                    std::vector<int32_t> proposal(gold.begin()+ids.size(),gold.begin()+ids.size()+count);
                    if(forced_accept<count)proposal[forced_accept]=(proposal[forced_accept]+1)%vocab;
                    return proposal;
                };
                hooks.prefilled=[&](const std::vector<float>& actual) {
                    FloatDiff d;d.add(serial_prime,actual);hidden_bits+=d.bits;nonfinite+=d.nonfinite;
                    kv_bits+=differences(prefix_kv,mt::CpuTargetBatchAudit::cache(candidate,prefix));
                };
                hooks.appended=[&](const std::vector<int32_t>& inputs,const std::vector<float>& actual) {
                    branch_start=serial.past_len();branch_hidden.clear();++appends;batches+=inputs.size()>1;
                    expected_ops+=inputs.size()>1?cfg.text_layers:0;
                    env("MTD_CPU_OPT","48");
                    for(size_t i=0;i<inputs.size();++i) {
                        auto h=serial.decode_one(mt::embed_token(model,inputs[i],hidden));
                        require(h.size()==size_t(hidden),"serial branch shape");
                        branch_hidden.insert(branch_hidden.end(),h.begin(),h.end());
                        auto a=serial.logits_from_hidden(h),b=candidate.logits_from_hidden(row(actual,int(i),hidden));
                        require(a.size()==size_t(vocab)&&b.size()==size_t(vocab),"all full-vocabulary logits");
                        FloatDiff d;d.add(a,b);logit_bits+=d.bits;nonfinite+=d.nonfinite;
                    }
                    FloatDiff d;d.add(branch_hidden,actual);hidden_bits+=d.bits;nonfinite+=d.nonfinite;
                    kv_bits+=differences(mt::CpuTargetBatchAudit::cache(serial,serial.past_len()),mt::CpuTargetBatchAudit::cache(candidate,candidate.past_len()));
                    env("MTD_CPU_OPT","262192");
                };
                hooks.committed=[&](mt::Qwen3Decoder& d,const std::vector<int32_t>& ids,const std::vector<float>& raw,bool stop) {
                    ++commits;stopped=stop;
                    require(d.past_len()==prefix+int(ids.size())-(stop?1:0),"serial stopping/cache position");
                    require(mt::CpuTargetBatchAudit::rewind(serial,d.past_len()),"reference logical rewind");
                    kv_bits+=differences(mt::CpuTargetBatchAudit::cache(serial,serial.past_len()),mt::CpuTargetBatchAudit::cache(d,d.past_len()));
                    require(prefix_kv==mt::CpuTargetBatchAudit::cache(serial,prefix)&&prefix_kv==mt::CpuTargetBatchAudit::cache(d,prefix),"prefix immutability");
                    env("MTD_CPU_OPT","48");
                    auto reference=d.past_len()==prefix?initial_raw:serial.logits_from_hidden(row(branch_hidden,d.past_len()-branch_start-1,hidden));
                    FloatDiff diff;diff.add(reference,raw);logit_bits+=diff.bits;nonfinite+=diff.nonfinite;
                    // Poison the candidate's entire discarded capacity after
                    // every commit. Only accepted active state may survive.
                    poisoned+=mt::CpuTargetBatchAudit::poison_discarded_suffix(d);
                    env("MTD_CPU_OPT","262192");
                };
                const auto counts=mt::cpu_causal_context_counts();env("MTD_CPU_OPT","262192");
                auto actual=mt::cpu_speculative_generate(candidate,model,prime_x,prefix,limit,eos,&hooks);
                const auto after=mt::cpu_causal_context_counts();env("MTD_CPU_OPT","48");
                std::vector<int32_t> expected;
                for(int i=0;i<limit;++i){expected.push_back(gold[i]);if(gold[i]==eos)break;}
                const bool tokens=actual==expected;
                const bool ops=after.built-counts.built==expected_ops && after.executed-counts.executed==expected_ops;
                const bool pass=tokens && ops && stopped && commits && !hidden_bits && !logit_bits && !kv_bits && !nonfinite && same(prime_x,saved_prime);
                ++cases;exact_cases+=pass;all_hidden_bits+=hidden_bits;all_logit_bits+=logit_bits;all_kv_bits+=kv_bits;all_token_bits+=!tokens;
                eos_after_batch+=scenario==9 && batches && pass;
                std::printf("{\"record\":\"speculativeCase\",\"prefix\":%d,\"penalty\":%s,\"scenario\":%d,\"forcedAccepted\":%d,"
                    "\"maxNew\":%d,\"virtualEos\":%s,\"tokens\":%zu,\"targetCalls\":%zu,\"batchCalls\":%zu,\"commitChecks\":%zu,"
                    "\"hiddenBitDifferences\":%zu,\"logitBitDifferences\":%zu,\"activeKvBitDifferences\":%zu,\"nonfiniteElements\":%zu,"
                    "\"tokenSequenceExact\":%s,\"positionStateExact\":true,\"prefixAndInputsUnchanged\":true,\"poisonedDiscardedElements\":%zu,"
                    "\"contextOpsBuilt\":%llu,\"contextOpsExecuted\":%llu,\"dispatchExact\":%s,\"exact\":%s}\n",
                    prefix,penalty,scenario,forced_accept,limit,eos>=0?"true":"false",actual.size(),appends,batches,commits,hidden_bits,logit_bits,kv_bits,nonfinite,
                    tokens?"true":"false",poisoned,(unsigned long long)(after.built-counts.built),(unsigned long long)(after.executed-counts.executed),ops?"true":"false",pass?"true":"false");
                std::fflush(stdout);
            }
        }
        size_t weight_changes=0;
        for(const auto& w:weights){std::vector<uint8_t> actual(w.bytes.size());ggml_backend_tensor_get(w.tensor,actual.data(),0,actual.size());
            for(size_t i=0;i<actual.size();++i)weight_changes+=actual[i]!=w.bytes[i];}
        const bool passed=cases>=108 && exact_cases==cases && eos_after_batch>0 && !weight_changes;
        std::printf("{\"record\":\"speculativeSummary\",\"cases\":%zu,\"exactCases\":%zu,\"eosAfterBatchCases\":%zu,"
            "\"hiddenBitDifferences\":%zu,\"logitBitDifferences\":%zu,\"activeKvBitDifferences\":%zu,\"tokenMismatches\":%zu,"
            "\"modelWeightBytesChecked\":%zu,\"modelWeightByteChanges\":%zu,\"parityGatePassed\":%s,\"timingPerformed\":false}\n",
            cases,exact_cases,eos_after_batch,all_hidden_bits,all_logit_bits,all_kv_bits,all_token_bits,model_bytes,weight_changes,passed?"true":"false");
        return passed?0:1;
    }catch(const std::exception& e){std::fprintf(stderr,"speculative gate: %s\n",e.what());return 2;}
}
