// MIT. Stop at the first internal mismatch on an actual private audio prefix.
// This executable owns all decoder states and serializes every backend call.
#include "backend.hpp"
#include "audio_io.hpp"
#include "audio_encoder.hpp"
#include "audio_span.hpp"
#include "tokenizer.hpp"
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


struct FirstDifference {};
int main(int argc,char** argv) {
    try {
        require(argc==3,"usage: moss_speculative_audio_gate MODEL PRIVATE_AUDIO");
        env("MTD_DEVICE","cpu");env("MTD_CPU_OPT","48");env("MTD_THREADS","16");env("OMP_NUM_THREADS","16");env("OMP_DYNAMIC","FALSE");env("MTD_TRACE_TOKENS","0");
        env("MTD_REPETITION_PENALTY","1.0");env("MTD_LOOP_GUARD","1");
        for(const char* name:{"MTD_THREADS_WHISPER","MTD_THREADS_ADAPTOR","MTD_THREADS_PREFILL","MTD_THREADS_DECODE","MTD_THREADS_LOGITS"})env(name,"16");
        mt::ModelLoader model;require(model.load(argv[1]),"model load");model.promote_small_f16_to_f32();
        require(std::string(mt::backend_name())=="CPU" && mt::cpu_thread_count()==16,"CPU budget");
        const auto& cfg=model.config();const int hidden=cfg.text_hidden;
        std::vector<WeightSnapshot> weights;size_t model_bytes=0;
        for(const auto& name:model.tensor_names()) {
            auto* t=model.tensor(name);WeightSnapshot w{t,std::vector<uint8_t>(ggml_nbytes(t))};
            ggml_backend_tensor_get(t,w.bytes.data(),0,w.bytes.size());model_bytes+=w.bytes.size();weights.push_back(std::move(w));
        }
        mt::Audio audio;require(mt::load_audio_16k_mono(argv[2],audio),"audio load");
        mt::AudioEncoder encoder(model);int n_audio=0;
        auto audio_x=encoder.encode(audio.samples,n_audio,hidden);require(!audio_x.empty(),"audio encode");
        mt::Tokenizer tokenizer;require(tokenizer.load(model),"tokenizer load");
        auto prime_ids=mt::build_input_ids(tokenizer,cfg,cfg.default_prompt,n_audio);
        auto prime_x=mt::fuse_embeds(model,prime_ids,audio_x,n_audio,hidden,cfg.audio_token_id);
        const int prefix=int(prime_ids.size());require(!prime_x.empty(),"actual fused prefix");
        mt::Qwen3Decoder serial,candidate;require(serial.load(model,prefix+4096+16)&&candidate.load(model,prefix+4096+16),"paired decoder states");
        std::vector<float> serial_raw,branch_hidden,initial_raw;std::vector<int32_t> committed;
        Bits prefix_cache;int branch_start=prefix;size_t appends=0,commits=0;bool failed=false;
        mt::CpuSpeculativeHooks hooks;
        hooks.prefilled=[&](const std::vector<float>& h) {
            env("MTD_CPU_OPT","48");std::vector<float> reference;
            require(serial.prefill(prime_x,prefix,&reference),"reference prefill");
            require(same(h,reference),"actual prefix hidden parity");
            prefix_cache=mt::CpuTargetBatchAudit::cache(serial,prefix);
            require(prefix_cache==mt::CpuTargetBatchAudit::cache(candidate,prefix),"actual prefix cache parity");
            initial_raw=serial_raw=serial.logits_from_hidden(row(reference,prefix-1,hidden));env("MTD_CPU_OPT","262192");
        };
        hooks.appended=[&](const std::vector<int32_t>& inputs,const std::vector<float>& h) {
            ++appends;branch_start=serial.past_len();branch_hidden.clear();
            require(branch_start==candidate.past_len()-int(inputs.size()),"pre-append logical position");
            env("MTD_CPU_OPT","48");
            const bool seed_exact=inputs[0]==mt::cpu_greedy_choice(serial_raw,committed,1.f);
            for(size_t i=0;i<inputs.size();++i) {
                std::vector<float> x,reference;require(mt::embed_rows_f32(model.tensor("token_embd.weight"),&inputs[i],1,hidden,&x),"serial embedding");
                require(mt::CpuTargetBatchAudit::append(serial,x,1,&reference),"serial append");
                branch_hidden.insert(branch_hidden.end(),reference.begin(),reference.end());
                auto candidate_row=row(h,int(i),hidden);FloatDiff hidden_diff,logit_diff;
                hidden_diff.add(reference,candidate_row);
                auto a=serial.logits_from_hidden(reference),b=candidate.logits_from_hidden(candidate_row);logit_diff.add(a,b);
                if(hidden_diff.bits || logit_diff.bits || !seed_exact) {
                    auto cache_bits=differences(mt::CpuTargetBatchAudit::cache(serial,serial.past_len()),mt::CpuTargetBatchAudit::cache(candidate,serial.past_len()));
                    std::printf("{\"record\":\"audioFirstDifference\",\"prefixTokens\":%d,\"committedBefore\":%zu,\"appendCall\":%zu,\"appendTokens\":%zu,\"queryRow\":%zu,\"queryPosition\":%d,\"hiddenBitDifferences\":%zu,\"logitBitDifferences\":%zu,\"activeKvBitDifferences\":%zu,\"hiddenMaxAbs\":%.17g,\"logitMaxAbs\":%.17g,\"nonfiniteElements\":%zu,\"rowFirstMaximumMatches\":%s,\"seedFirstMaximumMatches\":%s}\n",prefix,committed.size(),appends,inputs.size(),i,branch_start+int(i),hidden_diff.bits,logit_diff.bits,cache_bits,hidden_diff.max_abs,logit_diff.max_abs,hidden_diff.nonfinite+logit_diff.nonfinite,argmax(a)==argmax(b)?"true":"false",seed_exact?"true":"false");
                    std::fflush(stdout);throw FirstDifference{};
                }
            }
            require(mt::CpuTargetBatchAudit::cache(serial,serial.past_len())==mt::CpuTargetBatchAudit::cache(candidate,candidate.past_len()),"active cache parity");
            env("MTD_CPU_OPT","262192");
        };
        hooks.committed=[&](mt::Qwen3Decoder& d,const std::vector<int32_t>& ids,const std::vector<float>& raw,bool stopped) {
            ++commits;require(d.past_len()==prefix+int(ids.size())-(stopped?1:0),"committed logical position");
            require(mt::CpuTargetBatchAudit::rewind(serial,d.past_len()),"reference rewind");
            env("MTD_CPU_OPT","48");
            serial_raw=d.past_len()==prefix?initial_raw:serial.logits_from_hidden(row(branch_hidden,d.past_len()-branch_start-1,hidden));
            require(same(serial_raw,raw),"committed raw logit parity");
            require(prefix_cache==mt::CpuTargetBatchAudit::cache(d,prefix),"prefix immutable");committed=ids;env("MTD_CPU_OPT","262192");
        };
        env("MTD_CPU_OPT","262192");
        try {mt::cpu_speculative_generate(candidate,model,prime_x,prefix,4096,cfg.eos_token_id,&hooks);}
        catch(const FirstDifference&){failed=true;}
        size_t changes=0;for(const auto& w:weights){std::vector<uint8_t> actual(w.bytes.size());ggml_backend_tensor_get(w.tensor,actual.data(),0,actual.size());for(size_t i=0;i<actual.size();++i)changes+=actual[i]!=w.bytes[i];}
        std::printf("{\"record\":\"audioDiagnosticSummary\",\"prefixTokens\":%d,\"appendCallsInspected\":%zu,\"exactCommitChecksBeforeStop\":%zu,\"stoppedAtFirstInternalDifference\":%s,\"modelWeightBytesChecked\":%zu,\"modelWeightByteChanges\":%zu,\"timingPerformed\":false,\"fullSequenceParityEstablished\":false}\n",prefix,appends,commits,failed?"true":"false",model_bytes,changes);
        return changes?2:failed?1:0;
    }catch(const std::exception& e){std::fprintf(stderr,"audio diagnostic: %s\n",e.what());return 2;}
}
