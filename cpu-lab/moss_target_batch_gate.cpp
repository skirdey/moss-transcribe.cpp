// MIT. Model-backed target-batch arithmetic/state audit. Numeric output only.
// This executable owns all decoder states and serializes every backend call.
#include "backend.hpp"
#include "cpu_context.hpp"
#include "generate.hpp"
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

int main(int argc, char** argv) {
    try {
        require(argc == 2 || (argc == 3 && !std::strcmp(argv[2],"--causal-context")),
                "usage: moss_target_batch_gate MODEL [--causal-context]");
        const bool causal_context = argc == 3;
        if (causal_context) std::printf("{\"record\":\"configuration\",\"referenceCpuOpt\":48,"
            "\"candidateCpuOpt\":262192,\"candidateScope\":\"batchAppendOnly\"}\n");
        env("MTD_DEVICE", "cpu"); env("MTD_CPU_OPT", "48"); env("MTD_THREADS", "16");
        env("OMP_NUM_THREADS", "16"); env("OMP_DYNAMIC", "FALSE");
        for (const char* name : {"MTD_THREADS_WHISPER", "MTD_THREADS_ADAPTOR",
             "MTD_THREADS_PREFILL", "MTD_THREADS_DECODE", "MTD_THREADS_LOGITS"}) env(name, "16");
        mt::ModelLoader model;
        require(model.load(argv[1]), "model load"); model.promote_small_f16_to_f32();
        require(std::string(mt::backend_name()) == "CPU" && mt::cpu_thread_count() == 16,
                "fixed CPU backend and thread budget");
        const auto& cfg = model.config();
        const int hidden = cfg.text_hidden, vocab = cfg.text_vocab, max_seq = 1088;
        require(hidden > 0 && vocab > 2 && cfg.eos_token_id >= 0 && cfg.eos_token_id < vocab,
                "model dimensions and EOS");
        std::vector<WeightSnapshot> weights;
        size_t model_bytes = 0;
        for (const auto& name : model.tensor_names()) {
            auto* tensor = model.tensor(name); require(tensor, "model tensor snapshot");
            WeightSnapshot saved{tensor, std::vector<uint8_t>(ggml_nbytes(tensor))};
            ggml_backend_tensor_get(tensor, saved.bytes.data(), 0, saved.bytes.size());
            model_bytes += saved.bytes.size(); weights.push_back(std::move(saved));
        }
        size_t cases = 0, exact_cases = 0, t1_exact = 0, drift_cases = 0;
        size_t hidden_bits = 0, logit_bits = 0, kv_bits = 0, rollback_bits = 0;
        size_t all_nonfinite = 0, argmax_mismatches = 0, eos_mismatches = 0;
        const int prefixes[] = {31,32,33,127,128,129,511,512,513,1023,1024,1025};
        for (int mode = 0; mode < 2; ++mode) {
            for (int prefix : prefixes) {
                std::vector<float> prefix_x;
                uint32_t seed = 0x12345678u;
                auto next = [&]() { seed = seed*1664525u+1013904223u; return seed; };
                if (mode == 0) {
                    prefix_x.resize((size_t)prefix*hidden);
                    for (auto& f : prefix_x) f = (int(next() >> 8)-8388608)*(0.2f/8388608.f);
                } else {
                    std::vector<int32_t> ids(prefix);
                    for (auto& id : ids) id = int32_t(next()%vocab);
                    require(mt::embed_rows_f32(model.tensor("token_embd.weight"), ids.data(), prefix,
                            hidden, &prefix_x), "real prefix token embeddings");
                }
                std::vector<int32_t> tail_ids(8);
                for (int i = 0; i < 8; ++i) tail_ids[i] = (int64_t(i)*104729+41)%vocab;
                std::vector<float> tail;
                require(mt::embed_rows_f32(model.tensor("token_embd.weight"), tail_ids.data(), 8,
                        hidden, &tail), "real append token embeddings");
                const auto saved_prefix = prefix_x, saved_tail = tail;
                for (int count : {1,2,4,8}) {
                    mt::Qwen3Decoder serial, batch;
                    require(serial.load(model, max_seq) && batch.load(model, max_seq), "owned decoder load");
                    std::vector<float> serial_prefix, batch_prefix;
                    require(serial.prefill(prefix_x, prefix, &serial_prefix) &&
                            batch.prefill(prefix_x, prefix, &batch_prefix), "paired prefix prefill");
                    FloatDiff prime; prime.add(serial_prefix, batch_prefix);
                    require(!prime.bits && !prime.nonfinite, "identical prefix hidden states");
                    const auto initial_cache = mt::CpuTargetBatchAudit::cache(serial, prefix);
                    require(initial_cache == mt::CpuTargetBatchAudit::cache(batch, prefix),
                            "identical prefix KV states");
                    std::vector<float> serial_hidden, batch_hidden;
                    const std::vector<float> append_x(tail.begin(), tail.begin()+(size_t)count*hidden);
                    const auto saved_append = append_x;
                    for (int i = 0; i < count; ++i) {
                        auto h = serial.decode_one(row(append_x, i, hidden));
                        require(h.size() == (size_t)hidden, "sequential decode shape");
                        serial_hidden.insert(serial_hidden.end(), h.begin(), h.end());
                    }
                    const auto before_context = mt::cpu_causal_context_counts();
                    if (causal_context) env("MTD_CPU_OPT","262192");
                    require(mt::CpuTargetBatchAudit::append(batch, append_x, count, &batch_hidden),
                            "batch append");
                    if (causal_context) env("MTD_CPU_OPT","48");
                    const auto after_context = mt::cpu_causal_context_counts();
                    const uint64_t context_ops = causal_context && count>1 ? cfg.text_layers : 0;
                    require(after_context.built-before_context.built == context_ops &&
                            after_context.executed-before_context.executed == context_ops,
                            "actual candidate context dispatch count");
                    if (causal_context) std::printf("{\"record\":\"dispatchAudit\",\"mode\":%d,\"prefix\":%d,"
                        "\"appendTokens\":%d,\"contextOpsBuilt\":%llu,\"contextOpsExecuted\":%llu}\n",
                        mode,prefix,count,(unsigned long long)context_ops,(unsigned long long)context_ops);
                    require(serial.past_len() == prefix+count && batch.past_len() == prefix+count,
                            "append position state");
                    FloatDiff hdiff, ldiff;
                    hdiff.add(serial_hidden, batch_hidden);
                    size_t greedy_bad = 0, eos_bad = 0;
                    for (int i = 0; i < count; ++i) {
                        auto a = serial.logits_from_hidden(row(serial_hidden, i, hidden));
                        auto b = batch.logits_from_hidden(row(batch_hidden, i, hidden));
                        require(a.size() == (size_t)vocab && b.size() == (size_t)vocab, "full logits shape");
                        ldiff.add(a,b); const int ai = argmax(a), bi = argmax(b);
                        greedy_bad += ai != bi;
                        eos_bad += (ai == cfg.eos_token_id) != (bi == cfg.eos_token_id);
                    }
                    const size_t active_bad = differences(mt::CpuTargetBatchAudit::cache(serial, prefix+count),
                                                         mt::CpuTargetBatchAudit::cache(batch, prefix+count));
                    size_t prefix_bad = differences(initial_cache, mt::CpuTargetBatchAudit::cache(serial, prefix)) +
                                        differences(initial_cache, mt::CpuTargetBatchAudit::cache(batch, prefix));
                    const int accepted = count/2, rewind_position = prefix+accepted;
                    require(!mt::CpuTargetBatchAudit::rewind(batch, -1) && batch.past_len() == prefix+count &&
                            !mt::CpuTargetBatchAudit::rewind(batch, prefix+count+1) && batch.past_len() == prefix+count,
                            "invalid rewinds leave state unchanged");
                    require(mt::CpuTargetBatchAudit::rewind(serial, rewind_position) &&
                            mt::CpuTargetBatchAudit::rewind(batch, rewind_position), "accepted prefix rewind");
                    const auto serial_accepted = mt::CpuTargetBatchAudit::cache(serial, rewind_position);
                    const auto batch_accepted = mt::CpuTargetBatchAudit::cache(batch, rewind_position);
                    const size_t poisoned = mt::CpuTargetBatchAudit::poison_discarded_suffix(batch);
                    require(poisoned > 0 && batch_accepted == mt::CpuTargetBatchAudit::cache(batch, rewind_position),
                            "poison leaves accepted cache untouched");
                    const int32_t changed_id = (tail_ids[accepted]+1)%vocab;
                    auto changed = mt::embed_token(model, changed_id, hidden);
                    require(changed.size() == (size_t)hidden && !same(changed, row(tail, accepted, hidden)),
                            "changed rejected-token embedding");
                    const auto saved_changed = changed;
                    auto ah = serial.decode_one(changed), bh = batch.decode_one(changed);
                    FloatDiff rh, rl; rh.add(ah,bh);
                    auto al = serial.logits_from_hidden(ah), bl = batch.logits_from_hidden(bh);
                    require(al.size() == (size_t)vocab && bl.size() == (size_t)vocab, "rollback full logits shape");
                    rl.add(al,bl); greedy_bad += argmax(al) != argmax(bl);
                    eos_bad += (argmax(al) == cfg.eos_token_id) != (argmax(bl) == cfg.eos_token_id);
                    require(serial.past_len() == rewind_position+1 && batch.past_len() == rewind_position+1,
                            "changed decode position");
                    const size_t rollback_cache_bad = differences(mt::CpuTargetBatchAudit::cache(serial, rewind_position+1),
                                                                 mt::CpuTargetBatchAudit::cache(batch, rewind_position+1));
                    require(serial_accepted == mt::CpuTargetBatchAudit::cache(serial, rewind_position) &&
                            batch_accepted == mt::CpuTargetBatchAudit::cache(batch, rewind_position),
                            "changed decode leaves accepted cache untouched");
                    prefix_bad += differences(initial_cache, mt::CpuTargetBatchAudit::cache(serial, prefix)) +
                                  differences(initial_cache, mt::CpuTargetBatchAudit::cache(batch, prefix));
                    require(!prefix_bad && same(prefix_x,saved_prefix) && same(tail,saved_tail) &&
                            same(append_x,saved_append) && same(changed,saved_changed), "prefix and input immutability");
                    const size_t rb = rh.bits+rl.bits+rollback_cache_bad;
                    const size_t nf = hdiff.nonfinite+ldiff.nonfinite+rh.nonfinite+rl.nonfinite;
                    const bool exact = !hdiff.bits && !ldiff.bits && !active_bad && !rb && !nf && !greedy_bad && !eos_bad;
                    ++cases; exact_cases += exact; drift_cases += !exact; t1_exact += count == 1 && exact;
                    hidden_bits += hdiff.bits; logit_bits += ldiff.bits; kv_bits += active_bad;
                    rollback_bits += rb; all_nonfinite += nf; argmax_mismatches += greedy_bad; eos_mismatches += eos_bad;
                    std::printf("{\"record\":\"case\",\"mode\":%d,\"prefix\":%d,\"appendTokens\":%d,\"acceptedTokens\":%d,"
                        "\"hiddenElements\":%zu,\"logitElements\":%zu,\"hiddenBitDifferences\":%zu,\"logitBitDifferences\":%zu,"
                        "\"hiddenMaxAbs\":%.17g,\"logitMaxAbs\":%.17g,\"activeKvBitDifferences\":%zu,\"prefixKvBitDifferences\":%zu,"
                        "\"rollbackHiddenBitDifferences\":%zu,\"rollbackLogitBitDifferences\":%zu,\"rollbackKvBitDifferences\":%zu,"
                        "\"rollbackHiddenMaxAbs\":%.17g,\"rollbackLogitMaxAbs\":%.17g,\"nonfiniteElements\":%zu,"
                        "\"argmaxMismatches\":%zu,\"eosMismatches\":%zu,\"poisonedSuffixElements\":%zu,"
                        "\"initialPrefixExact\":true,\"inputsUnchanged\":true,\"acceptedCacheUnchanged\":true,"
                        "\"invalidRewindsRejected\":true,\"positionStateExact\":true,\"exact\":%s}\n",
                        mode,prefix,count,accepted,serial_hidden.size(),(size_t)count*vocab,hdiff.bits,ldiff.bits,
                        hdiff.max_abs,ldiff.max_abs,active_bad,prefix_bad,rh.bits,rl.bits,rollback_cache_bad,
                        rh.max_abs,rl.max_abs,nf,greedy_bad,eos_bad,poisoned,exact ? "true":"false");
                    std::fflush(stdout);
                }
            }
        }
        size_t weight_changes = 0;
        for (const auto& saved : weights) {
            std::vector<uint8_t> actual(saved.bytes.size());
            ggml_backend_tensor_get(saved.tensor, actual.data(), 0, actual.size());
            for (size_t i = 0; i < actual.size(); ++i) weight_changes += actual[i] != saved.bytes[i];
        }
        const bool gate = cases == 96 && exact_cases == cases && t1_exact == 24 && !weight_changes;
        std::printf("{\"record\":\"summary\",\"cases\":%zu,\"exactCases\":%zu,\"driftCases\":%zu,\"exactT1Controls\":%zu,"
            "\"hiddenBitDifferences\":%zu,\"logitBitDifferences\":%zu,\"activeKvBitDifferences\":%zu,\"rollbackBitDifferences\":%zu,"
            "\"nonfiniteElements\":%zu,\"argmaxMismatches\":%zu,\"eosMismatches\":%zu,\"modelWeightBytesChecked\":%zu,"
            "\"modelWeightByteChanges\":%zu,\"cpuOpt\":48,\"cpuThreadBudget\":16,\"maxSequence\":1088,"
            "\"parityGatePassed\":%s,\"timingEligible\":%s,\"timingPerformed\":false,\"productionPromoted\":false}\n",
            cases,exact_cases,drift_cases,t1_exact,hidden_bits,logit_bits,kv_bits,rollback_bits,all_nonfinite,
            argmax_mismatches,eos_mismatches,model_bytes,weight_changes,gate ? "true":"false",gate ? "true":"false");
        return gate ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "target batch gate: %s\n", e.what()); return 2;
    }
}
