#include <algorithm>
#include <cstdlib>
#include <deque>
#include <regex>
#include <unordered_set>
#include "tokenizer.hpp"
#include "generate.hpp"

#include "backend.hpp"
#include "common.hpp"
#include "ggml_extend.hpp"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <chrono>
#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace mt {

// Host-side fallback for embed_rows_f32: fetch each selected row's raw bytes
// from the (possibly device) weight buffer via ggml_backend_tensor_get and
// dequantize on the CPU with the type's to_float trait. This is the reference
// dequantization, so results match the get_rows graph path. Needed because
// ggml's GPU backends implement GET_ROWS only for a subset of quant types
// (CUDA has no K-quant get_rows kernels as of ggml 0.15) and submitting the
// op anyway GGML_ABORTs the whole process (mudler/LocalAI#10862).
static bool embed_rows_f32_host(struct ggml_tensor* tok, const int32_t* ids,
                                int n_ids, int hidden,
                                std::vector<float>* out) {
    const ggml_type type = tok->type;
    const ggml_type_traits* traits = ggml_get_type_traits(type);
    if (type != GGML_TYPE_F32 && (!traits || !traits->to_float)) {
        MT_LOGE("embed_rows_f32_host: no to_float for type %s",
                ggml_type_name(type));
        return false;
    }
    if (!tok->buffer) {
        MT_LOGE("embed_rows_f32_host: token_embd has no buffer");
        return false;
    }
    // The flat per-row offset math below needs dense rows.
    const size_t row_bytes = ggml_row_size(type, tok->ne[0]);
    if (tok->nb[1] != row_bytes) {
        MT_LOGE("embed_rows_f32_host: token_embd rows are not dense");
        return false;
    }
    out->resize((size_t)n_ids * (size_t)hidden);
    std::vector<uint8_t> raw(row_bytes);
    for (int p = 0; p < n_ids; ++p) {
        ggml_backend_tensor_get(tok, raw.data(), (size_t)ids[p] * row_bytes,
                                row_bytes);
        float* dst = out->data() + (size_t)p * (size_t)hidden;
        if (type == GGML_TYPE_F32) {
            std::memcpy(dst, raw.data(), (size_t)hidden * sizeof(float));
        } else {
            traits->to_float(raw.data(), dst, hidden);
        }
    }
    return true;
}

bool embed_rows_f32(struct ggml_tensor* tok, const int32_t* ids,
                    int n_ids, int hidden, std::vector<float>* out) {
    if (!tok || !ids || n_ids <= 0 || hidden <= 0 || !out) return false;

    // CPU buffers can be read directly. Dequantize only selected rows using
    // ggml's existing type trait; avoid graph allocation/dispatch per token.
    const char* opt = std::getenv("MTD_CPU_OPT");
    if (opt && (std::atoi(opt) & 2) && ggml_backend_is_cpu(backend()))
        return embed_rows_f32_host(tok, ids, n_ids, hidden, out);
    const size_t max_nodes = 8;
    size_t buf_sz = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead() +
                    (1u << 16);
    std::vector<uint8_t> buf(buf_sz);
    GgmlCtxPtr cctx = make_ctx_buf(buf.data(), buf.size(), /*no_alloc=*/true);
    ggml_context* ctx = cctx.get();
    if (!ctx) return false;

    struct ggml_tensor* idt = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_ids);
    ggml_set_input(idt);
    struct ggml_tensor* rows = ggml_get_rows(ctx, tok, idt);  // [hidden, n_ids] F32
    ggml_set_output(rows);

    // GPU backends support GET_ROWS only for a subset of src types (ggml's
    // CUDA backend aborts on K-quants), so ask first and dequantize the rows
    // on the host when the op can't run on the active backend.
    if (!ggml_backend_supports_op(backend(), rows)) {
        static std::once_flag warn_once;
        std::call_once(warn_once, [&] {
            MT_LOGW("embed lookup: %s GET_ROWS unsupported on %s backend, "
                    "using host-side dequant",
                    ggml_type_name(tok->type), backend_name());
        });
        return embed_rows_f32_host(tok, ids, n_ids, hidden, out);
    }

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, rows);

    const bool ok = compute_graph_with_inputs(gf, [&]() {
        ggml_backend_tensor_set(idt, ids, 0, (size_t)n_ids * sizeof(int32_t));
    });
    if (!ok) return false;
    return read_tensor_f32(rows, out);
}

std::vector<float> fuse_embeds(ModelLoader& m,
                               const std::vector<int32_t>& input_ids,
                               const std::vector<float>& audio_embeds,
                               int n_audio, int hidden, int audio_token_id) {
    std::vector<float> out;
    if (hidden <= 0) {
        MT_LOGE("fuse_embeds: invalid hidden=%d", hidden);
        return out;
    }
    struct ggml_tensor* tok = m.tensor("token_embd.weight");
    if (!tok) {
        MT_LOGE("fuse_embeds: missing token_embd.weight");
        return out;
    }
    // token_embd.weight ne=[hidden, vocab]: column t is the embedding of token
    // id t, laid out feature-fastest (token_embd_data[t*hidden + h]).
    if ((int)tok->ne[0] != hidden) {
        MT_LOGE("fuse_embeds: token_embd hidden %lld != %d",
                (long long)tok->ne[0], hidden);
        return out;
    }
    const int64_t vocab = tok->ne[1];
    const size_t  seq   = input_ids.size();
    if (seq == 0) return out;

    // Validate ids before the get_rows lookup (out-of-range ids would read
    // garbage rows).
    for (size_t p = 0; p < seq; ++p) {
        int32_t t = input_ids[p];
        if (t < 0 || t >= vocab) {
            MT_LOGE("fuse_embeds: input_ids[%zu]=%d out of range [0,%lld)",
                    p, t, (long long)vocab);
            return out;
        }
    }

    // 1) Embed lookup via ggml_get_rows: dequantizes rows for ANY token_embd
    //    type (F32/F16/quant) and returns F32 rows [hidden, seq]. On an F32
    //    token_embd this is bit-identical to the old raw column copy.
    if (!embed_rows_f32(tok, input_ids.data(), (int)seq, hidden, &out)) {
        MT_LOGE("fuse_embeds: get_rows lookup failed");
        out.clear();
        return out;
    }

    // 2) Audio injection (masked_scatter): walk positions in increasing index;
    //    the k-th position with id==audio_token_id gets audio_embeds row k.
    int k = 0;
    for (size_t p = 0; p < seq; ++p) {
        if (input_ids[p] != audio_token_id) continue;
        if (k >= n_audio) {
            MT_LOGE("fuse_embeds: more audio positions than audio rows "
                    "(n_audio=%d)", n_audio);
            out.clear();
            return out;
        }
        const float* src = audio_embeds.data() + (size_t)k * (size_t)hidden;
        float* dst = out.data() + p * (size_t)hidden;
        for (int h = 0; h < hidden; ++h) dst[h] = src[h];
        ++k;
    }
    if (k != n_audio) {
        MT_LOGW("fuse_embeds: consumed %d of %d audio rows", k, n_audio);
    }
    return out;
}

std::vector<float> embed_token(ModelLoader& m, int32_t t, int hidden) {
    std::vector<float> out;
    if (hidden <= 0) return out;
    struct ggml_tensor* tok = m.tensor("token_embd.weight");
    if (!tok) { MT_LOGE("embed_token: missing token_embd.weight"); return out; }
    if ((int)tok->ne[0] != hidden) {
        MT_LOGE("embed_token: token_embd hidden %lld != %d", (long long)tok->ne[0], hidden);
        return out;
    }
    const int64_t vocab = tok->ne[1];
    if (t < 0 || t >= vocab) {
        MT_LOGE("embed_token: id %d out of range [0,%lld)", (int)t, (long long)vocab);
        return out;
    }
    // Single-row lookup via ggml_get_rows: dequantizes for ANY token_embd type
    // (F32/F16/quant), F32 result [hidden]. Bit-identical to the old raw read
    // on an F32 token_embd.
    if (!embed_rows_f32(tok, &t, 1, hidden, &out)) {
        MT_LOGE("embed_token: get_rows lookup failed for id %d", (int)t);
        out.clear();
    }
    return out;
}

// argmax with FIRST-index-on-tie semantics (torch argmax): strict >.
static int argmax_first(const std::vector<float>& v) {
    int best = 0;
    for (int i = 1; i < (int)v.size(); ++i) if (v[i] > v[best]) best = i;
    return best;
}


// Dense empty timestamp/speaker turns are decoder loops. Lexical repetitions
// and event labels break the run and are never removed by this guard.
static bool empty_marker_loop(const std::string& text) {
    static const std::regex marker(R"(\[(\d+(?:\.\d+)?)\]\s*\[S\d+\]\s*\[(\d+(?:\.\d+)?)\])");
    std::deque<std::pair<double,double>> run;
    size_t previous_end = 0;
    for (std::sregex_iterator it(text.begin(), text.end(), marker), end; it != end; ++it) {
        const auto& match = *it;
        auto gap = text.substr(previous_end, static_cast<size_t>(match.position()) - previous_end);
        if (gap.find_first_not_of(" \t\r\n") != std::string::npos) run.clear();
        try { run.emplace_back(std::stod(match[1].str()), std::stod(match[2].str())); }
        catch (...) { run.clear(); }
        if (run.size() > 12) run.pop_front();
        previous_end = static_cast<size_t>(match.position() + match.length());
        if (run.size() == 12) {
            double lo = run.front().first, hi = lo;
            for (const auto& pair : run) { lo = std::min({lo,pair.first,pair.second}); hi = std::max({hi,pair.first,pair.second}); }
            if (hi - lo <= 2.0) return true;
        }
    }
    return false;
}

std::vector<int32_t> greedy_generate(Qwen3Decoder& dec, ModelLoader& m,
                                     const std::vector<float>& fused, int seq,
                                     int max_new, int eos) {
    std::vector<int32_t> ids;
    const int H = dec.hidden();
    if (H <= 0 || seq <= 0 || max_new <= 0) {
        MT_LOGE("greedy_generate: bad args (H=%d seq=%d max_new=%d)", H, seq, max_new);
        return ids;
    }

    using Clock = std::chrono::steady_clock;
    double embed_seconds = 0, decoder_seconds = 0, logits_seconds = 0;
    auto prefill_start = Clock::now();
    std::vector<float> hid;
    if (!dec.prefill(fused, seq, &hid)) { MT_LOGE("greedy_generate: prefill failed"); return ids; }
    if ((int)hid.size() < H * seq) { MT_LOGE("greedy_generate: short prefill hidden"); return ids; }

    double prefill_seconds = std::chrono::duration<double>(Clock::now()-prefill_start).count();
    auto first_logits_start = Clock::now();
    // Logits from the last prefilled position.
    std::vector<float> last(hid.end() - H, hid.end());
    std::vector<float> logits = dec.logits_from_hidden(last);
    if (logits.empty()) { MT_LOGE("greedy_generate: logits failed"); return ids; }

    logits_seconds += std::chrono::duration<double>(Clock::now()-first_logits_start).count();
    const char* penalty_env = std::getenv("MTD_REPETITION_PENALTY");
    float penalty = penalty_env ? static_cast<float>(std::atof(penalty_env)) : 1.0f;
    penalty = std::max(1.0f, std::min(1.5f, penalty));
    const bool guarded = std::getenv("MTD_LOOP_GUARD") != nullptr;
    Tokenizer guard_tokenizer;
    if (guarded && !guard_tokenizer.load(m)) return ids;
    ids.reserve((size_t)max_new);
    for (;;) {
        if (penalty > 1.0f) {
            std::unordered_set<int32_t> recent;
            for (int i = std::max(0, static_cast<int>(ids.size()) - 100); i < static_cast<int>(ids.size()); ++i) recent.insert(ids[i]);
            for (int32_t id : recent) if (id >= 0 && id < static_cast<int32_t>(logits.size())) {
                logits[id] = logits[id] > 0 ? logits[id] / penalty : logits[id] * penalty;
            }
        }
        int t = argmax_first(logits);
        ids.push_back(t);
        if (t == eos) { MT_LOGI("BENCH_GENERATION tokens=%zu stop=eos", ids.size()); break; }
        if ((int)ids.size() >= max_new) { MT_LOGI("BENCH_GENERATION tokens=%zu stop=token_limit", ids.size()); break; }
        if (guarded && ids.size() >= 128 && ids.size() % 32 == 0) {
            auto begin = ids.begin() + std::max(0, static_cast<int>(ids.size()) - 512);
            std::vector<int32_t> tail(begin, ids.end());
            if (empty_marker_loop(guard_tokenizer.decode(tail))) {
                MT_LOGI("BENCH_GENERATION tokens=%zu stop=empty_marker_loop", ids.size());
                break;
            }
        }

        auto phase_start = Clock::now();
        std::vector<float> emb = embed_token(m, t, H);
        embed_seconds += std::chrono::duration<double>(Clock::now()-phase_start).count();
        phase_start = Clock::now();
        if (emb.empty()) { MT_LOGE("greedy_generate: embed failed @%d", t); break; }
        std::vector<float> h1 = dec.decode_one(emb);
        if ((int)h1.size() < H) { MT_LOGE("greedy_generate: decode_one failed"); break; }
        decoder_seconds += std::chrono::duration<double>(Clock::now()-phase_start).count();
        phase_start = Clock::now();
        logits = dec.logits_from_hidden(h1);
        logits_seconds += std::chrono::duration<double>(Clock::now()-phase_start).count();
        if (logits.empty()) { MT_LOGE("greedy_generate: logits failed"); break; }
    }
    MT_LOGI("CPU_PROFILE prefill=%.6f embedding=%.6f decoder=%.6f logits=%.6f",
        prefill_seconds, embed_seconds, decoder_seconds, logits_seconds);
    // Explicit diagnostic only: token IDs can reconstruct private transcripts.
    // Keep these logs private; publish aggregate replay statistics and hashes.
    const char* trace = std::getenv("MTD_TRACE_TOKENS");
    if (trace && std::strcmp(trace, "1") == 0) {
        // The callback logger has a 2048-byte buffer. Even 128 ten-digit IDs
        // plus metadata fit, so long generations cannot silently lose a tail.
        constexpr size_t chunk = 128;
        for (size_t offset = 0; offset < ids.size(); offset += chunk) {
            std::string values = "[";
            for (size_t i = offset; i < std::min(ids.size(), offset+chunk); ++i) {
                if (i != offset) values += ',';
                values += std::to_string(ids[i]);
            }
            values += ']';
            MT_LOGI("BENCH_TOKEN_TRACE eos=%d total=%zu offset=%zu ids=%s",
                eos, ids.size(), offset, values.c_str());
        }
    }
    return ids;
}

}  // namespace mt
