"""Apply experimental CPU optimizations to an isolated pinned MOSS checkout.

Never run against the production checkout. MTD_CPU_OPT is a bitmask:
1 = reuse ggml pool metadata, 2 = direct CPU embedding dequantization,
4 = fused AVX SwiGLU, 8 = F32 flash attention during single-token decode,
16 = persistent transposed value cache with reference attention arithmetic.
Zero retains reference operations for profiling.
"""
import argparse
from pathlib import Path


def replace(path, old, new):
    text = path.read_text()
    if text.count(old) != 1:
        raise ValueError(f"Unexpected source in {path}: {old[:60]}")
    path.write_text(text.replace(old, new))


def apply(root):
    backend = root / "src/backend.cpp"
    replace(backend, "ggml_backend_cpu_set_n_threads(g_backend, nt);", '''ggml_backend_cpu_set_n_threads(g_backend, nt);
            // OpenMP already reuses workers; this reuses ggml's pool metadata.
            const char* opt = std::getenv("MTD_CPU_OPT");
            if (opt && (std::atoi(opt) & 1)) {
                auto params = ggml_threadpool_params_default(nt);
                static ggml_threadpool_t pool = ggml_threadpool_new(&params);
                if (pool) ggml_backend_cpu_set_threadpool(g_backend, pool);
            }''')
    generate = root / "src/generate.cpp"
    replace(generate, "    const size_t max_nodes = 8;", '''    // CPU buffers can be read directly. Dequantize only selected rows using
    // ggml's existing type trait; avoid graph allocation/dispatch per token.
    const char* opt = std::getenv("MTD_CPU_OPT");
    if (opt && (std::atoi(opt) & 2) && ggml_backend_is_cpu(backend()))
        return embed_rows_f32_host(tok, ids, n_ids, hidden, out);
    const size_t max_nodes = 8;''')
    replace(generate, '#include "ggml-backend.h"', '#include "ggml-backend.h"\n#include "ggml-cpu.h"\n#include <chrono>')
    replace(generate, "    std::vector<float> hid;", '''    using Clock = std::chrono::steady_clock;
    double embed_seconds = 0, decoder_seconds = 0, logits_seconds = 0;
    auto prefill_start = Clock::now();
    std::vector<float> hid;''')
    replace(generate, "    // Logits from the last prefilled position.", '''    double prefill_seconds = std::chrono::duration<double>(Clock::now()-prefill_start).count();
    auto first_logits_start = Clock::now();
    // Logits from the last prefilled position.''')
    replace(generate, "    const char* penalty_env", '''    logits_seconds += std::chrono::duration<double>(Clock::now()-first_logits_start).count();
    const char* penalty_env''')
    replace(generate, "        std::vector<float> emb = embed_token(m, t, H);", '''        auto phase_start = Clock::now();
        std::vector<float> emb = embed_token(m, t, H);
        embed_seconds += std::chrono::duration<double>(Clock::now()-phase_start).count();
        phase_start = Clock::now();''')
    replace(generate, "        logits = dec.logits_from_hidden(h1);", '''        decoder_seconds += std::chrono::duration<double>(Clock::now()-phase_start).count();
        phase_start = Clock::now();
        logits = dec.logits_from_hidden(h1);
        logits_seconds += std::chrono::duration<double>(Clock::now()-phase_start).count();''')
    replace(generate, "    return ids;\n}\n\n}  // namespace mt", '''    MT_LOGI("CPU_PROFILE prefill=%.6f embedding=%.6f decoder=%.6f logits=%.6f",
        prefill_seconds, embed_seconds, decoder_seconds, logits_seconds);
    return ids;
}

}  // namespace mt''')
    qwen = root / "src/qwen3.cpp"
    replace(qwen, "#include <cmath>", "#include <cmath>\n#include <cstdlib>")
    replace(qwen, "    const int hd     = hp.head_dim;", '''    const char* layout_opt = std::getenv("MTD_CPU_OPT");
    const bool transposed_v = k_cache && layout_opt && (std::atoi(layout_opt) & 16);
    const int hd     = hp.head_dim;''')
    replace(qwen, '''        struct ggml_tensor* v_dst = ggml_view_4d(ctx, v_cache, hd, n_kv_h, n_tokens, 1,
            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], (size_t)past_seq * v_cache->nb[2]);''', '''        struct ggml_tensor* v_dst = transposed_v
            ? ggml_view_4d(ctx, v_cache, n_tokens, hd, n_kv_h, 1,
                v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], (size_t)past_seq * sizeof(float))
            : ggml_view_4d(ctx, v_cache, hd, n_kv_h, n_tokens, 1,
                v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], (size_t)past_seq * v_cache->nb[2]);
        if (transposed_v) v = ggml_permute(ctx, v, 1, 2, 0, 3);''')
    replace(qwen, '''        v_used = ggml_view_4d(ctx, v_cache, hd, n_kv_h, kv, 1,
            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], 0);''', '''        v_used = transposed_v
            ? ggml_view_4d(ctx, v_cache, kv, hd, n_kv_h, 1,
                v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], 0)
            : ggml_view_4d(ctx, v_cache, hd, n_kv_h, kv, 1,
                v_cache->nb[1], v_cache->nb[2], v_cache->nb[3], 0);''')
    replace(qwen, "    struct ggml_tensor* v_p = ggml_permute(ctx, v_used, 0, 2, 1, 3);", '''    struct ggml_tensor* v_p = transposed_v ? ggml_transpose(ctx, v_used)
        : ggml_permute(ctx, v_used, 0, 2, 1, 3);''')
    replace(qwen, "    struct ggml_tensor* scores = ggml_mul_mat(ctx, k_p, q_p);", '''    const char* cpu_opt = std::getenv("MTD_CPU_OPT");
    struct ggml_tensor* o = nullptr;
    if (cpu_opt && (std::atoi(cpu_opt) & 8) && n_tokens == 1 && !mask && backend_supports_flash_attn()) {
        // Fused attention reads strided F32 KV cache directly, avoiding the
        // growing V-cache transpose/copy on every token in every layer.
        o = ggml_flash_attn_ext(ctx, q_p, k_p, v_p, nullptr, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
    } else {
    struct ggml_tensor* scores = ggml_mul_mat(ctx, k_p, q_p);''')
    replace(qwen, "    struct ggml_tensor* o   = ggml_mul_mat(ctx, v_t, attn);", "    o = ggml_mul_mat(ctx, v_t, attn);")
    replace(qwen, "    o = ggml_permute(ctx, o, 0, 2, 1, 3);", "    o = ggml_permute(ctx, o, 0, 2, 1, 3);\n    }")
    replace(qwen, "    struct ggml_tensor* v_t = maybe_cont(ctx, ggml_transpose(ctx, v_p));", '''    struct ggml_tensor* v_t = transposed_v ? v_used
        : maybe_cont(ctx, ggml_transpose(ctx, v_p));''')
    replace(qwen, "    struct ggml_tensor* f  = ggml_mul_mat(ctx, w.ffn_down, ggml_mul(ctx, ggml_silu(ctx, g), u));", '''    const char* opt = std::getenv("MTD_CPU_OPT");
    // One AVX-512/AVX2 pass instead of two passes and two graph barriers.
    // The fused kernel uses the same vector SiLU and multiply primitives.
    struct ggml_tensor* activation = opt && (std::atoi(opt) & 4)
        ? ggml_swiglu_split(ctx, g, u)
        : ggml_mul(ctx, ggml_silu(ctx, g), u);
    struct ggml_tensor* f = ggml_mul_mat(ctx, w.ffn_down, activation);''')
    decoder = root / "src/qwen3_decoder.cpp"
    replace(decoder, '#include "qwen3_decoder.hpp"', '#include "qwen3_decoder.hpp"\n#include <cstdlib>')
    replace(decoder, '''        v_cache_[l] = ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                                         hp_.head_dim, hp_.n_kv_heads, max_seq_, 1);''', '''        const char* opt = std::getenv("MTD_CPU_OPT");
        v_cache_[l] = opt && (std::atoi(opt) & 16)
            ? ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                max_seq_, hp_.head_dim, hp_.n_kv_heads, 1)
            : ggml_new_tensor_4d(kv_ctx_.get(), GGML_TYPE_F32,
                hp_.head_dim, hp_.n_kv_heads, max_seq_, 1);''')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    apply(parser.parse_args().source)
