// MIT. Default-off, strict greedy verification; hooks are native lab controls.
#pragma once
#include "generate.hpp"
#include <functional>

namespace mt {
std::vector<int32_t> cpu_ngram_draft(const std::vector<int32_t>& committed, int maximum);
int cpu_greedy_choice(const std::vector<float>& raw, const std::vector<int32_t>& history, float penalty);
struct CpuSpeculativeHooks {
    // Perfect/forced rejection proposals exist only in the standalone gate.
    std::function<std::vector<int32_t>(const std::vector<int32_t>&,int)> draft;
    std::function<void(const std::vector<float>&)> prefilled;
    std::function<void(const std::vector<int32_t>&,const std::vector<float>&)> appended;
    std::function<void(Qwen3Decoder&,const std::vector<int32_t>&,const std::vector<float>&,bool)> committed;
};
std::vector<int32_t> cpu_speculative_generate(Qwen3Decoder& dec,ModelLoader& m,
    const std::vector<float>& fused,int seq,int max_new,int eos,const CpuSpeculativeHooks* hooks=nullptr);
}
