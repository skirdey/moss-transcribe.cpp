// MIT. Explicit research route for measured, dense Whisper encoder shapes.
#pragma once
#include "ggml.h"
#include <memory>
#include <vector>

namespace mt {
// Every encode() owns one instance until graph execution and reads complete.
// A context belongs to one custom node; graph execution remains serialized.
class CpuEncoderQ8 {
public:
    CpuEncoderQ8();
    ~CpuEncoderQ8();
    CpuEncoderQ8(const CpuEncoderQ8&) = delete;
    CpuEncoderQ8& operator=(const CpuEncoderQ8&) = delete;
    // Returns null unless CPU, 16 workers, F32 contiguous input and one of the
    // three measured Q8_0 shapes at M=1500. The caller supplies eager fallback.
    ggml_tensor* mul_mat(ggml_context* ctx, ggml_tensor* weight, ggml_tensor* input);
    // One shared conversion/panel pack for three compatible 1024-wide Q/K/V
    // projections. Output [N,M,3] contains three contiguous, independent slices.
    ggml_tensor* qkv(ggml_context* ctx, ggml_tensor* input, ggml_tensor* const* weights, size_t count);
    bool ok() const;
    void record_profile() const; // after worker join, on the calling thread
    size_t nodes() const { return contexts_.size(); }
    unsigned long long executions() const;
    unsigned long long consumers() const;
    unsigned long long consumer_executions() const;
    unsigned long long qkv_nodes() const;
    unsigned long long qkv_executions() const;
    int min_workers() const;
    int max_workers() const;
    static bool supported();
private:
    struct Context;
    std::vector<std::unique_ptr<Context>> contexts_;
};
}
