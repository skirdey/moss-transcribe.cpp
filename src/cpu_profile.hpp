#pragma once
#include <chrono>

namespace mt {
// Opt-in, process-local instrumentation. Graph execution is serialized by the
// existing backend; custom-op workers never write these counters.
enum class CpuPhase { Other, Mel, Whisper, Adaptor, Prefill, Decode, Logits, Count };
enum class CpuStage { Build, Allocate, Input, Compute };
bool cpu_profile_enabled();
void cpu_profile_reset();
void cpu_profile_print();
void cpu_profile_record(CpuStage stage, double seconds);
void cpu_profile_record_shared_q8(bool cast, unsigned long long consumers);
class CpuTimer {
public:
    CpuTimer();
    double seconds() const;
private:
    bool enabled_;
    std::chrono::steady_clock::time_point start_;
};
class CpuPhaseScope {
public:
    explicit CpuPhaseScope(CpuPhase phase);
    ~CpuPhaseScope();
private:
    CpuPhase phase_, previous_;
    CpuTimer timer_;
};
} // namespace mt
