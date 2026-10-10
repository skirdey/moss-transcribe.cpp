#include "cpu_profile.hpp"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace mt {
namespace {
struct Counters {
    unsigned long long calls = 0, graphs = 0;
    unsigned long long shared_q8 = 0, shared_q8_cast = 0, shared_q8_consumers = 0;
    unsigned long long fused_q8 = 0, fused_q8_consumers = 0;
    double total = 0, build = 0, allocate = 0, input = 0, compute = 0;
};
thread_local std::array<Counters, static_cast<size_t>(CpuPhase::Count)> counters;
thread_local CpuPhase current = CpuPhase::Other;
constexpr const char* names[] = {"other", "mel", "whisper", "adaptor", "prefill", "decode", "logits"};
}
bool cpu_profile_enabled() {
    static const bool enabled = [] { const char* p = std::getenv("MTD_PROFILE"); return p && std::strcmp(p, "1") == 0; }();
    return enabled;
}
CpuTimer::CpuTimer() : enabled_(cpu_profile_enabled()) {
    if (enabled_) start_ = std::chrono::steady_clock::now();
}
double CpuTimer::seconds() const {
    return enabled_ ? std::chrono::duration<double>(std::chrono::steady_clock::now()-start_).count() : 0;
}
CpuPhaseScope::CpuPhaseScope(CpuPhase phase) : phase_(phase), previous_(current) {
    if (cpu_profile_enabled()) current = phase;
}
CpuPhaseScope::~CpuPhaseScope() {
    if (!cpu_profile_enabled()) return;
    auto& c = counters[static_cast<size_t>(phase_)];
    ++c.calls; c.total += timer_.seconds(); current = previous_;
}
void cpu_profile_reset() { counters = {}; current = CpuPhase::Other; }
void cpu_profile_record(CpuStage stage, double seconds) {
    if (!cpu_profile_enabled()) return;
    auto& c = counters[static_cast<size_t>(current)];
    switch (stage) {
        case CpuStage::Build: c.build += seconds; break;
        case CpuStage::Allocate: c.allocate += seconds; ++c.graphs; break;
        case CpuStage::Input: c.input += seconds; break;
        case CpuStage::Compute: c.compute += seconds; break;
    }
}
void cpu_profile_record_shared_q8(bool cast, unsigned long long consumers) {
    if (!cpu_profile_enabled()) return;
    auto& c=counters[static_cast<size_t>(current)];
    ++c.shared_q8; c.shared_q8_cast+=cast; c.shared_q8_consumers+=consumers;
}
void cpu_profile_print() {
    if (!cpu_profile_enabled()) return;
    std::fputs("CPU_PHASE_PROFILE {", stderr);
    for (size_t i = 0; i < counters.size(); ++i) {
        const auto& c = counters[i];
        std::fprintf(stderr, "%s\"%s\":{\"calls\":%llu,\"graphs\":%llu,\"totalSeconds\":%.9f,\"buildSeconds\":%.9f,\"allocateSeconds\":%.9f,\"inputSeconds\":%.9f,\"computeSeconds\":%.9f,\"sharedQ8Nodes\":%llu,\"sharedQ8CastNodes\":%llu,\"sharedQ8Consumers\":%llu,\"fusedQ8Nodes\":%llu,\"fusedQ8Consumers\":%llu}",
            i ? "," : "", names[i], c.calls, c.graphs, c.total, c.build, c.allocate, c.input, c.compute,
            c.shared_q8,c.shared_q8_cast,c.shared_q8_consumers,c.fused_q8,c.fused_q8_consumers);
    }
    std::fputs("}\n", stderr);
}
void cpu_profile_record_fused_q8(unsigned long long consumers) {
    if (!cpu_profile_enabled()) return;
    auto& c=counters[static_cast<size_t>(current)];
    ++c.fused_q8;c.fused_q8_consumers+=consumers;
}
} // namespace mt
