// Model-independent numerical and timing probe. Serial production DFT is the
// independent reference; compare full normalized outputs, including boundaries.
#include "mel.hpp"
#include "backend.hpp"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <vector>

static void set_opt(int opt) {
    const auto value = std::to_string(opt);
#ifdef _WIN32
    _putenv_s("MTD_CPU_OPT", value.c_str());
#else
    setenv("MTD_CPU_OPT", value.c_str(), 1);
#endif
}
int main(int argc, char** argv) {
    auto* backend = mt::backend();
    if (!ggml_backend_is_cpu(backend)) return 77;
    constexpr int bins = 201, mels = 80, frames = 3000, samples = 480000;
    auto path = std::filesystem::temp_directory_path() / ("moss-mel-cpu-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); } } cleanup{path};
    auto* ctx = ggml_init({ggml_tensor_overhead()*4 + bins*mels*sizeof(float) + 1024, nullptr, false});
    auto* filter = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, bins, mels);
    ggml_set_name(filter, "mel_filters");
    auto* data = static_cast<float*>(filter->data);
    // Triangular bands (synthetic, independent of private model weights).
    for (int mel = 0; mel < mels; ++mel) for (int b = 0; b < bins; ++b)
        data[mel*bins+b] = std::max(0.0f, 1.0f - std::abs(b - (mel+1)*2.4f)/3.0f);
    auto* gguf = gguf_init_empty();
    gguf_set_val_u32(gguf, "mtd.feat.n_fft", 400);
    gguf_set_val_u32(gguf, "mtd.feat.hop", 160);
    gguf_set_val_u32(gguf, "mtd.feat.feature_size", mels);
    gguf_set_val_u32(gguf, "mtd.feat.n_samples", samples);
    gguf_set_val_u32(gguf, "mtd.feat.nb_max_frames", frames);
    gguf_add_tensor(gguf, filter);
    if (!gguf_write_to_file(gguf, path.string().c_str(), false)) return 2;
    gguf_free(gguf); ggml_free(ctx);
    mt::ModelLoader model;
    if (!model.load(argc > 1 ? argv[1] : path.string())) return 2;
    mt::WhisperMel mel(model);
    bool exact = true, finite = true;
    for (const char* kind : {"zero", "noise", "impulse", "tone", "mixed", "extreme"}) {
        std::vector<float> input(samples, 0);
        std::mt19937 rng(7301); std::uniform_real_distribution<float> noise(-1, 1);
        for (int n = 0; n < samples; ++n) {
            if (!std::strcmp(kind, "noise")) input[n] = noise(rng);
            if (!std::strcmp(kind, "tone")) input[n] = std::sin(n * 0.17278759594743862);
            if (!std::strcmp(kind, "mixed")) input[n] = noise(rng)*0.1f + std::sin(n*0.06123)*0.9f;
            if (!std::strcmp(kind, "extreme")) input[n] = (n & 1 ? -1 : 1)*1e12f;
        }
        if (!std::strcmp(kind, "impulse")) for (int n : {0, 199, 200, 10000, samples-1}) input[n] = 1;
        int M = 0, T = 0; std::vector<float> ref;
        set_opt(0); auto start = std::chrono::steady_clock::now();
        mel.compute(input, ref, M, T);
        const double serial = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if (M != mels || T != frames || ref.size() != mels*frames) return 2;
        for (int threads : {1, 16}) for (int opt : {2048, 4096}) {
            ggml_backend_cpu_set_n_threads(backend, threads);
            std::vector<float> got; set_opt(opt); start = std::chrono::steady_clock::now();
            mel.compute(input, got, M, T, opt == 4096 ? mt::WhisperMel::Transform::ExperimentalFft : mt::WhisperMel::Transform::ParallelDft);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
            if (got.size() != ref.size() || M != mels || T != frames) return 2;
            size_t bits = 0; double max_error = 0, sum_sq = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                if (std::memcmp(&got[i], &ref[i], sizeof(float))) ++bits;
                const double error = double(got[i])-ref[i];
                finite &= std::isfinite(got[i]); max_error = std::max(max_error, std::abs(error)); sum_sq += error*error;
            }
            if (opt == 2048) exact &= bits == 0;
            std::printf("{\"signal\":\"%s\",\"threads\":%d,\"opt\":%d,\"floatBitDifferences\":%zu,\"maxAbsError\":%.12g,\"rmsError\":%.12g,\"serialSeconds\":%.9f,\"candidateSeconds\":%.9f,\"speedup\":%.6f}\n",
                kind, threads, opt, bits, max_error, std::sqrt(sum_sq/ref.size()), serial, elapsed, serial/elapsed);
        }
    }
    // FFT drift is reported, never presented as bit parity. Its acceptance
    // requires independent full-model transcript/timestamp/speaker/EOS gates.
    return exact && finite ? 0 : 1;
}
