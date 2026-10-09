#include "mel.hpp"
#include "backend.hpp"
#include "common.hpp"
#include "cpu_profile.hpp"
#include "ggml-cpu.h"
// Frames are distributed by ggml. Avoid a nested FFT worker pool.
#define POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft/pocketfft_hdronly.h"
#include "ggml-backend.h"
#include <cmath>
#include <algorithm>
#include <complex>
#include <cstdlib>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mt {

WhisperMel::WhisperMel(const ModelLoader& m) {
    const auto& c = m.config();
    n_fft_ = c.feat_n_fft;
    hop_   = c.feat_hop;
    n_mels_ = c.feat_size;
    n_bins_ = n_fft_ / 2 + 1;
    n_samples_ = c.feat_n_samples;
    nb_max_frames_ = c.feat_nb_max_frames;

    // Periodic Hann window (divide by n_fft, not n_fft-1).
    window_.resize(n_fft_);
    for (int n = 0; n < n_fft_; ++n)
        window_[n] = 0.5f * (1.0f - std::cos(2.0 * M_PI * n / (double)n_fft_));

    // Real-DFT twiddle tables. Whisper's n_fft (400) is not a power of two, so
    // the legacy radix-2 transform cannot be used. Retain the direct DFT
    // as the independent reference for the opt-in parallel/FFT paths.
    // X[k] = sum_n x[n] * exp(-2*pi*i * k*n / N)
    cos_.resize((size_t)n_bins_ * n_fft_);
    sin_.resize((size_t)n_bins_ * n_fft_);
    for (int k = 0; k < n_bins_; ++k) {
        for (int n = 0; n < n_fft_; ++n) {
            double ang = 2.0 * M_PI * (double)k * (double)n / (double)n_fft_;
            cos_[(size_t)k * n_fft_ + n] = (float)std::cos(ang);
            sin_[(size_t)k * n_fft_ + n] = (float)std::sin(ang);
        }
    }

    // mel_filters: conceptually (n_mels, n_bins) row-major. Read the raw f32
    // bytes off the backend buffer and reorder into fb_[mel*n_bins + bin],
    // handling either ggml `ne` orientation.
    fb_.assign((size_t)n_mels_ * n_bins_, 0.0f);
    ggml_tensor* t = m.tensor("mel_filters");
    if (t) {
        std::vector<float> raw((size_t)ggml_nelements(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size() * sizeof(float));
        const int64_t ne0 = t->ne[0];
        const int64_t ne1 = t->ne[1];
        if (ne0 == n_bins_ && ne1 == n_mels_) {
            // ne[0]=bins (fastest), ne[1]=mels -> flat[mel*n_bins + bin]
            for (int mel = 0; mel < n_mels_; ++mel)
                for (int bin = 0; bin < n_bins_; ++bin)
                    fb_[(size_t)mel * n_bins_ + bin] = raw[(size_t)mel * n_bins_ + bin];
        } else if (ne0 == n_mels_ && ne1 == n_bins_) {
            // ne[0]=mels (fastest), ne[1]=bins -> flat[bin*n_mels + mel]
            for (int mel = 0; mel < n_mels_; ++mel)
                for (int bin = 0; bin < n_bins_; ++bin)
                    fb_[(size_t)mel * n_bins_ + bin] = raw[(size_t)bin * n_mels_ + mel];
        }
    }
}

struct WhisperMel::FrameJob {
    const WhisperMel* mel;
    int frames;
    bool fft;
};

void WhisperMel::compute_frames(ggml_tensor* dst, const ggml_tensor*,
                                const ggml_tensor* samples, int ith, int nth, void* userdata) {
    const auto& job = *static_cast<const FrameJob*>(userdata);
    const auto& m = *job.mel;
    const auto* x = static_cast<const float*>(samples->data);
    auto* out = static_cast<float*>(dst->data);
    std::vector<float> frame(m.n_fft_);
    std::vector<double> power(m.n_bins_);
    std::vector<double> real(job.fft ? m.n_fft_ : 0);
    std::vector<std::complex<double>> spectrum(job.fft ? m.n_bins_ : 0);
    const int first = static_cast<int>((int64_t)job.frames * ith / nth);
    const int last = static_cast<int>((int64_t)job.frames * (ith+1) / nth);
    for (int t = first; t < last; ++t) {
        const int start = t * m.hop_;
        for (int n = 0; n < m.n_fft_; ++n) frame[n] = x[(size_t)start + n] * m.window_[n];
        if (job.fft) {
            // Preserve the F32-rounded windowed samples before the transform.
            std::copy(frame.begin(), frame.end(), real.begin());
            pocketfft::r2c<double>({static_cast<size_t>(m.n_fft_)},
                {static_cast<ptrdiff_t>(sizeof(double))},
                {static_cast<ptrdiff_t>(sizeof(std::complex<double>))},
                0, true, real.data(), spectrum.data(), 1.0, 1);
            for (int b = 0; b < m.n_bins_; ++b)
                power[b] = spectrum[b].real()*spectrum[b].real() + spectrum[b].imag()*spectrum[b].imag();
        } else {
            // Same coefficients and sequential reduction as the legacy path.
            for (int b = 0; b < m.n_bins_; ++b) {
                const float* cr = &m.cos_[(size_t)b * m.n_fft_];
                const float* sr = &m.sin_[(size_t)b * m.n_fft_];
                double re = 0.0, im = 0.0;
                for (int n = 0; n < m.n_fft_; ++n) {
                    re += (double)frame[n] * cr[n];
                    im -= (double)frame[n] * sr[n];
                }
                power[b] = re * re + im * im;
            }
        }
        for (int mel = 0; mel < m.n_mels_; ++mel) {
            const float* fbrow = &m.fb_[(size_t)mel * m.n_bins_];
            double acc = 0.0;
            for (int b = 0; b < m.n_bins_; ++b) acc += (double)fbrow[b] * power[b];
            out[(size_t)mel * job.frames + t] = (float)std::log10(std::max(acc, 1e-10));
        }
    }
}

void WhisperMel::compute(const std::vector<float>& samples, std::vector<float>& out,
                         int& n_mels, int& n_frames) const {
    CpuPhaseScope phase(CpuPhase::Mel);
    n_mels = n_mels_;
    n_frames = nb_max_frames_;

    // Center STFT: reflect-pad by n_fft/2 on both ends (numpy 'reflect', which
    // excludes the edge sample). This makes frame t = t*hop the centered frame
    // and yields exactly nb_max_frames frames (dropping the trailing frame).
    const int pad = n_fft_ / 2;
    const int64_t N = (int64_t)samples.size();
    std::vector<float> x((size_t)N + 2 * pad);
    for (int i = 0; i < pad; ++i) x[i] = samples[pad - i];
    for (int64_t i = 0; i < N; ++i) x[(size_t)pad + i] = samples[(size_t)i];
    for (int i = 0; i < pad; ++i)
        x[(size_t)pad + N + i] = samples[(size_t)(N - 2 - i)];

    out.assign((size_t)n_mels_ * n_frames, 0.0f);
    const char* env = std::getenv("MTD_CPU_OPT");
    const int opt = env ? std::atoi(env) : 0;
    bool parallel_done = false;
    if ((opt & (2048 | 4096)) && ggml_backend_is_cpu(mt::backend())) {
        CpuTimer build_timer;
        // The first custom-op input supplies the output shape only. The second
        // owns reflected samples; workers write disjoint frame columns.
        auto* ctx = ggml_init({ggml_tensor_overhead()*8 + ggml_graph_overhead_custom(8, false), nullptr, true});
        if (ctx) {
            auto* shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_frames, n_mels_);
            auto* input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, x.size());
            ggml_set_input(input);
            FrameJob job{this, n_frames, (opt & 4096) != 0};
            auto* result = ggml_map_custom2(ctx, shape, input, compute_frames, GGML_N_TASKS_MAX, &job);
            ggml_set_output(result);
            auto* graph = ggml_new_graph_custom(ctx, 8, false);
            ggml_build_forward_expand(graph, result);
            cpu_profile_record(CpuStage::Build, build_timer.seconds());
            parallel_done = compute_graph_with_inputs(graph, [&] {
                ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
            });
            if (parallel_done) ggml_backend_tensor_get(result, out.data(), 0, out.size()*sizeof(float));
            ggml_free(ctx);
        }
        if (!parallel_done) MT_LOGW("parallel mel graph failed; using serial DFT");
    }
    if (!parallel_done) {
        std::vector<float> frame(n_fft_);
        std::vector<double> power(n_bins_);

        for (int t = 0; t < n_frames; ++t) {
            const int start = t * hop_;
            for (int n = 0; n < n_fft_; ++n) frame[n] = x[(size_t)start + n] * window_[n];

            // Power spectrum |rfft(frame)|^2 over n_bins bins (computed once/frame).
            for (int b = 0; b < n_bins_; ++b) {
                const float* cr = &cos_[(size_t)b * n_fft_];
                const float* sr = &sin_[(size_t)b * n_fft_];
                double re = 0.0, im = 0.0;
                for (int n = 0; n < n_fft_; ++n) {
                    re += (double)frame[n] * cr[n];
                    im -= (double)frame[n] * sr[n];
                }
                power[b] = re * re + im * im;
            }

            // Mel projection: fb (n_mels x n_bins) @ power, then log10.
            for (int mel = 0; mel < n_mels_; ++mel) {
                const float* fbrow = &fb_[(size_t)mel * n_bins_];
                double acc = 0.0;
                for (int b = 0; b < n_bins_; ++b) acc += (double)fbrow[b] * power[b];
                out[(size_t)mel * n_frames + t] =
                    (float)std::log10(std::max(acc, 1e-10));
            }
        }

    } // serial DFT fallback

    // Per-chunk normalization over the full (n_mels, n_frames):
    // mx = max(all); floor = mx - 8; v = max(v, floor); v = (v + 4) / 4.
    float mx = out[0];
    for (float v : out) mx = std::max(mx, v);
    const float floor = mx - 8.0f;
    for (float& v : out) {
        v = std::max(v, floor);
        v = (v + 4.0f) / 4.0f;
    }
}

} // namespace mt
