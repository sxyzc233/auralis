#include "mel_frontend.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vh {
namespace {
constexpr double kPi = 3.14159265358979323846;
}

void computeLogMel(const float* audio, int num_samples, const float* filters, float* out) {
    // 1) pad / trim to exactly 30 s
    // Non-finite samples are replaced by silence: the capture path can hand over garbage
    // (released audio-capturer buffers produced values like 1e33 / FLT_MAX / NaN), and a single
    // NaN poisons the whole log-mel, which then makes MindSpore Lite's predict fail with -1.
    std::vector<double> a(kChunkSamples, 0.0);
    const int n = std::min(num_samples, kChunkSamples);
    for (int i = 0; i < n; ++i) {
        const float s = audio[i];
        a[i] = std::isfinite(s) ? (double)s : 0.0;
    }

    // 2) reflect pad by n_fft/2 (STFT center=True)
    const int pad = kNfft / 2;
    const int len = kChunkSamples + 2 * pad;
    std::vector<double> x(len);
    for (int i = 0; i < len; ++i) {
        int j = i - pad;
        if (j < 0) j = -j;
        if (j >= kChunkSamples) j = 2 * kChunkSamples - 2 - j;
        x[i] = a[j];
    }

    // periodic Hann window (torch.hann_window default)
    std::vector<double> win(kNfft);
    for (int i = 0; i < kNfft; ++i) win[i] = 0.5 - 0.5 * std::cos(2.0 * kPi * i / kNfft);

    // precomputed twiddle table (naive DFT: correct and portable; an FFT can replace it later)
    std::vector<double> cosT((size_t)kNFreqs * kNfft), sinT((size_t)kNFreqs * kNfft);
    for (int k = 0; k < kNFreqs; ++k)
        for (int i = 0; i < kNfft; ++i) {
            const double ang = -2.0 * kPi * k * i / kNfft;
            cosT[(size_t)k * kNfft + i] = std::cos(ang);
            sinT[(size_t)k * kNfft + i] = std::sin(ang);
        }

    std::vector<double> power(kNFreqs);
    for (int t = 0; t < kNFrames; ++t) {
        const int off = t * kHop;
        for (int k = 0; k < kNFreqs; ++k) {
            double re = 0.0, im = 0.0;
            const double* c = &cosT[(size_t)k * kNfft];
            const double* s = &sinT[(size_t)k * kNfft];
            for (int i = 0; i < kNfft; ++i) {
                const double v = x[off + i] * win[i];
                re += v * c[i];
                im += v * s[i];
            }
            power[k] = re * re + im * im;
        }
        for (int m = 0; m < kNMels; ++m) {
            double acc = 0.0;
            for (int k = 0; k < kNFreqs; ++k) acc += (double)filters[(size_t)k * kNMels + m] * power[k];
            out[(size_t)m * kNFrames + t] = (float)acc;
        }
    }

    // 3) log10 -> clamp to (max - 8) -> (x + 4) / 4
    double mx = -1e30;
    const size_t total = (size_t)kNMels * kNFrames;
    for (size_t i = 0; i < total; ++i) {
        const double l = std::log10(out[i] > 1e-10f ? (double)out[i] : 1e-10);
        out[i] = (float)l;
        if (l > mx) mx = l;
    }
    for (size_t i = 0; i < total; ++i) {
        double l = out[i];
        if (l < mx - 8.0) l = mx - 8.0;
        out[i] = (float)((l + 4.0) / 4.0);
    }
}

}  // namespace vh
