// Log-mel front-end for Whisper (on-device, native). Same spec as transformers'
// WhisperFeatureExtractor: pad/trim 30 s -> periodic Hann(400) -> STFT hop 160, center reflect
// -> power spectrum -> drop last frame -> mel filterbank (201x80) -> log10 -> clamp(max-8)
// -> (x + 4) / 4.  Output layout: mel-major [m * 3000 + t] (80 x 3000).
#pragma once

namespace vh {

constexpr int kSampleRate = 16000;
constexpr int kChunkSamples = 480000;
constexpr int kNfft = 400;
constexpr int kHop = 160;
constexpr int kNMels = 80;
constexpr int kNFreqs = 201;
constexpr int kNFrames = 3000;

// audio: mono float32 (shorter input is zero-padded to 30 s)
// filters: kNFreqs * kNMels float32, row-major (from assets/mel_filters.f32)
// out: kNMels * kNFrames float32
void computeLogMel(const float* audio, int num_samples, const float* filters, float* out);

}  // namespace vh
