#ifndef MFCC_UTILS_H
#define MFCC_UTILS_H

#include <cmath>

#include <Arduino.h>
#include "services/mfcc_config.h"

namespace mfcc_utils {
    constexpr size_t kAudioWindowSamples = AUDIO_WINDOW_SAMPLES;
    constexpr size_t kFftLength = AUDIO_CONFIG.fft_length;
    constexpr size_t kFftBins = (AUDIO_CONFIG.fft_length / 2) + 1;
    constexpr size_t kMfccCount = AUDIO_CONFIG.n_mfcc;
    constexpr size_t kMelCount = AUDIO_CONFIG.n_mels;
    constexpr size_t kDeltaWindow = 9;
    constexpr size_t kDeltaHalfWindow = kDeltaWindow / 2;
    constexpr int kRmsFrameLength = 2048;
    constexpr int kRmsPad = kRmsFrameLength / 2;
    constexpr float kEpsilon = 1e-10f;
    constexpr float kTopDb = 80.0f;
    constexpr float kRollPercent = 0.85f;

    extern float g_delta_coeffs[kDeltaWindow];
    extern float g_delta2_coeffs[kDeltaWindow];
    extern bool g_initialized;

    float hann_window(size_t index, size_t size);
    int reflect_index(int index, int length);
    float sample_reflect(const float* signal, int length, int index);
    bool invert_2x2(const double in[2][2], double out[2][2]);
    bool invert_3x3(const double in[3][3], double out[3][3]);
    void build_delta_coefficients(int derivative_order, float* coeffs);
    void compute_delta_series(const float* source, int num_coeffs, int num_frames, const float* coeffs, float* destination);
    void ensure_initialized();
}  // namespace mfcc_utils

#endif // MFCC_UTILS_H