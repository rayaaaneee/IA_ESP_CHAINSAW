#ifndef CONFIG_H
#define CONFIG_H
#include <cstdint>

// Configuration structure for audio feature extraction parameters
struct FeatureConfig {
    int32_t sample_rate;
    float window_seconds;
    float hop_seconds;
    int32_t n_mfcc;
    int32_t n_mels;
    int32_t fft_length;
    int32_t hop_length;
};

// Global configuration variable
inline constexpr FeatureConfig AUDIO_CONFIG = {
    .sample_rate = 8000,
    .window_seconds = 2.0f,
    .hop_seconds = 1.0f,
    .n_mfcc = 20,
    .n_mels = 32,
    .fft_length = 1024,
    .hop_length = 256,
};

#endif // CONFIG_H
