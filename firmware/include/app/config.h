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
extern const FeatureConfig AUDIO_CONFIG;

#endif // CONFIG_H