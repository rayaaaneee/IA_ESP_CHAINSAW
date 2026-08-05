#include "app/config.h"

// Global configuration variable with specific values
extern const constexpr FeatureConfig AUDIO_CONFIG = {
    .sample_rate = 8000,
    .window_seconds = 2.0f,
    .hop_seconds = 1.0f,
    .n_mfcc = 20,
    .n_mels = 32,
    .fft_length = 1024,
    .hop_length = 256
};