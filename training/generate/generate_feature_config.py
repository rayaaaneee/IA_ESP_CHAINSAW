from train import FeatureConfig

from .globals import FIRMWARE_DIR

FEATURE_CONFIG_HEADER_PATH = FIRMWARE_DIR / "include" / "app" / "config.h"

def main():
    with open(FEATURE_CONFIG_HEADER_PATH, "w") as f:
        f.write(
f"""#ifndef CONFIG_H
#define CONFIG_H
#include <cstdint>

// Configuration structure for audio feature extraction parameters
struct FeatureConfig {{
    int32_t sample_rate;
    float window_seconds;
    float hop_seconds;
    int32_t n_mfcc;
    int32_t n_mels;
    int32_t fft_length;
    int32_t hop_length;
}};

// Global configuration variable
inline constexpr FeatureConfig AUDIO_CONFIG = {{
    .sample_rate = {FeatureConfig.sample_rate},
    .window_seconds = {FeatureConfig.window_seconds}f,
    .hop_seconds = {FeatureConfig.hop_seconds}f,
    .n_mfcc = {FeatureConfig.n_mfcc},
    .n_mels = {FeatureConfig.n_mels},
    .fft_length = {FeatureConfig.fft_length},
    .hop_length = {FeatureConfig.hop_length},
}};

#endif // CONFIG_H
"""
    );

    print(f"Generated FeatureConfig constants in { FEATURE_CONFIG_HEADER_PATH }")
 
if __name__ == "__main__":
    main()