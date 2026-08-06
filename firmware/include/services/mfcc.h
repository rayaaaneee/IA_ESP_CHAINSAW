#ifndef MFCC_H
#define MFCC_H

#include <Arduino.h>
#include <cstddef>
#include <cstdint>

#include "app/config.h"

constexpr size_t AUDIO_WINDOW_SAMPLES = 16000;
constexpr size_t FEATURE_FRAME_COUNT = 63;
constexpr size_t FEATURE_VECTOR_SIZE = 194;
constexpr float DETECTION_THRESHOLD = 0.5f;

// Initializes the feature extractor, including building delta coefficients.
void init_feature_extractor();

// Extracts audio features from the provided audio buffer and fills the feature vector.
bool extract_features_from_audio(const int16_t* audio_buffer, float* feature_vector, size_t feature_vector_size);

// Checks if a chainsaw is detected based on the given probability and threshold.
bool is_chainsaw_detected(float probability, float threshold = DETECTION_THRESHOLD);

#endif // MFCC_H