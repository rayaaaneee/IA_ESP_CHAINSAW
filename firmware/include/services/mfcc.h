#ifndef MFCC_H
#define MFCC_H

#include <cstddef>
#include <cstdint>

#include <Arduino.h>
#include "services/mfcc_config.h"

// Initializes the feature extractor, including building delta coefficients.
void init_feature_extractor();

// Extracts audio features from the provided audio buffer and fills the feature vector.
bool extract_features_from_audio(const int16_t* audio_buffer, float* feature_vector, size_t feature_vector_size);

// Checks if a chainsaw is detected based on the given probability and threshold.
bool is_chainsaw_detected(float probability, float threshold = DETECTION_THRESHOLD);

#endif // MFCC_H