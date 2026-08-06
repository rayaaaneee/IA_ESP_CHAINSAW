#ifndef MFCC_CONFIG_H
#define MFCC_CONFIG_H

#include <Arduino.h>

#include "app/config.h"

constexpr size_t AUDIO_WINDOW_SAMPLES = 16000;
constexpr size_t FEATURE_FRAME_COUNT = 63;
constexpr size_t FEATURE_VECTOR_SIZE = 194;
constexpr float DETECTION_THRESHOLD = 0.5f;

#endif // MFCC_CONFIG_H