#ifndef MFCC_H
#define MFCC_H

#include <Arduino.h>
#include <arduinoFFT.h>
#include "app/config.h" // For the sample fft_length configuration

void extract_mfcc(int16_t* audio_buffer, float* mfcc_features);

#endif // MFCC_H