#ifndef AUDIO_H
#define AUDIO_H
#include <cstdint>
#include "app/config.h" // For the sample rate configuration
#include <driver/i2s.h>
#include <Arduino.h>

void init_audio();

bool record_audio(int16_t* buffer, int size);

#endif // AUDIO_H