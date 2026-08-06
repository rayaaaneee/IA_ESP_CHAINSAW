#ifndef AUDIO_H
#define AUDIO_H
#include <cstdint>
#include "app/config.h" // For the sample rate configuration
#include <driver/i2s.h>
#include <Arduino.h>

// PINS Definitions
#define I2S_SCK 14  // Horloge (BCLK)
#define I2S_WS  15  // Word Select (LRC / WS)
#define I2S_SD  32  // Serial Data (DIN / SD)
#define I2S_PORT I2S_NUM_0 // I2S port number

// Initializes the audio system (I2S driver) for audio recording.
// For INMP441, ensure the L/R pin is connected to GND for correct channel selection.
void init_audio();

// Records audio samples into the provided buffer.
bool record_audio(int16_t* buffer, int size);

#endif // AUDIO_H

/* Détail matériel critique (Ne te fais pas avoir) :
Sur le module INMP441, tu as une broche L/R (Left/Right). Pour que la configuration I2S_CHANNEL_FMT_ONLY_LEFT de mon code fonctionne, tu dois absolument relier la broche L/R de ton micro à la masse (GND). Si tu la laisses en l'air ou la branches sur le 3.3V, les données arriveront sur le mauvais canal et ton tableau sera rempli de zéros. */