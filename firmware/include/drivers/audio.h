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

/* Critical hardware detail:
On the INMP441 module, there is an L/R (Left/Right) pin. For the
I2S_CHANNEL_FMT_ONLY_LEFT configuration in this code to work, you must
connect the microphone's L/R pin to ground (GND). If you leave it floating
or connect it to 3.3V, the data will arrive on the wrong channel and your
buffer will be filled with zeros. */