#include "services/communication.h"

#include <Arduino.h>
#include <RadioLib.h>

#include "app/board_config.h"

#if ENABLE_LORA_COMMUNICATION
namespace {
Module radio_module(kLoraCsPin, kLoraDio0Pin, kLoraResetPin);
SX1276 radio(&radio_module);
bool g_radio_ready = false;
}  // namespace
#endif

bool init_communication() {
#if ENABLE_LORA_COMMUNICATION
    const int16_t state = radio.begin(kLoraFrequencyMhz);
    g_radio_ready = (state == RADIOLIB_ERR_NONE);
    if (!g_radio_ready) {
        Serial.printf("Warning: LoRa init failed (code %d), alerts disabled.\n", state);
    }
    return g_radio_ready;
#else
    return false;
#endif
}

void send_alert(float probability) {
#if ENABLE_LORA_COMMUNICATION
    if (!g_radio_ready) {
        return;
    }

    char message[32];
    snprintf(message, sizeof(message), "CHAINSAW,%.3f", probability);

    const int16_t state = radio.transmit(message);
    if (state != RADIOLIB_ERR_NONE) {
        Serial.printf("Warning: LoRa transmit failed (code %d).\n", state);
    }
#else
    (void)probability;
#endif
}
