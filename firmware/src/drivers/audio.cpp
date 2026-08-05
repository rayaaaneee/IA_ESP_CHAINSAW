#include "drivers/audio.h"

// PINS Definitions
#define I2S_SCK 14  // Horloge (BCLK)
#define I2S_WS  15  // Word Select (LRC / WS)
#define I2S_SD  32  // Serial Data (DIN / SD)
#define I2S_PORT I2S_NUM_0

// Initializes the I2S audio input with the specified configuration.
void init_audio() {
    const i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = AUDIO_CONFIG.sample_rate,
        .bits_per_sample = i2s_bits_per_sample_t(16),
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, 
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
        .intr_alloc_flags = 0, 
        .dma_buf_count = 8,
        .dma_buf_len = 1024,
        .use_apll = false
    };

    if (i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL) != ESP_OK) {
        Serial.println("Erreur fatale: Échec de l'installation du driver I2S");
    }

    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD
    };

    if (i2s_set_pin(I2S_PORT, &pin_config) != ESP_OK) {
        Serial.println("Erreur fatale: Échec de la configuration des pins I2S");
    }
}

// Capture audio in the provided buffer. Returns true if successful, false otherwise.
bool record_audio(int16_t* buffer, int size) {
    size_t bytes_read = 0;
    size_t bytes_to_read = size * sizeof(int16_t);

    esp_err_t result = i2s_read(I2S_PORT, (void*)buffer, bytes_to_read, &bytes_read, portMAX_DELAY);

    return (result == ESP_OK && bytes_read == bytes_to_read);
}

/* Détail matériel critique (Ne te fais pas avoir) :
Sur le module INMP441, tu as une broche L/R (Left/Right). Pour que la configuration I2S_CHANNEL_FMT_ONLY_LEFT de mon code fonctionne, tu dois absolument relier la broche L/R de ton micro à la masse (GND). Si tu la laisses en l'air ou la branches sur le 3.3V, les données arriveront sur le mauvais canal et ton tableau sera rempli de zéros. */