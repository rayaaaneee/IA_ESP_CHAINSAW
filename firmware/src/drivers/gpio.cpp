#include "drivers/gpio.h"

#include <Arduino.h>

#include "app/board_config.h"

void init_status_led() {
#if ENABLE_STATUS_LED
    pinMode(kStatusLedPin, OUTPUT);
    digitalWrite(kStatusLedPin, LOW);
#endif
}

void set_status_led(bool on) {
#if ENABLE_STATUS_LED
    digitalWrite(kStatusLedPin, on ? HIGH : LOW);
#endif
}
