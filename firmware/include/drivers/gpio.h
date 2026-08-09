#ifndef GPIO_H
#define GPIO_H

// Configures the onboard status LED pin (see kStatusLedPin in app/board_config.h). No-op when ENABLE_STATUS_LED is 0.
void init_status_led();

// Turns the status LED on or off, e.g. to reflect the current chainsaw detection state. No-op when ENABLE_STATUS_LED is 0.
void set_status_led(bool on);

#endif // GPIO_H
