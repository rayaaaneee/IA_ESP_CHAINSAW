#ifndef WATCHDOG_H
#define WATCHDOG_H

// Initializes and subscribes the calling task (the Arduino loop task) to the ESP32 Task Watchdog Timer.
// If loop() stalls (blocked I2S read, TFLite hang, ...) for longer than kWatchdogTimeoutSeconds, the device reboots.
void init_watchdog();

// Resets the watchdog countdown. Must be called at least once per kWatchdogTimeoutSeconds from within loop().
void feed_watchdog();

#endif // WATCHDOG_H
