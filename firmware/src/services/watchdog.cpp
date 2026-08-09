#include "services/watchdog.h"

#include <Arduino.h>
#include <esp_task_wdt.h>

#include "app/board_config.h"

void init_watchdog() {
    esp_task_wdt_init(kWatchdogTimeoutSeconds, true);
    esp_task_wdt_add(NULL);
}

void feed_watchdog() {
    esp_task_wdt_reset();
}
