#ifndef FLAGS_H
#define FLAGS_H

// Hand-maintained compile-time feature flags: set to 1/0 and rebuild to enable/disable a module.

// Set to 1 to enable verbose Serial debug output (feature vector preview, tensor dumps). Keep at 0 in production to save flash/CPU.
#define ENABLE_DEBUG_LOGS 0

// Set to 1 to drive the status LED (drivers/gpio). Set to 0 to make init_status_led()/set_status_led() a no-op.
#define ENABLE_STATUS_LED 0

// Set to 1 to initialize/use the LoRa radio (services/communication). Kept at 0 until the module is wired and its pins confirmed:
// while disabled, init_communication()/send_alert() are a no-op and never touch the radio hardware.
#define ENABLE_LORA_COMMUNICATION 0

#endif // FLAGS_H
