#ifndef COMMUNICATION_H
#define COMMUNICATION_H

// Initializes the LoRa radio (see LoRa pins/frequency in app/board_config.h).
// No-op (returns false) when ENABLE_LORA_COMMUNICATION is 0.
// Returns false if the radio could not be initialized (e.g. wrong wiring); alerts are then silently skipped.
bool init_communication();

// Sends a short chainsaw-detection alert over LoRa. No-op if ENABLE_LORA_COMMUNICATION is 0, or init_communication() failed/was not called.
void send_alert(float probability);

#endif // COMMUNICATION_H
