#include <Arduino.h>

// La plupart des ESP32 utilisent le GPIO 2 pour la LED intégrée
#define LED_BUILTIN 2

void setup() {
  // Initialisation de la broche de la LED en sortie
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_BUILTIN, HIGH);  // Allume la LED
  delay(1000);                      // Attend 1 seconde
  digitalWrite(LED_BUILTIN, LOW);   // Éteint la LED
  delay(1000);                      // Attend 1 seconde
}