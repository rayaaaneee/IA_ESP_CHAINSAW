#include <Arduino.h>
#include <RadioLib.h>
#include <arduinoFFT.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include "model/inference.h"

// PINS Definitions
#ifndef PINS
#define PIN_EXAMPLE 1
#endif

// --- Variables Globales IA ---
const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// La mémoire RAM allouée pour faire tourner le réseau (50 Ko ici, à ajuster si besoin)
constexpr int kTensorArenaSize = 50 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

void setup() {
  Serial.begin(115200);
  delay(2000); 

  Serial.println("Initialisation de l'IA...");

  tflite_model = tflite::GetModel(g_model_data);

  static tflite::AllOpsResolver resolver;

  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  static tflite::MicroInterpreter static_interpreter(
      tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Erreur: Impossible d'allouer la Tensor Arena !");
    while (1); 
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Modèle IA chargé et prêt à l'emploi !");
}

void loop() {
  // 1. Remplir le tenseur d'entrée (input) avec tes caractéristiques audio (MFCC)
  /* 
  Exemple de remplissage lorsque tu auras ton extraction audio :
  for (int i = 0; i < taille_entree; i++) {
    input->data.f[i] = tableau_mfcc[i]; 
  }
  */

  // 2. Exécuter l'inférence
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Erreur: Échec de l'inférence !");
    return;
  }

  // 3. Lire le résultat de la prédiction (neurone de sortie unique, activation Sigmoïde)
  float prob_tronconneuse = output->data.f[0];

  Serial.print("Probabilité Tronçonneuse : ");
  Serial.println(prob_tronconneuse);

  delay(1000);
}