#include <Arduino.h>

#include <RadioLib.h>
#include <arduinoFFT.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "model/inference.h"
#include "app/config.h"
#include "drivers/audio.h"

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
    Serial.println("Error: Unable to allocate tensors!");
    while (1); 
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("IA Model loaded and ready for inference.");

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Error: Unable to allocate tensors!");
    while (1); 
  }

  init_audio();
  
  // Affiche la taille exacte calculée par le planificateur de mémoire
  Serial.printf("Taille strictement necessaire pour l'arene : %d octets\n", interpreter->arena_used_bytes());
}

// Variables globales à rajouter en haut de ton fichier (avant le setup)
// La taille dépend de ta config : 2 secondes * 8000 Hz = 16000 échantillons
constexpr int AUDIO_BUFFER_SIZE = 16000;
int16_t audio_buffer[AUDIO_BUFFER_SIZE]; 

// La taille de ton tableau MFCC dépend de la sortie de ton script Python
// Exemple : 20 coefficients sur X trames temporelles.
float tableau_mfcc[1000]; // Ajuste la taille selon l'architecture de ton modèle

void loop() {
  // 1. Acquisition audio (Bloquant jusqu'à ce que le buffer soit plein, ou via DMA)
  // Il faut acquérir 2 secondes d'audio à 8000 Hz.
  bool audio_ready = record_audio(audio_buffer, AUDIO_BUFFER_SIZE);
  
  if (!audio_ready) {
    return; // On attend d'avoir suffisamment de données
  }

  // 2. Extraction des caractéristiques (DSP)
  // C'est ici que tu appelles arduinoFFT pour transformer l'audio brut en MFCC
  extract_features_from_audio(audio_buffer, tableau_mfcc);

  // 3. Remplissage du tenseur d'entrée
  const int taille_entree = input->bytes / sizeof(float);
  
  // Vérification de sécurité pour éviter un débordement de mémoire (Buffer Overflow)
  // Assure-toi que ton tableau_mfcc généré fait bien la même taille que l'entrée attendue
  for (int i = 0; i < taille_entree; i++) {
    input->data.f[i] = tableau_mfcc[i]; 
  }

  // 4. Exécuter l'inférence
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Erreur: L'inférence a échoué !");
    return;
  }

  // 5. Lire le résultat de la prédiction
  float prob_tronconneuse = output->data.f[0];
  Serial.printf("Probabilité Tronçonneuse : %f\n", prob_tronconneuse);

  // Pas de delay(1000) ici ! On repart immédiatement enregistrer la trame suivante.
}