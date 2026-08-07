#include <Arduino.h>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "app/config.h"
#include "drivers/audio.h"
#include "model/inference.h"
#include "services/mfcc.h"
#include "utils/debug.h"

bool extract_features_from_audio(const int16_t* audio_buffer, float* feature_vector, size_t feature_vector_size);

const tflite::Model* tflite_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Use arena_used_bytes() to check how much of the arena is actually used after AllocateTensors() is called after each model change.
constexpr int kTensorArenaSize = 6 * 1024;

alignas(16) uint8_t tensor_arena[kTensorArenaSize];

constexpr size_t kAudioBufferSize = AUDIO_WINDOW_SAMPLES;
constexpr float kInferenceThreshold = DETECTION_THRESHOLD;

int16_t audio_buffer[kAudioBufferSize];
float feature_vector[FEATURE_VECTOR_SIZE];

// OLED ssd1306 display(0x3C, 21, 22); // I2C address and pins for SDA and SCL
void setup() {

    Serial.begin(115200);
    delay(2000);

    Serial.println("Initializing the AI...");

    tflite_model = tflite::GetModel(g_model_data);
    // debug::print_model_tensor_types(tflite_model);

    static tflite::AllOpsResolver resolver;
    static tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter = &micro_error_reporter;

    static tflite::MicroInterpreter static_interpreter(
        tflite_model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      Serial.println("Error: unable to allocate tensors!");
      while (1) {
      }
    }

    input = interpreter->input(0);
    output = interpreter->output(0);

    init_audio();
    init_feature_extractor();

    Serial.println("AI model loaded and ready for inference.");
    Serial.printf("Arena used: %d bytes\n", interpreter->arena_used_bytes());
    Serial.printf("Feature vector size: %u\n", static_cast<unsigned>(FEATURE_VECTOR_SIZE));
    Serial.printf("Detection threshold: %.2f\n", kInferenceThreshold);
    // debug::print_tensor_debug(input);
    // debug::print_tensor_debug(output);

}

void loop() {

    if (!record_audio(audio_buffer, static_cast<int>(kAudioBufferSize))) {
      Serial.println("Error: unable to read audio.");
      return;
    }

    if (!extract_features_from_audio(audio_buffer, feature_vector, FEATURE_VECTOR_SIZE)) {
      Serial.println("Error: unable to extract features.");
      return;
    }

    const int input_size = input->bytes / static_cast<int>(sizeof(float));
    if (input_size != static_cast<int>(FEATURE_VECTOR_SIZE)) {
      Serial.printf("Error: unexpected input size (%d, expected %u).\n", input_size, static_cast<unsigned>(FEATURE_VECTOR_SIZE));
      return;
    }

    for (int index = 0; index < input_size; ++index) {
      input->data.f[index] = feature_vector[index];
    }

    // debug::print_audio_debug(audio_buffer, kAudioBufferSize);
    /* if (!debug::print_feature_debug(feature_vector, FEATURE_VECTOR_SIZE)) {
      Serial.println("Error: invalid feature vector.");
      return;
    } */
    // debug::print_input_tensor_debug(input, 16);

    if (interpreter->Invoke() != kTfLiteOk) {
      Serial.println("Error: inference failed!");
      return;
    }

    /* if (!debug::print_tensor_debug(output)) {
      Serial.println("Error: invalid model output.");
      return;
    } */

    const float probability = output->data.f[0];
    const bool detected = is_chainsaw_detected(probability, kInferenceThreshold);
    Serial.printf("Chainsaw probability: %.6f | verdict: %s\n", probability, detected ? "CHAINSAW DETECTED" : "NO CHAINSAW");

}