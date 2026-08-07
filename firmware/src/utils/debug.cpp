#include "utils/debug.h"

#include <Arduino.h>

#include <cmath>

namespace debug {
void print_audio_debug(const int16_t* buffer, size_t size) {
  int16_t minimum_value = buffer[0];
  int16_t maximum_value = buffer[0];
  long long sum = 0;
  size_t non_zero_count = 0;

  for (size_t index = 0; index < size; ++index) {
    const int16_t value = buffer[index];
    if (value < minimum_value) {
      minimum_value = value;
    }
    if (value > maximum_value) {
      maximum_value = value;
    }
    if (value != 0) {
      ++non_zero_count;
    }
    sum += value;
  }

  Serial.printf("Audio debug: min=%d max=%d mean=%.2f non_zero=%u first=%d second=%d third=%d\n",
                minimum_value,
                maximum_value,
                static_cast<double>(sum) / static_cast<double>(size),
                static_cast<unsigned>(non_zero_count),
                buffer[0],
                buffer[1],
                buffer[2]);
}

bool print_feature_debug(const float* features, size_t size) {
  bool has_invalid_value = false;
  size_t invalid_index = 0;

  for (size_t index = 0; index < size; ++index) {
    const float value = features[index];
    if (!std::isfinite(value)) {
      has_invalid_value = true;
      invalid_index = index;
      break;
    }
  }

  Serial.printf("Feature debug: f0=%.6f f1=%.6f f2=%.6f f3=%.6f f4=%.6f\n",
                features[0],
                features[1],
                features[2],
                features[3],
                features[4]);

  if (has_invalid_value) {
    Serial.printf("Feature debug: invalid value at index %u -> %.6f\n",
                  static_cast<unsigned>(invalid_index),
                  features[invalid_index]);
  }

  return !has_invalid_value;
}

void print_input_tensor_debug(const TfLiteTensor* tensor, size_t preview_count) {
  if (tensor == nullptr || tensor->type != kTfLiteFloat32 || tensor->bytes < static_cast<int>(sizeof(float))) {
    Serial.println("Input tensor debug: invalid input tensor state.");
    return;
  }

  const size_t tensor_size = static_cast<size_t>(tensor->bytes / static_cast<int>(sizeof(float)));
  const size_t values_to_print = preview_count < tensor_size ? preview_count : tensor_size;

  Serial.printf("Input tensor preview (%u values):\n", static_cast<unsigned>(values_to_print));

  float min_value = tensor->data.f[0];
  float max_value = tensor->data.f[0];

  for (size_t index = 0; index < values_to_print; ++index) {
    const float value = tensor->data.f[index];
    if (!std::isfinite(value)) {
      Serial.printf("  [%03u] = NaN/Inf\n", static_cast<unsigned>(index));
    } else {
      Serial.printf("  [%03u] = %.6f\n", static_cast<unsigned>(index), value);
    }

    if (value < min_value) {
      min_value = value;
    }
    if (value > max_value) {
      max_value = value;
    }
  }

  Serial.printf("Input tensor range: min=%.6f | max=%.6f\n", min_value, max_value);
}

bool print_tensor_debug(const TfLiteTensor* tensor) {
  if (tensor == nullptr || tensor->type != kTfLiteFloat32 || tensor->bytes < static_cast<int>(sizeof(float))) {
    Serial.println("Tensor debug: invalid output tensor state.");
    return false;
  }

  const float probability = tensor->data.f[0];
  if (!std::isfinite(probability)) {
    Serial.printf("Tensor debug: output probability is invalid -> %.6f\n", probability);
    return false;
  }

  return true;
}

bool run_model_debug(tflite::MicroInterpreter* interpreter,
                     TfLiteTensor* input,
                     TfLiteTensor* output,
                     const float* features,
                     size_t size,
                     float threshold,
                     const char* label) {
  if (interpreter == nullptr || input == nullptr || output == nullptr || features == nullptr) {
    Serial.println("Debug: invalid model pointers.");
    return false;
  }

  const int input_size = input->bytes / static_cast<int>(sizeof(float));
  if (input_size != static_cast<int>(size)) {
    Serial.printf("%s: unexpected input size (%d, expected %u).\n", label, input_size, static_cast<unsigned>(size));
    return false;
  }

  if (input->type != kTfLiteFloat32 || output->type != kTfLiteFloat32) {
    Serial.printf("%s: model tensors are not float32.\n", label);
    return false;
  }

  for (int index = 0; index < input_size; ++index) {
    input->data.f[index] = features[index];
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.printf("%s: inference failed.\n", label);
    return false;
  }

  const float probability = output->data.f[0];
  if (!std::isfinite(probability)) {
    Serial.printf("%s: output probability is invalid -> %.6f\n", label, probability);
    return false;
  }

  const bool detected = probability >= threshold;
  Serial.printf("%s: chainsaw probability = %.6f | verdict: %s\n", label, probability, detected ? "CHAINSAW DETECTED" : "NO CHAINSAW");
  return true;
}
}  // namespace debug