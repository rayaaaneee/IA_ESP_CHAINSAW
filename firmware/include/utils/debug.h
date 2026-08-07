#ifndef DEBUG_H
#define DEBUG_H

#include <cstddef>
#include <cstdint>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/micro/micro_interpreter.h>

namespace debug {

    // Prints debug information about the audio buffer, including minimum, maximum, mean, and non-zero sample count.
    void print_audio_debug(const int16_t* buffer, size_t size);

    // Prints debug information about the feature vector, including the first five features and checks for invalid values.
    bool print_feature_debug(const float* features, size_t size);

    // Prints a preview of the input tensor values before inference.
    void print_input_tensor_debug(const TfLiteTensor* tensor, size_t preview_count);

    // Prints debug information about the TensorFlow Lite tensor, including its type, dimensions, and first few values.
    bool print_tensor_debug(const TfLiteTensor* tensor);

    // Runs inference on the provided feature vector and prints the verdict.
    bool run_model_debug(tflite::MicroInterpreter* interpreter,
                         TfLiteTensor* input,
                         TfLiteTensor* output,
                         const float* features,
                         size_t size,
                         float threshold,
                         const char* label);

}  // namespace debug

#endif // DEBUG_H