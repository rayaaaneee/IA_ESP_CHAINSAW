#ifndef DEBUG_H
#define DEBUG_H

#include <cstddef>
#include <cstdint>

#include <tensorflow/lite/c/common.h>

namespace debug {
    
    // Prints debug information about the audio buffer, including minimum, maximum, mean, and non-zero sample count.
    void print_audio_debug(const int16_t* buffer, size_t size);

    // Prints debug information about the feature vector, including the first five features and checks for invalid values.
    bool print_feature_debug(const float* features, size_t size);

    // Prints debug information about the TensorFlow Lite tensor, including its type, dimensions, and first few values.
    bool print_tensor_debug(const TfLiteTensor* tensor);

}  // namespace debug

#endif // DEBUG_H