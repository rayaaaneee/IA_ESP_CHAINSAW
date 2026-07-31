from pathlib import Path

import tensorflow as tf

from train import MODEL_PATH

MODEL_BASE_PATH = Path(__file__).resolve().parent.parent

MODEL_OUT_PATH = Path(__file__).resolve().parent / "model" / "model.tflite"
MODEL_HEADER_PATH = MODEL_BASE_PATH / "firmware" / "include" / "model" / "inference.h"
MODEL_CPP_PATH = MODEL_BASE_PATH / "firmware" / "src" / "model" / "inference.cpp"

MODEL_HEADER_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load the trained Keras/TensorFlow pretrained model 

model = tf.keras.models.load_model(MODEL_PATH)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Quantization for reduce model size (optional but recommended for TinyML on ESP32)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the TensorFlow Lite model to a .tflite file
with open(MODEL_OUT_PATH, 'wb') as f:
    f.write(tflite_model)

# Convert the .tflite model to a C array and write it to model.h
# This code writes the model data as a C array in a header file for use in embedded systems like ESP32.
def hex_to_c_array(hex_data, var_name):
    c_str = f"alignas(8) const unsigned char {var_name}[] = {{"
    for i, byte in enumerate(hex_data):
        if i % 12 == 0: c_str += "\n  "
        c_str += f"0x{byte:02x}, "
    c_str = c_str[:-2] + "\n};"
    return c_str

# Generate the content of the .h file (DECLARATION)
header_content = """#ifndef INFERENCE_H
#define INFERENCE_H

// Declaration of the TensorFlow Lite model data
extern const unsigned char g_model_data[];

// Declaration of the length of the TensorFlow Lite model data
extern const unsigned int g_model_data_len;

#endif // INFERENCE_H
"""

# Generate the content of the .cpp file (DEFINITION)
cpp_content = '#include "model/inference.h"\n\n'
cpp_content += hex_to_c_array(tflite_model, "g_model_data")
cpp_content += f"\n\nconst unsigned int g_model_data_len = {len(tflite_model)};\n"

# Write the header and cpp files
with open(MODEL_HEADER_PATH, 'w') as f:
    f.write(header_content)

with open(MODEL_CPP_PATH, 'w') as f:
    f.write(cpp_content)

print("Model successfully converted to TensorFlow Lite format and saved to model.tflite and inference.h")