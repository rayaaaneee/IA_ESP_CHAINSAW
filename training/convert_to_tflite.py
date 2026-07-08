from pathlib import Path

import tensorflow as tf

from train import MODEL_PATH

MODEL_OUT_PATH = Path(__file__).resolve().parent.parent / "firmware" / "model" / "model.tflite"

MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

MODEL_HEADER_PATH = Path(__file__).resolve().parent.parent / "firmware" / "model" / "model.h"

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

header_content = "#ifndef MODEL_H\n#define MODEL_H\n\n"
header_content += hex_to_c_array(tflite_model, "g_model_data")
header_content += "\n\nconst unsigned int g_model_data_len = " + str(len(tflite_model)) + ";"
header_content += "\n\n#endif // MODEL_H"

# Write the header content to model.h
with open(MODEL_HEADER_PATH, 'w') as f:
    f.write(header_content)

print("Model successfully converted to TensorFlow Lite format and saved to model.tflite and model.h")