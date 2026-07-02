import os

import tensorflow as tf

# Load the trained Keras/TensorFlow pretrained model 
model = tf.keras.models.load_model('training/model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Quantization for reduce model size (optional but recommended for TinyML on ESP32)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the TensorFlow Lite model to a .tflite file
with open('firmware/src/model/model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert the .tflite model to a C array and write it to model.h
# This code writes the model data as a C array in a header file for use in embedded systems like ESP32.
def hex_to_c_array(hex_data, var_name):
    c_str = f"unsigned char {var_name}[] = {{"
    for i, byte in enumerate(hex_data):
        if i % 12 == 0: c_str += "\n  "
        c_str += f"0x{byte:02x}, "
    c_str = c_str[:-2] + "\n};"
    return c_str

header_content = "#ifndef MODEL_H\n#define MODEL_H\n\n"
header_content += hex_to_c_array(tflite_model, "g_model_data")
header_content += "\n\n#endif // MODEL_H"
header_content += "\n\nconst unsigned int g_model_data_len = " + str(len(tflite_model)) + ";"

# Write the header content to model.h
with open('firmware/src/model/model.h', 'w') as f:
    f.write(header_content)

print("Conversion terminée : model.h généré avec succès !")