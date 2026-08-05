from pathlib import Path

import tensorflow as tf

from train import MODEL_PATH, TFLITE_MODEL_PATH

# Load the trained Keras/TensorFlow pretrained model
model = tf.keras.models.load_model(MODEL_PATH)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Quantization for reduce model size (optional but recommended for TinyML on ESP32)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the TensorFlow Lite model to a .tflite file
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)

print("Model successfully converted to TensorFlow Lite format.")