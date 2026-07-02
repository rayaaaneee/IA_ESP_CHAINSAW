import os

import tensorflow as tf

# 1. Charger ton modèle Keras/TensorFlow préalablement entraîné
model = tf.keras.models.load_model('training/model.h5')

# 2. Convertir le modèle au format TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optionnel : Ajouter des optimisations pour réduire la taille (Quantification)
# C'est très important pour TinyML sur ESP32
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# 3. Sauvegarder le modèle .tflite (pour vérification)
with open('firmware/src/model.tflite', 'wb') as f:
    f.write(tflite_model)

# 4. Convertir le modèle binaire en fichier header C++ (.h)
# Ce code écrit un tableau d'octets que le compilateur C++ peut lire
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

# 5. Écrire le fichier final model.h dans ton dossier firmware
with open('firmware/src/model.h', 'w') as f:
    f.write(header_content)

print("Conversion terminée : model.h généré avec succès !")