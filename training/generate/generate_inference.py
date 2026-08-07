import hashlib
from pathlib import Path

from train import TFLITE_MODEL_PATH

from .globals import FIRMWARE_DIR

MODEL_CPP_PATH = FIRMWARE_DIR / "src" / "model" / "inference.cpp"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():

    with open(TFLITE_MODEL_PATH, 'rb') as f:
        tflite_model = f.read()

    tflite_hash = compute_sha256(tflite_model)

    # Convert the .tflite model to a C array and write it to model.h
    # This code writes the model data as a C array in a header file for use in embedded systems like ESP32.
    def hex_to_c_array(hex_data, var_name):
        c_str = f"alignas(8) const uint8_t {var_name}[] = {{"
        for i, byte in enumerate(hex_data):
            if i % 12 == 0: c_str += "\n    "
            c_str += f"0x{byte:02x}, "
        c_str = c_str[:-2] + "\n};"
        return c_str

    # Generate the content of the .cpp file (DEFINITION)
    cpp_content = '#include "model/inference.h"\n\n'
    cpp_content += f'// Source TFLite SHA-256: {tflite_hash}\n'
    cpp_content += f'// Source TFLite size: {len(tflite_model)} bytes\n\n'
    cpp_content += hex_to_c_array(tflite_model, "g_model_data")
    cpp_content += f"\n\nconst uint32_t g_model_data_len = {len(tflite_model)};\n"

    with open(MODEL_CPP_PATH, 'w') as f:
        f.write(cpp_content)

    with open(MODEL_CPP_PATH, 'r', encoding='utf-8') as f:
        generated_cpp = f.read()

    if f"// Source TFLite SHA-256: {tflite_hash}" not in generated_cpp:
        raise RuntimeError("Generated inference.cpp does not contain the expected TFLite hash metadata.")

    print(f"Generated C++ inference model in { MODEL_CPP_PATH }")
    print(f"Source TFLite SHA-256: {tflite_hash}")

if __name__ == "__main__":
    main()