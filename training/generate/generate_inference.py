import hashlib
import re
from pathlib import Path

from train import TFLITE_MODEL_PATH

from .globals import FIRMWARE_DIR

MODEL_CPP_PATH = FIRMWARE_DIR / "src" / "model" / "inference.cpp"


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def verify_tflite_hash(model_cpp_path: Path, expected_hash: str) -> bool:
    """
    Verify that the generated C array in inference.cpp matches the expected
    SHA-256 hash of the original TFLite model.
    """
    with open(model_cpp_path, "r", encoding="utf-8") as f:
        generated_cpp = f.read()

    # Use regex to extract the g_model_data array from the generated C++ file
    match = re.search(
        r"g_model_data\[\]\s*=\s*\{(.*?)\};",
        generated_cpp,
        re.DOTALL,
    )

    if match is None:
        raise RuntimeError("Could not locate g_model_data in inference.cpp.")

    array_content = match.group(1)

    # Extract all bytes written as 0xXX
    hex_values = re.findall(r"0x([0-9a-fA-F]{2})", array_content)
    generated_bytes = bytes(int(v, 16) for v in hex_values)

    generated_hash = compute_sha256(generated_bytes)

    return generated_hash == expected_hash

# Convert the .tflite model to a C array and write it to model.h
# This code writes the model data as a C array in a header file for use in embedded systems like ESP32.
def hex_to_c_array(hex_data, var_name):
    c_str = f"alignas(8) const uint8_t {var_name}[] = {{"
    for i, byte in enumerate(hex_data):
        if i % 12 == 0: c_str += "\n    "
        c_str += f"0x{byte:02x}, "
    c_str = c_str[:-2] + "\n};"
    return c_str


def main():

    with open(TFLITE_MODEL_PATH, 'rb') as f:
        tflite_model = f.read()

    tflite_hash = compute_sha256(tflite_model)

    # Generate the content of the .cpp file (DEFINITION)
    cpp_content = '#include "model/inference.h"\n\n'
    cpp_content += f'// Source TFLite SHA-256: {tflite_hash}\n'
    cpp_content += f'// Source TFLite size: {len(tflite_model)} bytes\n\n'
    cpp_content += hex_to_c_array(tflite_model, "g_model_data")
    cpp_content += f"\n\nconst uint32_t g_model_data_len = {len(tflite_model)};\n"

    with open(MODEL_CPP_PATH, 'w') as f:
        f.write(cpp_content)

    if not verify_tflite_hash(MODEL_CPP_PATH, tflite_hash):
        MODEL_CPP_PATH.unlink(missing_ok=True)
        raise RuntimeError("Generated inference.cpp does not match the expected TFLite model hash.")

    print(f"Generated C++ inference model in { MODEL_CPP_PATH }")
    print(f"Source TFLite SHA-256: {tflite_hash}")


if __name__ == "__main__":
    main()