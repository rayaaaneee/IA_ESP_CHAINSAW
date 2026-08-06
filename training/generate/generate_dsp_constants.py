import librosa
import numpy as np
import scipy.fftpack

from train import FeatureConfig

from .globals import FIRMWARE_DIR

DSP_CONGIG_CPP_PATH = FIRMWARE_DIR / "src" / "app" / "dsp_constants.cpp"

def export_to_c(name: str, len_var_name: str, matrix: np.ndarray) -> str:
    flat = matrix.flatten()
    result = f"const float {name}[{len_var_name}] = {{\n"
    formatted_values = [f"{x:.6f}f" for x in flat]
    chunks = [formatted_values[i:i + 6] for i in range(0, len(formatted_values), 6)]
    lines = [", ".join(chunk) for chunk in chunks]
    result += ",\n".join([f"    {line}" for line in lines])
    result += "\n};\n"
    return result

def main():
    # Mel matrix and DCT matrix generation
    mel_basis = librosa.filters.mel(sr=FeatureConfig.sample_rate, n_fft=FeatureConfig.fft_length, n_mels=FeatureConfig.n_mels)

    dct_basis = scipy.fftpack.dct(np.eye(FeatureConfig.n_mels), axis=0, type=2, norm='ortho')[:FeatureConfig.n_mfcc]

    content = (
f"""#include "app/dsp_constants.h"

{ export_to_c("MEL_MATRIX", "mel_matrix_data_len", mel_basis) }

{ export_to_c("DCT_MATRIX", "dct_matrix_data_len", dct_basis) }
"""
    )

    with open(DSP_CONGIG_CPP_PATH, "w") as f:
        f.write(content)

    print(f"Generated DSP constants in { DSP_CONGIG_CPP_PATH }")

if __name__ == "__main__":
    main()