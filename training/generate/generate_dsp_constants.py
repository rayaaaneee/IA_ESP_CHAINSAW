import librosa
import numpy as np
import scipy.fftpack

from .globals import FIRMWARE_DIR

DSP_CONGIG_CPP_PATH = FIRMWARE_DIR / "src" / "app" / "dsp_constants.cpp"

# Tes paramètres stricts
sr = 8000
n_fft = 1024
n_mels = 32
n_mfcc = 20

def main():
    # 1. Génération de la matrice MEL
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

    # 2. Génération de la matrice DCT (identique à librosa en interne)
    # On applique la DCT (Type 2, norme orthogonale) sur une matrice identité (eye)
    dct_basis = scipy.fftpack.dct(np.eye(n_mels), axis=0, type=2, norm='ortho')[:n_mfcc]

    def export_to_c(name, matrix: np.ndarray) -> str:
        flat = matrix.flatten()
        result = f"const float {name}[{len(flat)}] = {{\n"
        values = ", ".join([f"{x:.6f}f" for x in flat])
        result += f"    {values}\n"
        result += "};\n"
        return result

    content = (
f"""#include "app/dsp_constants.h"

{ export_to_c("MEL_MATRIX", mel_basis) }
{ export_to_c("DCT_MATRIX", dct_basis) }"""
    )

    with open(DSP_CONGIG_CPP_PATH, "w") as f:
        f.write(content)

    print(f"Generated DSP constants in { DSP_CONGIG_CPP_PATH }")

if __name__ == "__main__":
    main()