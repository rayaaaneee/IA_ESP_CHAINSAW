#ifndef DSP_CONSTANTS_H
#define DSP_CONSTANTS_H
#include "app/config.h"

// Matrix used for the Mel Filterbank in MFCC computation.
inline constexpr uint32_t mel_matrix_data_len = AUDIO_CONFIG.n_mels * (AUDIO_CONFIG.fft_length / 2 + 1);
extern const float MEL_MATRIX[mel_matrix_data_len];

// Matrix used for the Discrete Cosine Transform (DCT) in MFCC computation.
inline constexpr uint32_t dct_matrix_data_len = AUDIO_CONFIG.n_mfcc * AUDIO_CONFIG.n_mels;
extern const float DCT_MATRIX[dct_matrix_data_len];

#endif // DSP_CONSTANTS_H