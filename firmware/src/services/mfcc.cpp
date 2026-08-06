#include "services/mfcc.h"

#include <ArduinoFFT.h>

#include <cmath>

#include "app/dsp_constants.h"

namespace {
constexpr size_t kFftLength = AUDIO_CONFIG.fft_length;
constexpr size_t kFftBins = (AUDIO_CONFIG.fft_length / 2) + 1;
constexpr size_t kMfccCount = AUDIO_CONFIG.n_mfcc;
constexpr size_t kMelCount = AUDIO_CONFIG.n_mels;
constexpr size_t kDeltaWindow = 9;
constexpr size_t kDeltaHalfWindow = kDeltaWindow / 2;
constexpr int kRmsFrameLength = 2048;
constexpr int kRmsPad = kRmsFrameLength / 2;
constexpr float kEpsilon = 1e-10f;
constexpr float kTopDb = 80.0f;
constexpr float kRollPercent = 0.85f;

float g_delta_coeffs[kDeltaWindow];
float g_delta2_coeffs[kDeltaWindow];
bool g_initialized = false;
float g_signal[AUDIO_WINDOW_SAMPLES];
float g_frame_real[kFftLength];
float g_frame_imag[kFftLength];
float g_magnitude[kFftBins];
float g_mel_energies[kMelCount][FEATURE_FRAME_COUNT];
float g_mel_db[kMelCount][FEATURE_FRAME_COUNT];
float g_mfcc_matrix[kMfccCount][FEATURE_FRAME_COUNT];
float g_delta_matrix[kMfccCount][FEATURE_FRAME_COUNT];
float g_delta2_matrix[kMfccCount][FEATURE_FRAME_COUNT];
float g_centroid[FEATURE_FRAME_COUNT];
float g_bandwidth[FEATURE_FRAME_COUNT];
float g_rolloff[FEATURE_FRAME_COUNT];
float g_zcr[FEATURE_FRAME_COUNT];
float g_rms[FEATURE_FRAME_COUNT];

float hann_window(size_t index, size_t size) {
    return 0.5f - 0.5f * cosf((2.0f * PI * static_cast<float>(index)) / static_cast<float>(size));
}

int reflect_index(int index, int length) {
    if (length <= 1) {
        return 0;
    }

    const int period = 2 * length - 2;
    index %= period;
    if (index < 0) {
        index += period;
    }
    if (index >= length) {
        index = period - index;
    }
    return index;
}

float sample_reflect(const float* signal, int length, int index) {
    return signal[reflect_index(index, length)];
}

bool invert_2x2(const double in[2][2], double out[2][2]) {
    const double det = in[0][0] * in[1][1] - in[0][1] * in[1][0];
    if (fabs(det) < 1e-12) {
        return false;
    }

    const double inv_det = 1.0 / det;
    out[0][0] = in[1][1] * inv_det;
    out[0][1] = -in[0][1] * inv_det;
    out[1][0] = -in[1][0] * inv_det;
    out[1][1] = in[0][0] * inv_det;
    return true;
}

bool invert_3x3(const double in[3][3], double out[3][3]) {
    const double a = in[0][0];
    const double b = in[0][1];
    const double c = in[0][2];
    const double d = in[1][0];
    const double e = in[1][1];
    const double f = in[1][2];
    const double g = in[2][0];
    const double h = in[2][1];
    const double i = in[2][2];

    const double cofactor00 = e * i - f * h;
    const double cofactor01 = -(d * i - f * g);
    const double cofactor02 = d * h - e * g;
    const double cofactor10 = -(b * i - c * h);
    const double cofactor11 = a * i - c * g;
    const double cofactor12 = -(a * h - b * g);
    const double cofactor20 = b * f - c * e;
    const double cofactor21 = -(a * f - c * d);
    const double cofactor22 = a * e - b * d;

    const double det = a * cofactor00 + b * cofactor01 + c * cofactor02;
    if (fabs(det) < 1e-12) {
        return false;
    }

    const double inv_det = 1.0 / det;
    out[0][0] = cofactor00 * inv_det;
    out[0][1] = cofactor10 * inv_det;
    out[0][2] = cofactor20 * inv_det;
    out[1][0] = cofactor01 * inv_det;
    out[1][1] = cofactor11 * inv_det;
    out[1][2] = cofactor21 * inv_det;
    out[2][0] = cofactor02 * inv_det;
    out[2][1] = cofactor12 * inv_det;
    out[2][2] = cofactor22 * inv_det;
    return true;
}

void build_delta_coefficients(int derivative_order, float* coeffs) {
    if (derivative_order == 1) {
        const double ata[2][2] = {
            {9.0, 0.0},
            {0.0, 60.0},
        };
        double ata_inv[2][2] = {};
        if (!invert_2x2(ata, ata_inv)) {
            return;
        }

        for (int offset = -static_cast<int>(kDeltaHalfWindow); offset <= static_cast<int>(kDeltaHalfWindow); ++offset) {
            const int column = offset + static_cast<int>(kDeltaHalfWindow);
            const double basis[2] = {1.0, static_cast<double>(offset)};
            double coefficient = 0.0;
            for (int j = 0; j < 2; ++j) {
                coefficient += ata_inv[1][j] * basis[j];
            }
            coeffs[column] = static_cast<float>(coefficient);
        }
        return;
    }

    const double ata[3][3] = {
        {9.0, 0.0, 60.0},
        {0.0, 60.0, 0.0},
        {60.0, 0.0, 708.0},
    };
    double ata_inv[3][3] = {};
    if (!invert_3x3(ata, ata_inv)) {
        return;
    }

    for (int offset = -static_cast<int>(kDeltaHalfWindow); offset <= static_cast<int>(kDeltaHalfWindow); ++offset) {
        const int column = offset + static_cast<int>(kDeltaHalfWindow);
        const double basis[3] = {1.0, static_cast<double>(offset), static_cast<double>(offset * offset)};
        double coefficient = 0.0;
        for (int j = 0; j < 3; ++j) {
            coefficient += ata_inv[2][j] * basis[j];
        }
        coeffs[column] = static_cast<float>(2.0 * coefficient);
    }
}

float compute_mean(const float* values, size_t count) {
    double sum = 0.0;
    for (size_t index = 0; index < count; ++index) {
        sum += values[index];
    }
    return static_cast<float>(sum / static_cast<double>(count));
}

float compute_std(const float* values, size_t count, float mean) {
    double sum = 0.0;
    for (size_t index = 0; index < count; ++index) {
        const double delta = static_cast<double>(values[index]) - static_cast<double>(mean);
        sum += delta * delta;
    }
    return static_cast<float>(sqrt(sum / static_cast<double>(count)));
}

void append_summary(float* features, size_t& write_index, const float* values, size_t count) {
    const float mean = compute_mean(values, count);
    const float std = compute_std(values, count, mean);
    features[write_index++] = mean;
    features[write_index++] = std;
}

void compute_delta_series(const float* source, int num_coeffs, int num_frames, const float* coeffs, float* destination) {
    for (int coeff = 0; coeff < num_coeffs; ++coeff) {
        const float* source_row = source + coeff * num_frames;
        float* destination_row = destination + coeff * num_frames;

        for (int frame = 0; frame < num_frames; ++frame) {
            double sum = 0.0;
            for (int offset = -static_cast<int>(kDeltaHalfWindow); offset <= static_cast<int>(kDeltaHalfWindow); ++offset) {
                const int coefficient_index = offset + static_cast<int>(kDeltaHalfWindow);
                const int sample_index = reflect_index(frame + offset, num_frames);
                sum += static_cast<double>(coeffs[coefficient_index]) * static_cast<double>(source_row[sample_index]);
            }
            destination_row[frame] = static_cast<float>(sum);
        }
    }
}

}  // namespace

void init_feature_extractor() {
    if (g_initialized) {
        return;
    }

    build_delta_coefficients(1, g_delta_coeffs);
    build_delta_coefficients(2, g_delta2_coeffs);
    g_initialized = true;
}

bool is_chainsaw_detected(float probability, float threshold) {
    return probability >= threshold;
}

bool extract_features_from_audio(const int16_t* audio_buffer, float* feature_vector, size_t feature_vector_size) {
    if (audio_buffer == nullptr || feature_vector == nullptr || feature_vector_size < FEATURE_VECTOR_SIZE) {
        return false;
    }

    init_feature_extractor();

    constexpr int signal_length = static_cast<int>(AUDIO_WINDOW_SAMPLES);
    constexpr int fft_length = static_cast<int>(kFftLength);
    constexpr int spectral_bins = static_cast<int>(kFftBins);
    constexpr int frame_count = static_cast<int>(FEATURE_FRAME_COUNT);

    float peak = 0.0f;
    for (size_t index = 0; index < AUDIO_WINDOW_SAMPLES; ++index) {
        g_signal[index] = static_cast<float>(audio_buffer[index]) / 32768.0f;
        peak = fmaxf(peak, fabsf(g_signal[index]));
    }

    if (peak > 0.0f) {
        const float inverse_peak = 1.0f / peak;
        for (size_t index = 0; index < AUDIO_WINDOW_SAMPLES; ++index) {
            g_signal[index] *= inverse_peak;
        }
    }

    float global_mel_max = 0.0f;

    for (int frame = 0; frame < frame_count; ++frame) {
        const int spectral_center = frame * static_cast<int>(AUDIO_CONFIG.hop_length);
        const int spectral_start = spectral_center - (fft_length / 2);

        for (int sample = 0; sample < fft_length; ++sample) {
            const int source_index = spectral_start + sample;
            g_frame_real[sample] = sample_reflect(g_signal, signal_length, source_index) * hann_window(sample, fft_length);
            g_frame_imag[sample] = 0.0f;
        }

        ArduinoFFT<float> fft(g_frame_real, g_frame_imag, kFftLength, AUDIO_CONFIG.sample_rate);
        fft.compute(FFT_FORWARD);
        fft.complexToMagnitude();

        double magnitude_sum = 0.0;
        double weighted_sum = 0.0;
        double weighted_square_sum = 0.0;

        for (int bin = 0; bin < spectral_bins; ++bin) {
            const float mag = g_frame_real[bin];
            g_magnitude[bin] = mag;
            const double frequency = static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length);
            magnitude_sum += mag;
            weighted_sum += frequency * mag;
        }

        const float centroid_value = magnitude_sum > 0.0 ? static_cast<float>(weighted_sum / magnitude_sum) : 0.0f;
        g_centroid[frame] = centroid_value;

        for (int bin = 0; bin < spectral_bins; ++bin) {
            const double frequency = static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length);
            const double distance = frequency - static_cast<double>(centroid_value);
            weighted_square_sum += distance * distance * g_magnitude[bin];
        }

        g_bandwidth[frame] = magnitude_sum > 0.0 ? static_cast<float>(sqrt(weighted_square_sum / magnitude_sum)) : 0.0f;

        const double rolloff_threshold = magnitude_sum * kRollPercent;
        double cumulative = 0.0;
        float rolloff_value = 0.0f;
        for (int bin = 0; bin < spectral_bins; ++bin) {
            cumulative += g_magnitude[bin];
            if (cumulative >= rolloff_threshold) {
                rolloff_value = static_cast<float>(static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length));
                break;
            }
        }
        g_rolloff[frame] = rolloff_value;

        const int rms_start = frame * static_cast<int>(AUDIO_CONFIG.hop_length) - kRmsPad;
        double zero_crossings = 0.0;
        double rms_sum = 0.0;
        float previous_sample = sample_reflect(g_signal, signal_length, rms_start);

        for (int sample = 0; sample < kRmsFrameLength; ++sample) {
            const float current_sample = sample_reflect(g_signal, signal_length, rms_start + sample);
            if ((previous_sample >= 0.0f && current_sample < 0.0f) || (previous_sample < 0.0f && current_sample >= 0.0f)) {
                zero_crossings += 1.0;
            }
            rms_sum += static_cast<double>(current_sample) * static_cast<double>(current_sample);
            previous_sample = current_sample;
        }

        g_zcr[frame] = static_cast<float>(zero_crossings / static_cast<double>(kRmsFrameLength - 1));
        g_rms[frame] = static_cast<float>(sqrt(rms_sum / static_cast<double>(kRmsFrameLength)));

        for (int mel = 0; mel < static_cast<int>(kMelCount); ++mel) {
            double energy = 0.0;
            const float* mel_row = &MEL_MATRIX[mel * spectral_bins];
            for (int bin = 0; bin < spectral_bins; ++bin) {
                energy += static_cast<double>(mel_row[bin]) * static_cast<double>(g_magnitude[bin]) * static_cast<double>(g_magnitude[bin]);
            }
            const float mel_energy = static_cast<float>(energy);
            g_mel_energies[mel][frame] = mel_energy;
            if (mel_energy > global_mel_max) {
                global_mel_max = mel_energy;
            }
        }
    }

    if (global_mel_max < kEpsilon) {
        global_mel_max = kEpsilon;
    }

    for (int frame = 0; frame < frame_count; ++frame) {
        for (int mel = 0; mel < static_cast<int>(kMelCount); ++mel) {
            const float mel_energy = g_mel_energies[mel][frame] < kEpsilon ? kEpsilon : g_mel_energies[mel][frame];
            float db = 10.0f * log10f(mel_energy / global_mel_max);
            if (db < -kTopDb) {
                db = -kTopDb;
            }
            g_mel_db[mel][frame] = db;
        }
    }

    for (int frame = 0; frame < frame_count; ++frame) {
        for (int mfcc = 0; mfcc < static_cast<int>(kMfccCount); ++mfcc) {
            double sum = 0.0;
            for (int mel = 0; mel < static_cast<int>(kMelCount); ++mel) {
                sum += static_cast<double>(DCT_MATRIX[mfcc * kMelCount + mel]) * static_cast<double>(g_mel_db[mel][frame]);
            }
            g_mfcc_matrix[mfcc][frame] = static_cast<float>(sum);
        }
    }

    compute_delta_series(&g_mfcc_matrix[0][0], static_cast<int>(kMfccCount), frame_count, g_delta_coeffs, &g_delta_matrix[0][0]);
    compute_delta_series(&g_mfcc_matrix[0][0], static_cast<int>(kMfccCount), frame_count, g_delta2_coeffs, &g_delta2_matrix[0][0]);

    size_t write_index = 0;
    for (int mfcc = 0; mfcc < static_cast<int>(kMfccCount); ++mfcc) {
        append_summary(feature_vector, write_index, g_mfcc_matrix[mfcc], FEATURE_FRAME_COUNT);
    }
    for (int mfcc = 0; mfcc < static_cast<int>(kMfccCount); ++mfcc) {
        append_summary(feature_vector, write_index, g_delta_matrix[mfcc], FEATURE_FRAME_COUNT);
    }
    for (int mfcc = 0; mfcc < static_cast<int>(kMfccCount); ++mfcc) {
        append_summary(feature_vector, write_index, g_delta2_matrix[mfcc], FEATURE_FRAME_COUNT);
    }
    for (int mel = 0; mel < static_cast<int>(kMelCount); ++mel) {
        append_summary(feature_vector, write_index, g_mel_db[mel], FEATURE_FRAME_COUNT);
    }

    append_summary(feature_vector, write_index, g_centroid, FEATURE_FRAME_COUNT);
    append_summary(feature_vector, write_index, g_bandwidth, FEATURE_FRAME_COUNT);
    append_summary(feature_vector, write_index, g_rolloff, FEATURE_FRAME_COUNT);
    append_summary(feature_vector, write_index, g_zcr, FEATURE_FRAME_COUNT);
    append_summary(feature_vector, write_index, g_rms, FEATURE_FRAME_COUNT);

    return write_index == FEATURE_VECTOR_SIZE;
}