#include "services/mfcc.h"
#include "services/mfcc_utils.h"

#include <ArduinoFFT.h>

#include <cmath>
#include <cstdlib>

#include "app/dsp_constants.h"

namespace {
struct RunningStats {
    double sum = 0.0;
    double sum_squares = 0.0;

    void add(float value) {
        const double value_as_double = static_cast<double>(value);
        sum += value_as_double;
        sum_squares += value_as_double * value_as_double;
    }

    float mean(size_t count) const {
        return count == 0 ? 0.0f : static_cast<float>(sum / static_cast<double>(count));
    }

    float stddev(size_t count) const {
        if (count == 0) {
            return 0.0f;
        }

        const double average = sum / static_cast<double>(count);
        const double variance = (sum_squares / static_cast<double>(count)) - (average * average);
        return static_cast<float>(variance > 0.0 ? std::sqrt(variance) : 0.0);
    }
};

void write_stats(float* feature_vector, size_t& write_index, const RunningStats* stats, size_t count, size_t sample_count) {
    for (size_t index = 0; index < count; ++index) {
        feature_vector[write_index++] = stats[index].mean(sample_count);
        feature_vector[write_index++] = stats[index].stddev(sample_count);
    }
}
}  // namespace

bool extract_features_from_audio(const int16_t* audio_buffer, float* feature_vector, size_t feature_vector_size) {
    if (audio_buffer == nullptr || feature_vector == nullptr || feature_vector_size < FEATURE_VECTOR_SIZE) {
        return false;
    }

    constexpr int fft_length = static_cast<int>(mfcc_utils::kFftLength);
    constexpr int spectral_bins = static_cast<int>(mfcc_utils::kFftBins);
    constexpr int frame_count = static_cast<int>(FEATURE_FRAME_COUNT);
    constexpr size_t mfcc_matrix_size = mfcc_utils::kMfccCount * FEATURE_FRAME_COUNT;
    constexpr size_t mel_energy_matrix_size = mfcc_utils::kMelCount * FEATURE_FRAME_COUNT;

    float* frame_real = static_cast<float*>(std::malloc(sizeof(float) * mfcc_utils::kFftLength));
    float* frame_imag = static_cast<float*>(std::malloc(sizeof(float) * mfcc_utils::kFftLength));
    float* mel_energies = static_cast<float*>(std::malloc(sizeof(float) * mel_energy_matrix_size));
    float* mfcc_matrix = static_cast<float*>(std::malloc(sizeof(float) * mfcc_matrix_size));

    if (frame_real == nullptr || frame_imag == nullptr || mel_energies == nullptr || mfcc_matrix == nullptr) {
        std::free(frame_real);
        std::free(frame_imag);
        std::free(mel_energies);
        std::free(mfcc_matrix);
        return false;
    }

    RunningStats mfcc_stats[mfcc_utils::kMfccCount] = {};
    RunningStats delta_stats[mfcc_utils::kMfccCount] = {};
    RunningStats delta2_stats[mfcc_utils::kMfccCount] = {};
    RunningStats mel_stats[mfcc_utils::kMelCount] = {};
    RunningStats centroid_stats = {};
    RunningStats bandwidth_stats = {};
    RunningStats rolloff_stats = {};
    RunningStats zcr_stats = {};
    RunningStats rms_stats = {};

    float peak = 0.0f;
    for (size_t index = 0; index < mfcc_utils::kAudioWindowSamples; ++index) {
        peak = fmaxf(peak, fabsf(static_cast<float>(audio_buffer[index]) / 32768.0f));
    }

    const float inverse_peak = peak > 0.0f ? 1.0f / peak : 1.0f;

    float global_mel_max = 0.0f;

    for (int frame = 0; frame < frame_count; ++frame) {
        const int spectral_center = frame * static_cast<int>(AUDIO_CONFIG.hop_length);
        const int spectral_start = spectral_center - (fft_length / 2);

        for (int sample = 0; sample < fft_length; ++sample) {
            const int source_index = mfcc_utils::reflect_index(spectral_start + sample, static_cast<int>(mfcc_utils::kAudioWindowSamples));
            frame_real[sample] = (static_cast<float>(audio_buffer[source_index]) / 32768.0f) * inverse_peak * mfcc_utils::hann_window(sample, fft_length);
            frame_imag[sample] = 0.0f;
        }

        ArduinoFFT<float> fft(frame_real, frame_imag, mfcc_utils::kFftLength, AUDIO_CONFIG.sample_rate);
        fft.compute(FFT_FORWARD);
        fft.complexToMagnitude();

        double magnitude_sum = 0.0;
        double weighted_sum = 0.0;
        double weighted_square_sum = 0.0;

        for (int bin = 0; bin < spectral_bins; ++bin) {
            const float magnitude = frame_real[bin];
            const double frequency = static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length);
            magnitude_sum += magnitude;
            weighted_sum += frequency * magnitude;
        }

        const float centroid_value = magnitude_sum > 0.0 ? static_cast<float>(weighted_sum / magnitude_sum) : 0.0f;

        for (int bin = 0; bin < spectral_bins; ++bin) {
            const float magnitude = frame_real[bin];
            const double frequency = static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length);
            const double distance = frequency - static_cast<double>(centroid_value);
            weighted_square_sum += distance * distance * magnitude;
        }

        const float bandwidth_value = magnitude_sum > 0.0 ? static_cast<float>(std::sqrt(weighted_square_sum / magnitude_sum)) : 0.0f;

        const double rolloff_threshold = magnitude_sum * mfcc_utils::kRollPercent;
        double cumulative = 0.0;
        float rolloff_value = 0.0f;
        for (int bin = 0; bin < spectral_bins; ++bin) {
            cumulative += frame_real[bin];
            if (cumulative >= rolloff_threshold) {
                rolloff_value = static_cast<float>(static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length));
                break;
            }
        }

        const int rms_start = frame * static_cast<int>(AUDIO_CONFIG.hop_length) - mfcc_utils::kRmsPad;
        double zero_crossings = 0.0;
        double rms_sum = 0.0;
        float previous_sample = static_cast<float>(audio_buffer[mfcc_utils::reflect_index(rms_start, static_cast<int>(mfcc_utils::kAudioWindowSamples))]) / 32768.0f * inverse_peak;

        for (int sample = 0; sample < mfcc_utils::kRmsFrameLength; ++sample) {
            const float current_sample = static_cast<float>(audio_buffer[mfcc_utils::reflect_index(rms_start + sample, static_cast<int>(mfcc_utils::kAudioWindowSamples))]) / 32768.0f * inverse_peak;
            if ((previous_sample >= 0.0f && current_sample < 0.0f) || (previous_sample < 0.0f && current_sample >= 0.0f)) {
                zero_crossings += 1.0;
            }
            rms_sum += static_cast<double>(current_sample) * static_cast<double>(current_sample);
            previous_sample = current_sample;
        }

        const float zcr_value = static_cast<float>(zero_crossings / static_cast<double>(mfcc_utils::kRmsFrameLength - 1));
        const float rms_value = static_cast<float>(std::sqrt(rms_sum / static_cast<double>(mfcc_utils::kRmsFrameLength)));

        for (int mel = 0; mel < static_cast<int>(mfcc_utils::kMelCount); ++mel) {
            double energy = 0.0;
            const float* mel_row = &MEL_MATRIX[mel * spectral_bins];
            for (int bin = 0; bin < spectral_bins; ++bin) {
                const float magnitude = frame_real[bin];
                energy += static_cast<double>(mel_row[bin]) * static_cast<double>(magnitude) * static_cast<double>(magnitude);
            }
            const float mel_energy = static_cast<float>(energy);
            mel_energies[frame * mfcc_utils::kMelCount + mel] = mel_energy;
            if (mel_energy > global_mel_max) {
                global_mel_max = mel_energy;
            }
        }

        centroid_stats.add(centroid_value);
        bandwidth_stats.add(bandwidth_value);
        rolloff_stats.add(rolloff_value);
        zcr_stats.add(zcr_value);
        rms_stats.add(rms_value);
    }

    if (global_mel_max < mfcc_utils::kEpsilon) {
        global_mel_max = mfcc_utils::kEpsilon;
    }

    // matches librosa.feature.mfcc()'s internal power_to_db(..., ref=1.0): absolute scale, only the top_db floor is window-relative
    const float global_mfcc_max_db = 10.0f * std::log10(global_mel_max);

    for (int frame = 0; frame < frame_count; ++frame) {
        float mfcc_frame[mfcc_utils::kMfccCount] = {};
        const float* mel_energy_row = mel_energies + (frame * mfcc_utils::kMelCount);

        for (int mel = 0; mel < static_cast<int>(mfcc_utils::kMelCount); ++mel) {
            float mel_energy = mel_energy_row[mel] < mfcc_utils::kEpsilon ? mfcc_utils::kEpsilon : mel_energy_row[mel];

            float mfcc_db = 10.0f * std::log10(mel_energy);
            if (mfcc_db < global_mfcc_max_db - mfcc_utils::kTopDb) {
                mfcc_db = global_mfcc_max_db - mfcc_utils::kTopDb;
            }

            // matches extract_features.py's standalone `librosa.power_to_db(mel, ref=np.max)` used for the "mel" feature block
            float mel_db = 10.0f * std::log10(mel_energy / global_mel_max);
            if (mel_db < -mfcc_utils::kTopDb) {
                mel_db = -mfcc_utils::kTopDb;
            }

            mel_stats[mel].add(mel_db);

            for (int mfcc = 0; mfcc < static_cast<int>(mfcc_utils::kMfccCount); ++mfcc) {
                mfcc_frame[mfcc] += static_cast<float>(DCT_MATRIX[mfcc * mfcc_utils::kMelCount + mel]) * mfcc_db;
            }
        }

        for (int mfcc = 0; mfcc < static_cast<int>(mfcc_utils::kMfccCount); ++mfcc) {
            mfcc_matrix[frame * mfcc_utils::kMfccCount + mfcc] = mfcc_frame[mfcc];
            mfcc_stats[mfcc].add(mfcc_frame[mfcc]);
        }
    }

    std::free(mel_energies);

    float delta_row[FEATURE_FRAME_COUNT] = {};
    float delta2_row[FEATURE_FRAME_COUNT] = {};

    for (int mfcc = 0; mfcc < static_cast<int>(mfcc_utils::kMfccCount); ++mfcc) {
        const float* source_row = mfcc_matrix + (mfcc * FEATURE_FRAME_COUNT);
        mfcc_utils::compute_delta_series(source_row, 1, frame_count, mfcc_utils::g_delta_coeffs, delta_row);
        mfcc_utils::compute_delta_series(source_row, 1, frame_count, mfcc_utils::g_delta2_coeffs, delta2_row);

        for (int frame = 0; frame < frame_count; ++frame) {
            delta_stats[mfcc].add(delta_row[frame]);
            delta2_stats[mfcc].add(delta2_row[frame]);
        }
    }

    std::free(frame_real);
    std::free(frame_imag);
    std::free(mfcc_matrix);

    size_t write_index = 0;
    write_stats(feature_vector, write_index, mfcc_stats, mfcc_utils::kMfccCount, FEATURE_FRAME_COUNT);
    write_stats(feature_vector, write_index, delta_stats, mfcc_utils::kMfccCount, FEATURE_FRAME_COUNT);
    write_stats(feature_vector, write_index, delta2_stats, mfcc_utils::kMfccCount, FEATURE_FRAME_COUNT);
    write_stats(feature_vector, write_index, mel_stats, mfcc_utils::kMelCount, FEATURE_FRAME_COUNT);

    feature_vector[write_index++] = centroid_stats.mean(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = centroid_stats.stddev(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = bandwidth_stats.mean(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = bandwidth_stats.stddev(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = rolloff_stats.mean(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = rolloff_stats.stddev(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = zcr_stats.mean(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = zcr_stats.stddev(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = rms_stats.mean(FEATURE_FRAME_COUNT);
    feature_vector[write_index++] = rms_stats.stddev(FEATURE_FRAME_COUNT);

    return write_index == FEATURE_VECTOR_SIZE;
}

void init_feature_extractor() {
    mfcc_utils::ensure_initialized();
}

bool is_chainsaw_detected(float probability, float threshold) {
    return probability >= threshold;
}
