#include "services/mfcc.h"
#include "services/mfcc_utils.h"

bool extract_features_from_audio(const int16_t* audio_buffer, float* feature_vector, size_t feature_vector_size) {
    if (audio_buffer == nullptr || feature_vector == nullptr || feature_vector_size < FEATURE_VECTOR_SIZE) {
        return false;
    }

    init_feature_extractor();

    constexpr int signal_length = static_cast<int>(mfcc_utils::kAudioWindowSamples);
    constexpr int fft_length = static_cast<int>(mfcc_utils::kFftLength);
    constexpr int spectral_bins = static_cast<int>(mfcc_utils::kFftBins);
    constexpr int frame_count = static_cast<int>(FEATURE_FRAME_COUNT);

    float peak = 0.0f;
    for (size_t index = 0; index < mfcc_utils::kAudioWindowSamples; ++index) {
        mfcc_utils::g_signal[index] = static_cast<float>(audio_buffer[index]) / 32768.0f;
        peak = fmaxf(peak, fabsf(mfcc_utils::g_signal[index]));
    }

    if (peak > 0.0f) {
        const float inverse_peak = 1.0f / peak;
        for (size_t index = 0; index < mfcc_utils::kAudioWindowSamples; ++index) {
            mfcc_utils::g_signal[index] *= inverse_peak;
        }
    }

    float global_mel_max = 0.0f;

    for (int frame = 0; frame < frame_count; ++frame) {
        const int spectral_center = frame * static_cast<int>(AUDIO_CONFIG.hop_length);
        const int spectral_start = spectral_center - (fft_length / 2);

        for (int sample = 0; sample < fft_length; ++sample) {
            const int source_index = spectral_start + sample;
            mfcc_utils::g_frame_real[sample] = mfcc_utils::sample_reflect(mfcc_utils::g_signal, signal_length, source_index) * mfcc_utils::hann_window(sample, fft_length);
            mfcc_utils::g_frame_imag[sample] = 0.0f;
        }

        ArduinoFFT<float> fft(mfcc_utils::g_frame_real, mfcc_utils::g_frame_imag, mfcc_utils::kFftLength, AUDIO_CONFIG.sample_rate);
        fft.compute(FFT_FORWARD);
        fft.complexToMagnitude();

        double magnitude_sum = 0.0;
        double weighted_sum = 0.0;
        double weighted_square_sum = 0.0;

        for (int bin = 0; bin < spectral_bins; ++bin) {
            const float mag = mfcc_utils::g_frame_real[bin];
            mfcc_utils::g_magnitude[bin] = mag;
            const double frequency = static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length);
            magnitude_sum += mag;
            weighted_sum += frequency * mag;
        }

        const float centroid_value = magnitude_sum > 0.0 ? static_cast<float>(weighted_sum / magnitude_sum) : 0.0f;
        mfcc_utils::g_centroid[frame] = centroid_value;

        for (int bin = 0; bin < spectral_bins; ++bin) {
            const double frequency = static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length);
            const double distance = frequency - static_cast<double>(centroid_value);
            weighted_square_sum += distance * distance * mfcc_utils::g_magnitude[bin];
        }

        mfcc_utils::g_bandwidth[frame] = magnitude_sum > 0.0 ? static_cast<float>(sqrt(weighted_square_sum / magnitude_sum)) : 0.0f;

        const double rolloff_threshold = magnitude_sum * mfcc_utils::kRollPercent;
        double cumulative = 0.0;
        float rolloff_value = 0.0f;
        for (int bin = 0; bin < spectral_bins; ++bin) {
            cumulative += mfcc_utils::g_magnitude[bin];
            if (cumulative >= rolloff_threshold) {
                rolloff_value = static_cast<float>(static_cast<double>(bin) * static_cast<double>(AUDIO_CONFIG.sample_rate) / static_cast<double>(fft_length));
                break;
            }
        }
        mfcc_utils::g_rolloff[frame] = rolloff_value;

        const int rms_start = frame * static_cast<int>(AUDIO_CONFIG.hop_length) - mfcc_utils::kRmsPad;
        double zero_crossings = 0.0;
        double rms_sum = 0.0;
        float previous_sample = mfcc_utils::sample_reflect(mfcc_utils::g_signal, signal_length, rms_start);

        for (int sample = 0; sample < mfcc_utils::kRmsFrameLength; ++sample) {
            const float current_sample = mfcc_utils::sample_reflect(mfcc_utils::g_signal, signal_length, rms_start + sample);
            if ((previous_sample >= 0.0f && current_sample < 0.0f) || (previous_sample < 0.0f && current_sample >= 0.0f)) {
                zero_crossings += 1.0;
            }
            rms_sum += static_cast<double>(current_sample) * static_cast<double>(current_sample);
            previous_sample = current_sample;
        }

        mfcc_utils::g_zcr[frame] = static_cast<float>(zero_crossings / static_cast<double>(mfcc_utils::kRmsFrameLength - 1));
        mfcc_utils::g_rms[frame] = static_cast<float>(sqrt(rms_sum / static_cast<double>(mfcc_utils::kRmsFrameLength)));

        for (int mel = 0; mel < static_cast<int>(mfcc_utils::kMelCount); ++mel) {
            double energy = 0.0;
            const float* mel_row = &MEL_MATRIX[mel * spectral_bins];
            for (int bin = 0; bin < spectral_bins; ++bin) {
                energy += static_cast<double>(mel_row[bin]) * static_cast<double>(mfcc_utils::g_magnitude[bin]) * static_cast<double>(mfcc_utils::g_magnitude[bin]);
            }
            const float mel_energy = static_cast<float>(energy);
            mfcc_utils::g_mel_energies[mel][frame] = mel_energy;
            if (mel_energy > global_mel_max) {
                global_mel_max = mel_energy;
            }
        }
    }

    if (global_mel_max < mfcc_utils::kEpsilon) {
        global_mel_max = mfcc_utils::kEpsilon;
    }

    for (int frame = 0; frame < frame_count; ++frame) {
        for (int mel = 0; mel < static_cast<int>(mfcc_utils::kMelCount); ++mel) {
            const float mel_energy = mfcc_utils::g_mel_energies[mel][frame] < mfcc_utils::kEpsilon ? mfcc_utils::kEpsilon : mfcc_utils::g_mel_energies[mel][frame];
            float db = 10.0f * log10f(mel_energy / global_mel_max);
            if (db < -mfcc_utils::kTopDb) {
                db = -mfcc_utils::kTopDb;
            }
            mfcc_utils::g_mel_db[mel][frame] = db;
        }
    }

    for (int frame = 0; frame < frame_count; ++frame) {
        for (int mfcc = 0; mfcc < static_cast<int>(mfcc_utils::kMfccCount); ++mfcc) {
            double sum = 0.0;
            for (int mel = 0; mel < static_cast<int>(mfcc_utils::kMelCount); ++mel) {
                sum += static_cast<double>(DCT_MATRIX[mfcc * mfcc_utils::kMelCount + mel]) * static_cast<double>(mfcc_utils::g_mel_db[mel][frame]);
            }
            mfcc_utils::g_mfcc_matrix[mfcc][frame] = static_cast<float>(sum);
        }
    }

    mfcc_utils::compute_delta_series(&mfcc_utils::g_mfcc_matrix[0][0], static_cast<int>(mfcc_utils::kMfccCount), frame_count, mfcc_utils::g_delta_coeffs, &mfcc_utils::g_delta_matrix[0][0]);
    mfcc_utils::compute_delta_series(&mfcc_utils::g_mfcc_matrix[0][0], static_cast<int>(mfcc_utils::kMfccCount), frame_count, mfcc_utils::g_delta2_coeffs, &mfcc_utils::g_delta2_matrix[0][0]);

    size_t write_index = 0;
    for (int mfcc = 0; mfcc < static_cast<int>(mfcc_utils::kMfccCount); ++mfcc) {
        mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_mfcc_matrix[mfcc], FEATURE_FRAME_COUNT);
    }
    for (int mfcc = 0; mfcc < static_cast<int>(mfcc_utils::kMfccCount); ++mfcc) {
        mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_delta_matrix[mfcc], FEATURE_FRAME_COUNT);
    }
    for (int mfcc = 0; mfcc < static_cast<int>(mfcc_utils::kMfccCount); ++mfcc) {
        mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_delta2_matrix[mfcc], FEATURE_FRAME_COUNT);
    }
    for (int mel = 0; mel < static_cast<int>(mfcc_utils::kMelCount); ++mel) {
        mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_mel_db[mel], FEATURE_FRAME_COUNT);
    }

    mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_centroid, FEATURE_FRAME_COUNT);
    mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_bandwidth, FEATURE_FRAME_COUNT);
    mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_rolloff, FEATURE_FRAME_COUNT);
    mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_zcr, FEATURE_FRAME_COUNT);
    mfcc_utils::append_summary(feature_vector, write_index, mfcc_utils::g_rms, FEATURE_FRAME_COUNT);

    return write_index == FEATURE_VECTOR_SIZE;
}

void init_feature_extractor() {
    mfcc_utils::ensure_initialized();
}

bool is_chainsaw_detected(float probability, float threshold) {
    return probability >= threshold;
}