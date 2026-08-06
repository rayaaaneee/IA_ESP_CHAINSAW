#include "services/mfcc_utils.h"

namespace mfcc_utils {

    float g_delta_coeffs[kDeltaWindow];
    float g_delta2_coeffs[kDeltaWindow];
    bool g_initialized = false;

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
        } else if (derivative_order == 2) {
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
            return;
        }
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

    void ensure_initialized() {
        if (g_initialized) {
            return;
        }

        build_delta_coefficients(1, g_delta_coeffs);
        build_delta_coefficients(2, g_delta2_coeffs);
        g_initialized = true;
    }

}  // namespace mfcc_utils
