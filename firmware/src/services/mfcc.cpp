#include "services/mfcc.h"


void extract_features_from_audio(const int16_t* audio_buffer, float* mfcc_features) {
    // 1. Calcul du nombre de trames
    // Avec 16000 échantillons, une FFT de 1024 et un saut de 256 :
    // (16000 - 1024) / 256 + 1 = 59 trames temporelles.
    const int num_frames = (16000 - AUDIO_CONFIG.fft_length) / AUDIO_CONFIG.hop_length + 1;
    
    float vReal[AUDIO_CONFIG.fft_length];
    float vImag[AUDIO_CONFIG.fft_length];
    
    // Création de l'objet FFT
    ArduinoFFT<float> FFT = ArduinoFFT<float>(vReal, vImag, AUDIO_CONFIG.fft_length, AUDIO_CONFIG.sample_rate);

    int mfcc_index = 0;

    // 2. Boucle sur chaque trame temporelle
    for (int frame = 0; frame < num_frames; frame++) {
        int start_idx = frame * AUDIO_CONFIG.hop_length;

        // Étape A : Extraction de la trame et application de la fenêtre (ex: Hanning)
        for (int i = 0; i < AUDIO_CONFIG.fft_length; i++) {
            vReal[i] = (float)audio_buffer[start_idx + i];
            vImag[i] = 0.0f;
        }
        FFT.windowing(FFT_WIN_TYP_HANN, FFT_FORWARD);

        // Étape B : Calcul de la transformée de Fourier (FFT)
        FFT.compute(FFT_FORWARD);
        FFT.complexToMagnitude(); // vReal contient maintenant le spectre de magnitude

        // Étape C : Calcul de l'énergie (Power Spectrum)
        // (vReal[i] * vReal[i]) / fft_length
        
        // Étape D : Banc de filtres Mel (Mel Filterbank)
        // Tu dois multiplier ton spectre par une matrice de 32 filtres triangulaires
        // que tu auras pré-calculée en Python et exportée en C++.
        float mel_energies[AUDIO_CONFIG.n_mels] = {0};
        // [Boucle for pour appliquer la matrice Mel]...

        // Étape E : Passage au Logarithme
        for (int i = 0; i < AUDIO_CONFIG.n_mels; i++) {
            mel_energies[i] = log10f(mel_energies[i] + 1e-6); // Ajout d'un epsilon pour éviter log(0)
        }

        // Étape F : Transformée en Cosinus Discrète (DCT)
        // Compression des 32 bandes Mel en 20 coefficients MFCC
        for (int i = 0; i < AUDIO_CONFIG.n_mfcc; i++) {
            float sum = 0.0f;
            // [Boucle for pour le calcul matriciel de la DCT]...
            
            mfcc_features[mfcc_index] = sum;
            mfcc_index++;
        }
    }
}