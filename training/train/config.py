from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 8_000
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    n_mfcc: int = 20
    n_mels: int = 32
    fft_length: int = 1024
    hop_length: int = 256