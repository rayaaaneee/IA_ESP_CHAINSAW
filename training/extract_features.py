from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"
LEGACY_DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
DATASET_ROOT = DEFAULT_DATA_ROOT if DEFAULT_DATA_ROOT.exists() else LEGACY_DATASET_ROOT
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "model" / "feature_dataset.npz"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "model" / "feature_dataset_manifest.json"

POSITIVE_TOKENS = ("chainsaw", "motosierra", "scie", "saw")
NEGATIVE_TOKENS = ("environment", "environmental", "lluvia", "rain", "selva", "noise", "background", "ambiente")


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    n_mfcc: int = 20
    n_mels: int = 32
    fft_length: int = 1024
    hop_length: int = 256


def infer_label(audio_path: Path) -> int:
    searchable = " ".join(part.lower() for part in audio_path.parts + (audio_path.stem,))

    if any(token in searchable for token in POSITIVE_TOKENS):
        return 1
    if any(token in searchable for token in NEGATIVE_TOKENS):
        return 0

    raise ValueError(f"Impossible d'inférer un label pour {audio_path}")


def load_audio(audio_path: Path, config: FeatureConfig) -> np.ndarray:
    signal, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    if signal.size == 0:
        raise ValueError(f"Audio vide: {audio_path}")

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak

    return signal.astype(np.float32)


def iter_windows(signal: np.ndarray, config: FeatureConfig) -> Iterable[np.ndarray]:
    window_size = int(config.sample_rate * config.window_seconds)
    hop_size = int(config.sample_rate * config.hop_seconds)

    if signal.size <= window_size:
        yield np.pad(signal, (0, window_size - signal.size))
        return

    last_start = signal.size - window_size
    starts = list(range(0, last_start + 1, hop_size))
    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        yield signal[start : start + window_size]


def summarize(feature_matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([feature_matrix.mean(axis=1), feature_matrix.std(axis=1)], axis=0)


def extract_feature_vector(signal: np.ndarray, config: FeatureConfig) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=config.sample_rate,
        n_mfcc=config.n_mfcc,
        n_fft=config.fft_length,
        hop_length=config.hop_length,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=config.sample_rate,
        n_fft=config.fft_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=20,
        fmax=config.sample_rate // 2,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    spectral_centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=config.sample_rate,
        n_fft=config.fft_length,
        hop_length=config.hop_length,
    )
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=config.sample_rate,
        n_fft=config.fft_length,
        hop_length=config.hop_length,
    )
    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=signal,
        sr=config.sample_rate,
        n_fft=config.fft_length,
        hop_length=config.hop_length,
    )
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal, hop_length=config.hop_length)
    rms = librosa.feature.rms(y=signal, hop_length=config.hop_length)

    feature_parts = [
        summarize(mfcc),
        summarize(delta),
        summarize(delta2),
        summarize(mel_db),
        summarize(spectral_centroid),
        summarize(spectral_bandwidth),
        summarize(spectral_rolloff),
        summarize(zero_crossing_rate),
        summarize(rms),
    ]
    return np.concatenate(feature_parts).astype(np.float32)


def discover_audio_files(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.rglob("*.wav") if path.is_file())


def build_feature_dataset(dataset_root: Path, config: FeatureConfig) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    audio_files = discover_audio_files(dataset_root)
    if not audio_files:
        raise FileNotFoundError(f"Aucun fichier WAV trouvé dans {dataset_root}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    manifest: list[dict[str, object]] = []

    for audio_path in audio_files:
        label = infer_label(audio_path)
        signal = load_audio(audio_path, config)
        windows = list(iter_windows(signal, config))

        for window in windows:
            features.append(extract_feature_vector(window, config))
            labels.append(label)

        manifest.append(
            {
                "file": str(audio_path.relative_to(dataset_root)),
                "label": label,
                "windows": len(windows),
            }
        )

    x_data = np.vstack(features).astype(np.float32)
    y_data = np.asarray(labels, dtype=np.int32)
    return x_data, y_data, manifest


def save_feature_dataset(output_path: Path, manifest_path: Path, x_data: np.ndarray, y_data: np.ndarray, config: FeatureConfig, manifest: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, features=x_data, labels=y_data)
    manifest_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "samples": len(y_data),
                "feature_dim": int(x_data.shape[1]),
                "files": manifest,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio features from the dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Path to the WAV dataset root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the feature cache .npz")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to the feature manifest .json")
    args = parser.parse_args()

    config = FeatureConfig()
    x_data, y_data, manifest = build_feature_dataset(args.dataset, config)
    save_feature_dataset(args.output, args.manifest, x_data, y_data, config, manifest)

    print(f"Features extraites: {x_data.shape[0]} échantillons, dimension {x_data.shape[1]}")
    print(f"Cache sauvegardé dans {args.output}")
    print(f"Manifest sauvegardé dans {args.manifest}")


if __name__ == "__main__":
    main()

