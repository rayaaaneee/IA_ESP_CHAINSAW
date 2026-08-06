from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np

from train.config import FeatureConfig

DATASET_ROOT = Path(__file__).resolve().parent / "data" / "raw"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "data" / "processed" / "feature_dataset.npz"
DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent / "data" /  "processed" / "feature_dataset_manifest.json"

LABEL_RULE_VERSION = 3

# Keywords for automatic label inference based on dataset structure.
# Labels are resolved from paths relative to the dataset root so the workspace name never affects classification.
POSITIVE_TOKENS = ("chainsaw", "motosierra")
NEGATIVE_TOKENS = (
    "environment",
    "birds",
    "bird",
    "jaguar",
    "monkey",
    "motocross",
    "lluvia",
    "rainforest",
    "snake",
    "ambience",
    "ambient",
)

def infer_label(audio_path: Path, dataset_root: Path) -> tuple[int, str]:
    relative_path = audio_path.relative_to(dataset_root)
    relative_parts = [part.lower() for part in relative_path.parts]
    searchable = " ".join(relative_parts + [relative_path.stem.lower()])

    if any(part in POSITIVE_TOKENS for part in relative_parts):
        return 1, "path:positive"
    if any(part in NEGATIVE_TOKENS for part in relative_parts):
        return 0, "path:negative"

    if any(token in searchable for token in POSITIVE_TOKENS):
        return 1, "token:positive"
    if any(token in searchable for token in NEGATIVE_TOKENS):
        return 0, "token:negative"

    raise ValueError(
        f"Unable to infer label for {audio_path}; please update POSITIVE_TOKENS/NEGATIVE_TOKENS or dataset layout"
    )


def infer_subgroup(audio_path: Path, dataset_root: Path) -> str:
    relative_path = audio_path.relative_to(dataset_root)
    subgroup = relative_path.parent.as_posix()
    if subgroup == "." or not subgroup:
        return relative_path.stem.lower()
    return subgroup


def compute_dataset_signature(dataset_root: Path) -> str:
    digest = hashlib.sha256()
    for audio_path in discover_audio_files(dataset_root):
        relative_path = audio_path.relative_to(dataset_root).as_posix()
        stat = audio_path.stat()
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_audio(audio_path: Path, config: FeatureConfig) -> np.ndarray:
    signal, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    if signal.size == 0:
        raise ValueError(f"Empty audio file: {audio_path}")

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
        raise FileNotFoundError(f"No WAV files found in {dataset_root}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    manifest: list[dict[str, object]] = []

    for audio_path in audio_files:
        final_label, label_source = infer_label(audio_path, dataset_root)
        relative_path = audio_path.relative_to(dataset_root)
        subgroup = infer_subgroup(audio_path, dataset_root)

        signal = load_audio(audio_path, config)
        windows = list(iter_windows(signal, config))

        for window in windows:
            features.append(extract_feature_vector(window, config))
            labels.append(int(final_label))

        manifest_entry: dict[str, object] = {
            "file": str(relative_path),
            "group": relative_path.as_posix(),
            "subgroup": subgroup,
            "label": int(final_label),
            "label_source": label_source,
            "windows": len(windows),
        }

        manifest.append(manifest_entry)

    x_data = np.vstack(features).astype(np.float32)
    y_data = np.asarray(labels, dtype=np.int32)
    return x_data, y_data, manifest


def save_feature_dataset(
    output_path: Path,
    manifest_path: Path,
    x_data: np.ndarray,
    y_data: np.ndarray,
    config: FeatureConfig,
    manifest: list[dict[str, object]],
    *,
    dataset_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, features=x_data, labels=y_data)
    manifest_path.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "label_rule_version": LABEL_RULE_VERSION,
                "dataset_root": str(dataset_root),
                "dataset_signature": compute_dataset_signature(dataset_root),
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
    save_feature_dataset(args.output, args.manifest, x_data, y_data, config, manifest, dataset_root=args.dataset)

    print(f"Features extracted: {x_data.shape[0]} samples, dimension {x_data.shape[1]}")
    print(f"Cache saved to {args.output}")
    print(f"Manifest saved to {args.manifest}")


if __name__ == "__main__":
    main()