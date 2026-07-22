from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import \
    compute_class_weight as skl_compute_class_weight

from ai_model import AIModel
from extract_features import (DATASET_ROOT, DEFAULT_MANIFEST_PATH,
                              DEFAULT_OUTPUT_PATH, FeatureConfig,
                              build_feature_dataset, save_feature_dataset)


def next_index() -> int:
    existing_reports = list(REPORT_DIR.glob("training_report_*.json"))
    if not existing_reports:
        return 0
    indices = [int(re.search(r"training_report_(\d+)\.json", report.name).group(1)) for report in existing_reports]
    return max(indices) + 1

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.h5"
REPORT_DIR = Path(__file__).resolve().parent / "model" / "report"
REPORT_PATH = REPORT_DIR / f"training_report_{next_index():02d}.json"
	
def load_or_build_feature_cache(dataset_root: Path, cache_path: Path, manifest_path: Path, *, force_extract: bool = False, no_extract: bool = False) -> tuple[np.ndarray, np.ndarray, FeatureConfig]:
	# If both cache and manifest exist, validate them before trusting the cache.
	if cache_path.exists() and manifest_path.exists() and not force_extract:
		try:
			cached = np.load(cache_path)
			if "features" in cached and "labels" in cached:
				features = cached["features"]
				labels = cached["labels"]
				manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
				# Basic consistency checks: sample counts and feature dim
				manifest_samples = int(manifest.get("samples", -1))
				manifest_dim = int(manifest.get("feature_dim", -1))
				features_shape0 = int(labels.size)
				features_dim = int(features.shape[1]) if features.ndim > 1 else 1
				if manifest_samples == features_shape0 and manifest_dim == features_dim:
					print(f"Using cached features ({cache_path}) and manifest ({manifest_path}).")
					return features, labels, FeatureConfig(**manifest["config"])
				else:
					print("Cache/manifest mismatch: rebuilding features")
					print(f" - manifest.samples={manifest_samples}, labels.size={features_shape0}")
					print(f" - manifest.feature_dim={manifest_dim}, features.shape[1]={(features.shape[1] if features.ndim>1 else None)})")
			else:
				print("Cache file missing expected arrays: rebuilding features")
		except Exception as exc:
			print(f"Failed to load cache ({cache_path}): {exc}; rebuilding features")

	# If no_extract was requested, fail early instead of rebuilding
	if no_extract:
		raise RuntimeError("Feature cache or manifest invalid or missing and --no-extract was requested.")

	# Build and save feature dataset
	config = FeatureConfig()
	features, labels, manifest = build_feature_dataset(dataset_root, config)
	save_feature_dataset(cache_path, manifest_path, features, labels, config, manifest)
	print(f"Features extracted and cached to {cache_path}")
	return features, labels, config

def stratified_split(
	features: np.ndarray,
	labels: np.ndarray,
	train_ratio: float = 0.7,
	validation_ratio: float = 0.15,
	seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if features.shape[0] != labels.shape[0]:
		raise ValueError("Features and labels must have the same number of samples.")

	# First split: train / temp (val+test)
	sss = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_ratio, random_state=seed)
	train_idx, temp_idx = next(sss.split(features, labels))

	temp_features, temp_labels = features[temp_idx], labels[temp_idx]

	# Second split inside temp: validation / test with relative size
	if (1.0 - train_ratio) <= 0:
		raise ValueError("train_ratio must be < 1.0")
	val_relative = validation_ratio / (1.0 - train_ratio)
	val_relative = min(max(val_relative, 0.0), 1.0)
	sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - val_relative, random_state=seed)
	val_idx_rel, test_idx_rel = next(sss2.split(temp_features, temp_labels))

	x_train, y_train = features[train_idx], labels[train_idx]
	x_val, y_val = temp_features[val_idx_rel], temp_labels[val_idx_rel]
	x_test, y_test = temp_features[test_idx_rel], temp_labels[test_idx_rel]

	return x_train, y_train, x_val, y_val, x_test, y_test


def compute_class_weight(labels: np.ndarray) -> dict[int, float]:
	classes = np.unique(labels)
	weights = skl_compute_class_weight(class_weight="balanced", classes=classes, y=labels)
	return {int(c): float(w) for c, w in zip(classes, weights)}


def serialize_metrics(metrics: dict[str, float]) -> dict[str, float]:
	return {key: float(value) for key, value in metrics.items()}


def main() -> None:
	parser = argparse.ArgumentParser(description="Train the chainsaw detection model.")
	parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Path to the WAV dataset root")
	parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the extracted feature cache")
	parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to the feature manifest")
	parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to the output Keras model")
	parser.add_argument("--no-extract", action="store_true", help="Do not extract features: fail if cache/manifest are missing or inconsistent")
	parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of features even if cache/manifest exist")
	parser.add_argument("--epochs", type=int, default=25)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	features, labels, config = load_or_build_feature_cache(args.dataset, args.cache, args.manifest, force_extract=args.force_extract, no_extract=args.no_extract)
	# Split dataset with stratification and then show label distributions for debug
	x_train, y_train, x_validation, y_validation, x_test, y_test = stratified_split(features, labels, seed=args.seed)

	print("Label distribution after split:")
	print(" - train:", dict(Counter(y_train)))
	print(" - validation:", dict(Counter(y_validation)))
	print(" - test:", dict(Counter(y_test)))

	for name, arr in [("train", y_train), ("validation", y_validation), ("test", y_test)]:
		if len(np.unique(arr)) < 2:
			print(f"WARNING: {name} set contains only one class: {np.unique(arr)}")

	model = AIModel(input_dim=x_train.shape[1])
	class_weight = compute_class_weight(y_train)

	callbacks = [
		tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
		tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
	]

	history = model.train(
		x_train,
		y_train,
		validation_data=(x_validation, y_validation),
		epochs=args.epochs,
		batch_size=args.batch_size,
		class_weight=class_weight,
		callbacks=callbacks,
	)

	validation_metrics = serialize_metrics(model.evaluate(x_validation, y_validation))
	test_metrics = serialize_metrics(model.evaluate(x_test, y_test))

	model.save_model(args.model)

	REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
	REPORT_PATH.write_text(
		json.dumps(
			{
				"feature_config": asdict(config),
				"dataset_size": int(labels.size),
				"train_size": int(y_train.size),
				"validation_size": int(y_validation.size),
				"test_size": int(y_test.size),
				"class_weight": class_weight,
				"history": {key: [float(value) for value in values] for key, values in history.history.items()},
				"validation_metrics": validation_metrics,
				"test_metrics": test_metrics,
				"model_path": str(args.model),
			},
			indent=2,
			ensure_ascii=False,
		),
		encoding="utf-8",
	)

	print(f"Model successfully trained and saved to {args.model}")
	print(f"Training history: {history.history}")
	print(f"Validation: {validation_metrics}")
	print(f"Test: {test_metrics}")


if __name__ == "__main__":
	main()
