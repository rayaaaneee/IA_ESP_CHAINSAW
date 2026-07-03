from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf

from ai_model import AIModel
from extract_features import (DATASET_ROOT, DEFAULT_MANIFEST_PATH,
                              DEFAULT_OUTPUT_PATH, FeatureConfig,
                              build_feature_dataset, save_feature_dataset)

MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.h5"
REPORT_PATH = Path(__file__).resolve().parent / "model" / "report" / "training_report.json"


def load_or_build_feature_cache(dataset_root: Path, cache_path: Path, manifest_path: Path) -> tuple[np.ndarray, np.ndarray, FeatureConfig]:
	if cache_path.exists() and manifest_path.exists():
		cached = np.load(cache_path)
		features = cached["features"]
		labels = cached["labels"]
		manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
		return features, labels, FeatureConfig(**manifest["config"])

	config = FeatureConfig()
	features, labels, manifest = build_feature_dataset(dataset_root, config)
	save_feature_dataset(cache_path, manifest_path, features, labels, config, manifest)
	return features, labels, config


def stratified_split(
	features: np.ndarray,
	labels: np.ndarray,
	train_ratio: float = 0.7,
	validation_ratio: float = 0.15,
	seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	if features.shape[0] != labels.shape[0]:
		raise ValueError("features et labels doivent avoir le même nombre d'exemples")

	rng = np.random.default_rng(seed)
	train_indices: list[int] = []
	validation_indices: list[int] = []
	test_indices: list[int] = []

	for class_value in np.unique(labels):
		class_indices = np.where(labels == class_value)[0]
		rng.shuffle(class_indices)

		class_count = len(class_indices)
		if class_count < 3:
			train_indices.extend(class_indices[:1])
			validation_indices.extend(class_indices[1:2])
			test_indices.extend(class_indices[2:])
			continue

		train_count = max(1, int(class_count * train_ratio))
		validation_count = max(1, int(class_count * validation_ratio))
		if train_count + validation_count >= class_count:
			validation_count = max(1, class_count - train_count - 1)
		test_count = class_count - train_count - validation_count

		train_indices.extend(class_indices[:train_count])
		validation_indices.extend(class_indices[train_count : train_count + validation_count])
		test_indices.extend(class_indices[train_count + validation_count : train_count + validation_count + test_count])

	def shuffled_subset(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
		ordered = np.asarray(indices, dtype=np.int32)
		rng.shuffle(ordered)
		return features[ordered], labels[ordered]

	return (*shuffled_subset(train_indices), *shuffled_subset(validation_indices), *shuffled_subset(test_indices))


def compute_class_weight(labels: np.ndarray) -> dict[int, float]:
	total = labels.size
	weight_zero = total / (2.0 * max(1, int(np.sum(labels == 0))))
	weight_one = total / (2.0 * max(1, int(np.sum(labels == 1))))
	return {0: float(weight_zero), 1: float(weight_one)}


def serialize_metrics(metrics: dict[str, float]) -> dict[str, float]:
	return {key: float(value) for key, value in metrics.items()}


def main() -> None:
	parser = argparse.ArgumentParser(description="Train the chainsaw detection model.")
	parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Path to the WAV dataset root")
	parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the extracted feature cache")
	parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to the feature manifest")
	parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to the output Keras model")
	parser.add_argument("--epochs", type=int, default=25)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	features, labels, config = load_or_build_feature_cache(args.dataset, args.cache, args.manifest)
	x_train, y_train, x_validation, y_validation, x_test, y_test = stratified_split(features, labels, seed=args.seed)

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

	print(f"Modèle entraîné et sauvegardé dans {args.model}")
	print(f"Rapport d'entraînement sauvegardé dans {REPORT_PATH}")
	print(f"Validation: {validation_metrics}")
	print(f"Test: {test_metrics}")


if __name__ == "__main__":
	main()
