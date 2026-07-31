from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import \
    compute_class_weight as skl_compute_class_weight

from ai_model import AIModel
from extract_features import (DATASET_ROOT, DEFAULT_MANIFEST_PATH,
                              DEFAULT_OUTPUT_PATH, FeatureConfig,
                              build_feature_dataset, save_feature_dataset)
from train import *

REPORT_DIR = Path(__file__).resolve().parent / "model" / "report"
TFLITE_MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.tflite"
KERAS_MODEL_PATH = Path(__file__).resolve().parent / "model" / "model.h5"



def next_index(report_dir: Path, prefix: str = "tflite_report") -> int:
	existing_reports = list(report_dir.glob(f"{prefix}_*.json"))
	if not existing_reports:
		return 0

	indices: list[int] = []
	for report in existing_reports:
		match = re.search(rf"{re.escape(prefix)}_(\d+)\.json", report.name)
		if match is not None:
			indices.append(int(match.group(1)))

	return max(indices) + 1 if indices else 0


def latest_training_report(report_dir: Path) -> Path | None:
	reports = sorted(report_dir.glob("training_report_*.json"))
	return reports[-1] if reports else None


def load_or_build_feature_cache(
	dataset_root: Path,
	cache_path: Path,
	manifest_path: Path,
	*,
	force_extract: bool = False,
	no_extract: bool = False,
) -> tuple[np.ndarray, np.ndarray, FeatureConfig]:
	if cache_path.exists() and manifest_path.exists() and not force_extract:
		try:
			cached = np.load(cache_path)
			if "features" in cached and "labels" in cached:
				features = cached["features"]
				labels = cached["labels"]
				manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
				manifest_samples = int(manifest.get("samples", -1))
				manifest_dim = int(manifest.get("feature_dim", -1))
				features_shape0 = int(labels.size)
				features_dim = int(features.shape[1]) if features.ndim > 1 else 1
				if manifest_samples == features_shape0 and manifest_dim == features_dim:
					print(f"Using cached features ({cache_path}) and manifest ({manifest_path}).")
					return features, labels, FeatureConfig(**manifest["config"])
				print("Cache/manifest mismatch: rebuilding features")
			else:
				print("Cache file missing expected arrays: rebuilding features")
		except Exception as exc:
			print(f"Failed to load cache ({cache_path}): {exc}; rebuilding features")

	if no_extract:
		raise RuntimeError("Feature cache or manifest invalid or missing and --no-extract was requested.")

	config = FeatureConfig()
	features, labels, manifest = build_feature_dataset(dataset_root, config)
	save_feature_dataset(cache_path, manifest_path, features, labels, config, manifest)
	print(f"Features extracted and cached to {cache_path}")
	return features, labels, config


def load_sample_groups(manifest_path: Path, sample_count: int) -> np.ndarray:
	manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
	files = manifest.get("files", [])
	groups: list[str] = []

	for entry in files:
		windows = int(entry.get("windows", 1))
		group = str(entry.get("group") or entry.get("file") or "")
		groups.extend([group] * windows)

	if len(groups) != sample_count:
		raise ValueError(
			f"Manifest groups do not match sample count: groups={len(groups)} samples={sample_count}. Rebuild the feature cache with --force-extract."
		)

	return np.asarray(groups, dtype=object)


def stratified_group_split(
	features: np.ndarray,
	labels: np.ndarray,
	groups: np.ndarray,
	train_ratio: float = 0.7,
	validation_ratio: float = 0.15,
	seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	del train_ratio, validation_ratio

	if features.shape[0] != labels.shape[0]:
		raise ValueError("Features and labels must have the same number of samples.")
	if features.shape[0] != groups.shape[0]:
		raise ValueError("Features and groups must have the same number of samples.")

	unique_groups, first_indices = np.unique(groups, return_index=True)
	group_labels = labels[first_indices]
	group_indices = np.arange(unique_groups.shape[0])

	if unique_groups.shape[0] < 5:
		raise ValueError("Not enough distinct audio files to perform a 5-fold stratified group split.")

	sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
	fold_groups: list[np.ndarray] = []
	for _, fold_group_idx in sgkf.split(group_indices, group_labels, groups=unique_groups):
		fold_groups.append(unique_groups[fold_group_idx])

	if len(fold_groups) != 5:
		raise RuntimeError("Unexpected number of folds returned by StratifiedGroupKFold.")

	train_groups = np.concatenate(fold_groups[:3])
	val_groups = fold_groups[3]
	test_groups = fold_groups[4]

	train_mask = np.isin(groups, train_groups)
	val_mask = np.isin(groups, val_groups)
	test_mask = np.isin(groups, test_groups)

	if not np.all(train_mask | val_mask | test_mask):
		raise RuntimeError("Some samples were not assigned to any split.")
	if np.any(train_mask & val_mask) or np.any(train_mask & test_mask) or np.any(val_mask & test_mask):
		raise RuntimeError("Group split produced overlapping samples.")

	x_train, y_train = features[train_mask], labels[train_mask]
	x_val, y_val = features[val_mask], labels[val_mask]
	x_test, y_test = features[test_mask], labels[test_mask]

	return x_train, y_train, x_val, y_val, x_test, y_test


def compute_class_weight(labels: np.ndarray) -> dict[int, float]:
	classes = np.unique(labels)
	weights = skl_compute_class_weight(class_weight="balanced", classes=classes, y=labels)
	return {int(c): float(w) for c, w in zip(classes, weights)}


def serialize_metrics(metrics: dict[str, float | int | np.generic | np.ndarray]) -> dict[str, float | int | None]:
	serialized: dict[str, float | int | None] = {}
	for key, value in metrics.items():
		if value is None:
			serialized[key] = None
		elif isinstance(value, (np.integer, int)):
			serialized[key] = int(value)
		elif isinstance(value, (np.floating, float)):
			serialized[key] = float(value)
		else:
			serialized[key] = float(np.asarray(value).reshape(-1)[0])
	return serialized


def compute_binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, *, threshold: float = 0.5) -> dict[str, float | int | None]:
	y_true = np.asarray(y_true, dtype=np.int32).reshape(-1)
	probabilities = np.asarray(probabilities, dtype=np.float32).reshape(-1)

	if y_true.shape[0] != probabilities.shape[0]:
		raise ValueError("y_true and probabilities must have the same length.")

	predictions = (probabilities >= threshold).astype(np.int32)

	tp = int(np.sum((predictions == 1) & (y_true == 1)))
	tn = int(np.sum((predictions == 0) & (y_true == 0)))
	fp = int(np.sum((predictions == 1) & (y_true == 0)))
	fn = int(np.sum((predictions == 0) & (y_true == 1)))

	total = int(y_true.size)
	accuracy = float((tp + tn) / total) if total else None
	precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
	recall = float(tp / (tp + fn)) if (tp + fn) else 0.0

	auc: float | None
	if np.unique(y_true).size < 2:
		auc = None
	else:
		auc_metric = tf.keras.metrics.AUC(name="auc")
		auc_metric.update_state(y_true.astype(np.float32), probabilities.astype(np.float32))
		auc = float(auc_metric.result().numpy())

	loss_fn = tf.keras.losses.BinaryCrossentropy()
	loss = float(loss_fn(y_true.astype(np.float32), probabilities.astype(np.float32)).numpy())

	return {
		"loss": loss,
		"accuracy": accuracy,
		"precision": precision,
		"recall": recall,
		"auc": auc,
		"true_positives": tp,
		"true_negatives": tn,
		"false_positives": fp,
		"false_negatives": fn,
	}


def predict_tflite_probabilities(interpreter: tf.lite.Interpreter, features: np.ndarray) -> np.ndarray:
	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]

	input_index = int(input_details["index"])
	output_index = int(output_details["index"])
	input_dtype = input_details["dtype"]
	output_dtype = output_details["dtype"]
	input_scale, input_zero_point = input_details.get("quantization", (0.0, 0))
	output_scale, output_zero_point = output_details.get("quantization", (0.0, 0))

	probabilities: list[float] = []
	for sample in np.asarray(features, dtype=np.float32):
		sample_batch = sample.reshape(1, -1)

		if input_dtype == np.float32:
			input_data = sample_batch.astype(np.float32)
		else:
			if input_scale == 0:
				input_data = sample_batch.astype(input_dtype)
			else:
				input_data = np.round(sample_batch / input_scale + input_zero_point).astype(input_dtype)

		interpreter.set_tensor(input_index, input_data)
		interpreter.invoke()

		output_data = interpreter.get_tensor(output_index)
		if output_dtype == np.float32:
			output_value = np.asarray(output_data, dtype=np.float32)
		elif output_scale != 0:
			output_value = (np.asarray(output_data, dtype=np.float32) - output_zero_point) * output_scale
		else:
			output_value = np.asarray(output_data, dtype=np.float32)

		probabilities.append(float(output_value.reshape(-1)[0]))

	return np.asarray(probabilities, dtype=np.float32)


def compare_metrics(reference: dict[str, float | int | None], candidate: dict[str, float | int | None]) -> dict[str, float | None]:
	keys = ["loss", "accuracy", "precision", "recall", "auc"]
	comparison: dict[str, float | None] = {}
	for key in keys:
		ref_value = reference.get(key)
		cand_value = candidate.get(key)
		if ref_value is None or cand_value is None:
			comparison[f"{key}_delta"] = None
		else:
			comparison[f"{key}_delta"] = float(cand_value) - float(ref_value)
	return comparison


def summarize_predictions(reference_probabilities: np.ndarray, candidate_probabilities: np.ndarray, *, threshold: float = 0.5) -> dict[str, float | int]:
	reference_probabilities = np.asarray(reference_probabilities, dtype=np.float32).reshape(-1)
	candidate_probabilities = np.asarray(candidate_probabilities, dtype=np.float32).reshape(-1)

	if reference_probabilities.shape != candidate_probabilities.shape:
		raise ValueError("Probability arrays must have the same shape.")

	absolute_diff = np.abs(reference_probabilities - candidate_probabilities)
	reference_labels = (reference_probabilities >= threshold).astype(np.int32)
	candidate_labels = (candidate_probabilities >= threshold).astype(np.int32)

	return {
		"mean_abs_probability_diff": float(np.mean(absolute_diff)),
		"max_abs_probability_diff": float(np.max(absolute_diff)),
		"prediction_mismatch_count": int(np.sum(reference_labels != candidate_labels)),
	}


def main() -> int:
	parser = argparse.ArgumentParser(description="Validate the TFLite model against the Keras model on the project splits.")
	parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Path to the WAV dataset root")
	parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the extracted feature cache")
	parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to the feature manifest")
	parser.add_argument("--keras-model", type=Path, default=KERAS_MODEL_PATH, help="Path to the trained Keras model")
	parser.add_argument("--tflite-model", type=Path, default=TFLITE_MODEL_PATH, help="Path to the converted TFLite model")
	parser.add_argument("--report-dir", type=Path, default=REPORT_DIR, help="Directory where the validation report will be saved")
	parser.add_argument("--no-extract", action="store_true", help="Do not extract features if the cache is missing or invalid")
	parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of features even if cache/manifest exist")
	parser.add_argument("--seed", type=int, default=42, help="Seed used for the deterministic split")
	args = parser.parse_args()

	if not args.keras_model.exists():
		raise FileNotFoundError(f"Keras model not found: {args.keras_model}")
	if not args.tflite_model.exists():
		raise FileNotFoundError(f"TFLite model not found: {args.tflite_model}")

	features, labels, config = load_or_build_feature_cache(
		args.dataset,
		args.cache,
		args.manifest,
		force_extract=args.force_extract,
		no_extract=args.no_extract,
	)
	groups = load_sample_groups(args.manifest, labels.size)
	x_train, y_train, x_validation, y_validation, x_test, y_test = stratified_group_split(
		features,
		labels,
		groups,
		seed=args.seed,
	)
	keras_model = AIModel()
	keras_model.load_model(args.keras_model)

	interpreter = tf.lite.Interpreter(model_path=str(args.tflite_model))
	interpreter.allocate_tensors()

	keras_validation_probs = keras_model.predict(x_validation, threshold=None)
	keras_test_probs = keras_model.predict(x_test, threshold=None)
	tflite_validation_probs = predict_tflite_probabilities(interpreter, x_validation)
	tflite_test_probs = predict_tflite_probabilities(interpreter, x_test)

	keras_validation_metrics = serialize_metrics(keras_model.evaluate(x_validation, y_validation))
	keras_test_metrics = serialize_metrics(keras_model.evaluate(x_test, y_test))
	tflite_validation_metrics = compute_binary_metrics(y_validation, tflite_validation_probs)
	tflite_test_metrics = compute_binary_metrics(y_test, tflite_test_probs)

	latest_report = latest_training_report(args.report_dir)
	report_payload = {
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"feature_config": asdict(config),
		"dataset_size": int(labels.size),
		"train_size": int(y_train.size),
		"validation_size": int(y_validation.size),
		"test_size": int(y_test.size),
		"keras_model_path": str(args.keras_model),
		"tflite_model_path": str(args.tflite_model),
		"reference_training_report": str(latest_report) if latest_report is not None else None,
		"keras_validation_metrics": keras_validation_metrics,
		"keras_test_metrics": keras_test_metrics,
		"tflite_validation_metrics": serialize_metrics(tflite_validation_metrics),
		"tflite_test_metrics": serialize_metrics(tflite_test_metrics),
		"metric_deltas": {
			"validation": compare_metrics(keras_validation_metrics, tflite_validation_metrics),
			"test": compare_metrics(keras_test_metrics, tflite_test_metrics),
		},
		"prediction_deltas": {
			"validation": summarize_predictions(keras_validation_probs, tflite_validation_probs),
			"test": summarize_predictions(keras_test_probs, tflite_test_probs),
		},
	}

	(args.report_dir / "tflite").mkdir(parents=True, exist_ok=True)
	report_path = args.report_dir / "tflite" / f"tflite_report_{next_index(args.report_dir):02d}.json"
	report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print("Label distribution after split:")
	print(" - train:", dict(Counter(y_train)))
	print(" - validation:", dict(Counter(y_validation)))
	print(" - test:", dict(Counter(y_test)))
	print("Class weight on train split:", compute_class_weight(y_train))
	print(f"TFLite validation report saved to {report_path}")
	print("Keras validation:", keras_validation_metrics)
	print("TFLite validation:", serialize_metrics(tflite_validation_metrics))
	print("Keras test:", keras_test_metrics)
	print("TFLite test:", serialize_metrics(tflite_test_metrics))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
