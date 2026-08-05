from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from extract_features import (DATASET_ROOT, DEFAULT_MANIFEST_PATH,
                              DEFAULT_OUTPUT_PATH)
from train import MODEL_PATH, TFLITE_MODEL_PATH, REPORT_DIR
from train.feature_pipeline import (compare_metrics, compute_binary_metrics,
                                    compute_class_weight,
                                    load_or_build_feature_cache,
                                    load_sample_assignments,
                                    predict_tflite_probabilities,
                                    serialize_metrics, stratified_group_split,
                                    summarize_predictions)



def next_index(report_dir: Path, prefix: str = "tflite_report") -> int:
	existing_reports = list((report_dir / "tflite").glob(f"{prefix}_*.json"))
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


def latest_tflite_report(report_dir: Path) -> Path | None:
	reports = sorted((report_dir / "tflite").glob("tflite_report_*.json"))
	return reports[-1] if reports else None


def tflite_report_exists(report_dir: Path, reference_training_report: str | None, threshold: float) -> bool:
	for report_path in (report_dir / "tflite").glob("tflite_report_*.json"):
		try:
			payload = json.loads(report_path.read_text(encoding="utf-8"))
		except Exception:
			continue

		if payload.get("reference_training_report") != reference_training_report:
			continue

		stored_threshold = payload.get("threshold")
		if stored_threshold is None:
			continue

		try:
			if float(stored_threshold) == float(threshold):
				return True
		except (TypeError, ValueError):
			continue

	return False


def find_tflite_report(report_dir: Path, reference_training_report: str | None, threshold: float) -> Path | None:
	for report_path in sorted((report_dir / "tflite").glob("tflite_report_*.json")):
		try:
			payload = json.loads(report_path.read_text(encoding="utf-8"))
		except Exception:
			continue

		if payload.get("reference_training_report") != reference_training_report:
			continue

		stored_threshold = payload.get("threshold")
		if stored_threshold is None:
			continue

		try:
			if float(stored_threshold) == float(threshold):
				return report_path
		except (TypeError, ValueError):
			continue

	return None


def main() -> int:
	parser = argparse.ArgumentParser(description="Validate the TFLite model against the Keras model on the project splits.")
	parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Path to the WAV dataset root")
	parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the extracted feature cache")
	parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to the feature manifest")
	parser.add_argument("--keras-model", type=Path, default=MODEL_PATH, help="Path to the trained Keras model")
	parser.add_argument("--tflite-model", type=Path, default=TFLITE_MODEL_PATH, help="Path to the converted TFLite model")
	parser.add_argument("--report-dir", type=Path, default=REPORT_DIR, help="Directory where the validation report will be saved")
	parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold used to compute discrete metrics")
	parser.add_argument("--no-extract", action="store_true", help="Do not extract features if the cache is missing or invalid")
	parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of features even if cache/manifest exist")
	parser.add_argument("--seed", type=int, default=42, help="Seed used for the deterministic split")
	args = parser.parse_args()

	if not args.keras_model.exists():
		raise FileNotFoundError(f"Keras model not found: {args.keras_model}")
	if not args.tflite_model.exists():
		raise FileNotFoundError(f"TFLite model not found: {args.tflite_model}")

	latest_report = latest_training_report(args.report_dir)
	current_reference_training_report = str(latest_report) if latest_report is not None else None
	existing_report = find_tflite_report(args.report_dir, current_reference_training_report, float(args.threshold))
	if existing_report is not None:
		print(existing_report)
		return 0

	import tensorflow as tf

	from ai_model import AIModel

	features, labels, config = load_or_build_feature_cache(
		args.dataset,
		args.cache,
		args.manifest,
		force_extract=args.force_extract,
		no_extract=args.no_extract,
	)
	groups, strata = load_sample_assignments(args.manifest, labels.size)
	x_train, y_train, x_validation, y_validation, x_test, y_test = stratified_group_split(
		features,
		labels,
		groups,
		strata,
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

	keras_validation_metrics = serialize_metrics(keras_model.evaluate(x_validation, y_validation, threshold=args.threshold))
	keras_test_metrics = serialize_metrics(keras_model.evaluate(x_test, y_test, threshold=args.threshold))
	tflite_validation_metrics = compute_binary_metrics(y_validation, tflite_validation_probs, threshold=args.threshold)
	tflite_test_metrics = compute_binary_metrics(y_test, tflite_test_probs, threshold=args.threshold)

	latest_report = latest_training_report(args.report_dir)
	current_reference_training_report = str(latest_report) if latest_report is not None else None

	report_payload = {
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"feature_config": asdict(config),
		"dataset_size": int(labels.size),
		"train_size": int(y_train.size),
		"validation_size": int(y_validation.size),
		"test_size": int(y_test.size),
		"threshold": float(args.threshold),
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

	current_threshold = float(report_payload["threshold"])
	should_save_report = not tflite_report_exists(args.report_dir, current_reference_training_report, current_threshold)
	report_path = args.report_dir / "tflite" / f"tflite_report_{next_index(args.report_dir):02d}.json"
	if should_save_report:
		(args.report_dir / "tflite").mkdir(parents=True, exist_ok=True)
		report_path.write_text(json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8")
		print(f"TFLite validation report saved to {report_path}")
	else :
		existing_report = find_tflite_report(args.report_dir, current_reference_training_report, current_threshold)
		if existing_report is not None:
			print(existing_report)
		else:
			print("Skipping report save: a TFLite report already exists for this reference_training_report and threshold.")
	print("Label distribution after split:")
	print(" - train:", dict(Counter(y_train)))
	print(" - validation:", dict(Counter(y_validation)))
	print(" - test:", dict(Counter(y_test)))
	print("Class weight on train split:", compute_class_weight(y_train))
	print(f"Detection threshold: {args.threshold}")
	if should_save_report:
		print(f"TFLite validation report saved to {report_path}")

	print("Keras validation:", keras_validation_metrics)
	print("TFLite validation:", serialize_metrics(tflite_validation_metrics))
	print("Keras test:", keras_test_metrics)
	print("TFLite test:", serialize_metrics(tflite_test_metrics))

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
