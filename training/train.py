from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tensorflow as tf

from ai_model import AIModel
from extract_features import (DATASET_ROOT, DEFAULT_MANIFEST_PATH,
                              DEFAULT_OUTPUT_PATH)
from train import (MODEL_PATH, REPORT_DIR, compute_class_weight,
                   load_or_build_feature_cache, load_sample_assignments,
                   serialize_metrics, stratified_group_split)


def next_index() -> int:
    existing_reports = list(REPORT_DIR.glob("training_report_*.json"))
    if not existing_reports:
        return 0
    indices = [int(re.search(r"training_report_(\d+)\.json", report.name).group(1)) for report in existing_reports]
    return max(indices) + 1

REPORT_PATH = REPORT_DIR / f"training_report_{next_index():02d}.json"

def main() -> None:
	parser = argparse.ArgumentParser(description="Train the chainsaw detection model.")
	parser.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="Path to the WAV dataset root")
	parser.add_argument("--cache", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to the extracted feature cache")
	parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Path to the feature manifest")
	parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to the output Keras model")
	parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold used to compute discrete metrics")
	parser.add_argument("--no-extract", action="store_true", help="Do not extract features: fail if cache/manifest are missing or inconsistent")
	parser.add_argument("--force-extract", action="store_true", help="Force re-extraction of features even if cache/manifest exist")
	parser.add_argument("--epochs", type=int, default=25)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	features, labels, config = load_or_build_feature_cache(args.dataset, args.cache, args.manifest, force_extract=args.force_extract, no_extract=args.no_extract)
	groups, strata = load_sample_assignments(args.manifest, labels.size)
	# Split dataset with stratification and then show label distributions for debug
	x_train, y_train, x_validation, y_validation, x_test, y_test = stratified_group_split(features, labels, groups, strata, seed=args.seed)

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

	validation_metrics = serialize_metrics(model.evaluate(x_validation, y_validation, threshold=args.threshold))
	test_metrics = serialize_metrics(model.evaluate(x_test, y_test, threshold=args.threshold))

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
				"threshold": float(args.threshold),
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
	print(f"Detection threshold: {args.threshold}")
	print(f"Training history: {history.history}")
	print(f"Validation: {validation_metrics}")
	print(f"Test: {test_metrics}")


if __name__ == "__main__":
	main()
