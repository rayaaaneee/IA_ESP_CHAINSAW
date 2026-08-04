from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import \
    compute_class_weight as skl_compute_class_weight

from extract_features import (LABEL_RULE_VERSION, FeatureConfig,
                              build_feature_dataset, compute_dataset_signature,
                              save_feature_dataset)


def load_or_build_feature_cache(
    dataset_root: Path,
    cache_path: Path,
    manifest_path: Path,
    *,
    force_extract: bool = False,
    no_extract: bool = False,
) -> tuple[np.ndarray, np.ndarray, FeatureConfig]:
    def cache_is_valid(cached_manifest: dict[str, object], features: np.ndarray, labels: np.ndarray) -> bool:
        manifest_samples = int(cached_manifest.get("samples", -1))
        manifest_dim = int(cached_manifest.get("feature_dim", -1))
        manifest_version = int(cached_manifest.get("label_rule_version", -1))
        manifest_signature = str(cached_manifest.get("dataset_signature") or "")
        features_shape0 = int(labels.size)
        features_dim = int(features.shape[1]) if features.ndim > 1 else 1
        current_signature = compute_dataset_signature(dataset_root)

        if manifest_version != LABEL_RULE_VERSION:
            print(f"Cache label rule version mismatch: manifest={manifest_version}, expected={LABEL_RULE_VERSION}")
            return False
        if manifest_signature != current_signature:
            print("Cache dataset signature mismatch: rebuilding features")
            return False
        if manifest_samples != features_shape0 or manifest_dim != features_dim:
            print("Cache/manifest mismatch: rebuilding features")
            print(f" - manifest.samples={manifest_samples}, labels.size={features_shape0}")
            print(f" - manifest.feature_dim={manifest_dim}, features.shape[1]={features_dim}")
            return False
        return True

    if cache_path.exists() and manifest_path.exists() and not force_extract:
        try:
            cached = np.load(cache_path)
            if "features" in cached and "labels" in cached:
                features = cached["features"]
                labels = cached["labels"]
                cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if cache_is_valid(cached_manifest, features, labels):
                    print(f"Using cached features ({cache_path}) and manifest ({manifest_path}).")
                    return features, labels, FeatureConfig(**cached_manifest["config"])
            else:
                print("Cache file missing expected arrays: rebuilding features")
        except Exception as exc:
            print(f"Failed to load cache ({cache_path}): {exc}; rebuilding features")

    if no_extract:
        raise RuntimeError("Feature cache or manifest invalid or missing and --no-extract was requested.")

    config = FeatureConfig()
    features, labels, manifest = build_feature_dataset(dataset_root, config)
    save_feature_dataset(cache_path, manifest_path, features, labels, config, manifest, dataset_root=dataset_root)
    print(f"Features extracted and cached to {cache_path}")
    return features, labels, config


def load_sample_assignments(manifest_path: Path, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", [])
    groups: list[str] = []
    strata: list[str] = []

    for entry in files:
        windows = int(entry.get("windows", 1))
        group = str(entry.get("group") or entry.get("file") or "")
        subgroup = str(entry.get("subgroup") or Path(group).parent.as_posix() or Path(group).stem)
        groups.extend([group] * windows)
        strata.extend([subgroup] * windows)

    if len(groups) != sample_count or len(strata) != sample_count:
        raise ValueError(
            f"Manifest assignments do not match sample count: groups={len(groups)} strata={len(strata)} samples={sample_count}. Rebuild the feature cache with --force-extract."
        )

    return np.asarray(groups, dtype=object), np.asarray(strata, dtype=object)


def stratified_group_split(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    strata: np.ndarray,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    del train_ratio, validation_ratio

    if features.shape[0] != labels.shape[0]:
        raise ValueError("Features and labels must have the same number of samples.")
    if features.shape[0] != groups.shape[0]:
        raise ValueError("Features and groups must have the same number of samples.")
    if features.shape[0] != strata.shape[0]:
        raise ValueError("Features and strata must have the same number of samples.")

    unique_groups, first_indices = np.unique(groups, return_index=True)
    group_strata = strata[first_indices]
    group_indices = np.arange(unique_groups.shape[0])

    if unique_groups.shape[0] < 5:
        raise ValueError("Not enough distinct audio files to perform a 5-fold stratified group split.")

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_groups: list[np.ndarray] = []
    for _, fold_group_idx in sgkf.split(group_indices, group_strata, groups=unique_groups):
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
        import tensorflow as tf

        auc_metric = tf.keras.metrics.AUC(name="auc")
        auc_metric.update_state(y_true.astype(np.float32), probabilities.astype(np.float32))
        auc = float(auc_metric.result().numpy())

    import tensorflow as tf

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


def predict_tflite_probabilities(interpreter, features: np.ndarray) -> np.ndarray:
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
        if ou