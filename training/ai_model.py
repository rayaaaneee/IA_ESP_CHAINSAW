from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import tensorflow as tf


class AIModel:
    
    def __init__(self, input_dim: int | None = None, learning_rate: float = 1e-3, dropout: float = 0.3):
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.model = self._build_model(input_dim) if input_dim is not None else None

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(input_dim,), name="audio_features")
        normalizer = tf.keras.layers.Normalization(name="feature_normalization")
        x = normalizer(inputs)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout)(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="chainsaw_probability")(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="chainsaw_classifier")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
        return model

    def preprocess_data(self, data: Any) -> np.ndarray:
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"Data shape {array.shape} is not supported. Expected 1D or 2D array.")
        return array

    def train(
        self,
        data: Any,
        labels: Any,
        validation_data: tuple[Any, Any] | None = None,
        epochs: int = 25,
        batch_size: int = 32,
        class_weight: dict[int, float] | None = None,
        callbacks: Iterable[tf.keras.callbacks.Callback] | None = None,
        verbose: int = 1,
    ) -> tf.keras.callbacks.History:
        x_train = self.preprocess_data(data)
        y_train = np.asarray(labels, dtype=np.float32).reshape(-1)

        if self.model is None:
            self.input_dim = x_train.shape[1]
            self.model = self._build_model(self.input_dim)

        normalization_layer = self.model.get_layer("feature_normalization")
        normalization_layer.adapt(x_train)

        fit_kwargs: dict[str, Any] = {
            "x": x_train,
            "y": y_train,
            "epochs": epochs,
            "batch_size": batch_size,
            "class_weight": class_weight,
            "verbose": verbose,
        }

        if validation_data is not None:
            x_val, y_val = validation_data
            fit_kwargs["validation_data"] = (self.preprocess_data(x_val), np.asarray(y_val, dtype=np.float32).reshape(-1))

        if callbacks is not None:
            fit_kwargs["callbacks"] = list(callbacks)

        return self.model.fit(**fit_kwargs)

    def predict(self, data: Any, threshold: float = 0.5) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model isn't initialized.")
        x_data = self.preprocess_data(data)
        probabilities = self.model.predict(x_data, verbose=0).reshape(-1)
        if threshold is None:
            return probabilities
        return (probabilities >= threshold).astype(np.int32)

    def evaluate(self, data: Any, labels: Any, verbose: int = 0) -> dict[str, float]:
        if self.model is None:
            raise RuntimeError("Model isn't initialized.")

        x_eval = self.preprocess_data(data)
        y_eval = np.asarray(labels, dtype=np.float32).reshape(-1)
        scores = self.model.evaluate(x_eval, y_eval, verbose=verbose, return_dict=True)

        probabilities = self.model.predict(x_eval, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype(np.int32)
        truth = y_eval.astype(np.int32)

        tp = int(np.sum((predictions == 1) & (truth == 1)))
        tn = int(np.sum((predictions == 0) & (truth == 0)))
        fp = int(np.sum((predictions == 1) & (truth == 0)))
        fn = int(np.sum((predictions == 0) & (truth == 1)))

        scores.update(
            {
                "true_positives": float(tp),
                "true_negatives": float(tn),
                "false_positives": float(fp),
                "false_negatives": float(fn),
            }
        )

        return scores

    def save_model(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save. Please train or load a model before saving.")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(destination)

    def load_model(self, path: str | Path) -> None:
        self.model = tf.keras.models.load_model(path)
        self.input_dim = int(self.model.input_shape[-1])

    def postprocess_data(self, data: Any, threshold: float = 0.5) -> np.ndarray:
        probabilities = np.asarray(data, dtype=np.float32).reshape(-1)
        return (probabilities >= threshold).astype(np.int32)

    # def visualize_results(self, results: Any) -> None:
    #     raise NotImplementedError("La visualisation n'est pas encore implémentée.")

    # def optimize_hyperparameters(self, data: Any) -> None:
    #     raise NotImplementedError("L'optimisation d'hyperparamètres n'est pas encore implémentée.")

    # def explain_predictions(self, data: Any) -> None:
    #     raise NotImplementedError("L'explication des prédictions n'est pas encore implémentée.")