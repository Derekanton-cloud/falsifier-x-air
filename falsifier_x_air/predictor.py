"""Predictor interfaces; the included Ridge model is deliberately non-graph."""

from typing import Protocol

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .schema import PredictionOutput


class DelayPredictor(Protocol):
    def fit(self, features: np.ndarray, targets: np.ndarray) -> "DelayPredictor": ...
    def predict_with_uncertainty(self, features: np.ndarray) -> PredictionOutput: ...
    def feature_distance(self, features: np.ndarray) -> np.ndarray: ...


class BaselinePredictor:
    """Replaceable Ridge baseline with split-conformal-style residual intervals.

    It intentionally consumes tabular weather/capacity features only. A future graph
    predictor implements ``DelayPredictor`` without changing investigation logic.
    """

    def __init__(self, alpha: float = 1.0, interval_quantile: float = 0.95) -> None:
        self.model = Ridge(alpha=alpha)
        self.scaler = StandardScaler()
        self.interval_quantile = interval_quantile
        self.residual_quantile: float | None = None
        self.train_mean: np.ndarray | None = None
        self.train_std: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "BaselinePredictor":
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled, targets)
        self.residual_quantile = float(np.quantile(np.abs(targets - self.model.predict(scaled)), self.interval_quantile))
        self.train_mean = scaled.mean(axis=0)
        self.train_std = scaled.std(axis=0) + 1e-6
        return self

    def predict_with_uncertainty(self, features: np.ndarray) -> PredictionOutput:
        scaled = self.scaler.transform(features)
        mean = self.model.predict(scaled)
        radius = self.residual_quantile if self.residual_quantile is not None else 10.0
        return PredictionOutput(mean, mean - radius, mean + radius, np.full(len(mean), radius / 1.96))

    def predict(self, features: np.ndarray) -> PredictionOutput:
        return self.predict_with_uncertainty(features)

    def feature_distance(self, features: np.ndarray) -> np.ndarray:
        if self.train_mean is None or self.train_std is None:
            raise RuntimeError("fit must be called before feature_distance")
        scaled = self.scaler.transform(features)
        return np.sqrt(np.mean(((scaled - self.train_mean) / self.train_std) ** 2, axis=1))
