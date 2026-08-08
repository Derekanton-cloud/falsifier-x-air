from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

@dataclass
class Prediction:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

class BaselinePredictor:
    # v0.1 baseline. It intentionally excludes aircraft rotation.
    def __init__(self):
        self.model = Ridge(alpha=1.0)
        self.scaler = StandardScaler()
        self.residual_q = None
        self.train_mean = None
        self.train_std = None

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        pred = self.model.predict(Xs)
        self.residual_q = float(np.quantile(np.abs(y - pred), 0.95))
        self.train_mean = Xs.mean(axis=0)
        self.train_std = Xs.std(axis=0) + 1e-6
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        mean = self.model.predict(Xs)
        q = self.residual_q or 10.0
        return Prediction(mean, mean-q, mean+q)

    def feature_distance(self, X):
        Xs = self.scaler.transform(X)
        z = (Xs-self.train_mean)/self.train_std
        return np.sqrt((z*z).mean(axis=1))

class CrossModelAgreement:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=80, max_depth=6, random_state=7
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)
