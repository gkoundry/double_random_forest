from dataclasses import dataclass
from math import sqrt

import numpy as np

from .core import (
    build_regression_tree,
    build_multiclass_tree,
    predict_regression_tree,
    predict_multiclass_tree,
)


@dataclass
class Estimator:
    left_children: np.ndarray
    right_children: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    values: np.ndarray


class DoubleRandomForestBase:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        max_features: int | None = None,
        random_state: int = 1234,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth else -1
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_ = []
        self.classes_ = None

    def fit(self, X, y):
        if np.amax(y) > 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        if self.max_features is None:
            max_features = int(sqrt(X.shape[1]))
        elif self.max_features == "sqrt":
            max_features = int(sqrt(X.shape[1]))
        else:
            max_features = self.max_features
        for i in range(self.n_estimators):
            estimator = Estimator(
                *self.fit_function(
                    X,
                    y,
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    max_features=max_features,
                    random_state=self.random_state + i,
                )
            )
            self.estimators_.append(estimator)
        return self


class DoubleRandomForestRegressor(DoubleRandomForestBase):
    fit_function = build_regression_tree
    predict_function = predict_regression_tree

    def predict(self, X):
        predictions = [
            self.predict_function(
                X,
                estimator.left_children,
                estimator.right_children,
                estimator.features,
                estimator.thresholds,
                estimator.values,
            )
            for estimator in self.estimators_
        ]
        return np.mean(predictions, axis=0)


class DoubleRandomForestClassifier(DoubleRandomForestBase):
    fit_function = build_multiclass_tree
    predict_function = predict_multiclass_tree

    def predict(self, X):
        predictions = self.predict_proba(X)
        return np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions

    def predict_proba(self, X):
        predictions = [
            self.predict_function(
                X,
                estimator.left_children,
                estimator.right_children,
                estimator.features,
                estimator.thresholds,
                estimator.values,
            )
            for estimator in self.estimators_
        ]
        return np.mean(predictions, axis=0)
