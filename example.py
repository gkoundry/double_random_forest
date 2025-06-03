import numpy as np
from double_random_forest import (
    DoubleRandomForestRegressor,
    DoubleRandomForestClassifier,
)

# Create test data
X = np.random.rand(100, 10)
y_reg = np.sum(X, axis=1) + np.random.rand(100) * 0.1
y_binary = (y_reg > np.median(y_reg)).astype(int)
y_multiclass = np.digitize(y_reg, bins=np.percentile(y_reg, [25, 50, 75]))

# Train and test the Double Random Forest regressor
drf_reg = DoubleRandomForestRegressor(n_estimators=10)
drf_reg.fit(X, y_reg)
print("Double Random Forest Regressor Predictions:", drf_reg.predict(X))

# Train and test the Double Random Forest binary classifier
drf_clf_binary = DoubleRandomForestClassifier(n_estimators=10)
drf_clf_binary.fit(X, y_binary)
print("Double Random Forest Binary Classifier Predictions:", drf_clf_binary.predict(X))
print(
    "Double Random Forest Binary Classifier Probabilities:",
    drf_clf_binary.predict_proba(X),
)

# Train and test the Double Random Forest multiclass classifier
drf_clf_multiclass = DoubleRandomForestClassifier(n_estimators=10)
drf_clf_multiclass.fit(X, y_multiclass)
print(
    "Double Random Forest Multiclass Classifier Predictions:",
    drf_clf_multiclass.predict(X),
)
print(
    "Double Random Forest Multiclass Classifier Probabilities:",
    drf_clf_multiclass.predict_proba(X),
)
