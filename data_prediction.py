# data_prediction_memory_friendly.py
import warnings
import numpy as np

from pathlib import Path
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_preprocessing import load_and_preprocess, FEATURES

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 1) Load raw data
df = load_and_preprocess(Path("data/Selected Cities"), balance_method=None)
print("Cities in dataset:", df["city"].unique())

# 2) Build X, y, groups — downcast to float32 to halve your memory footprint
X = df[FEATURES].to_numpy(dtype=np.float32)
le = LabelEncoder().fit(df["building_main"])
y = le.transform(df["building_main"])
groups = df["city"].values

# 3) Split off a small dev‐set for final hold‐out metrics
X_trainval, X_holdout, y_trainval, y_holdout, grp_trainval, grp_holdout = (
    train_test_split(X, y, groups, test_size=0.2, random_state=42, stratify=y)
)

# 4) Pipeline with on‐the‐fly scaling + RF
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
        ),
    ]
)

# 5) Randomized search instead of full GridSearch (much fewer fits)
param_dist = {
    "clf__n_estimators": [100, 200, 300],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_leaf": [1, 5, 10],
}
cv = GroupKFold(n_splits=len(np.unique(grp_trainval)))
search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    scoring="f1_macro",
    cv=cv,
    verbose=1,
    n_jobs=1,
    random_state=42,
)
search.fit(X_trainval, y_trainval, groups=grp_trainval)
best_model = search.best_estimator_
print("Best params:", search.best_params_, "\n")

# 6) Evaluate on hold‐out set
y_pred = best_model.predict(X_holdout)
print("=== Overall Hold-out Metrics ===")
print("Accuracy :", accuracy_score(y_holdout, y_pred))
print("Macro-F1 :", f1_score(y_holdout, y_pred, average="macro"))
print(
    classification_report(y_holdout, y_pred, target_names=le.classes_, zero_division=0)
)
print("Confusion matrix:\n", confusion_matrix(y_holdout, y_pred))

# 7) Per‐city diagnostics (model has seen every city in trainval)
print("\n=== Per-city Accuracy on Full Data ===")
for city in np.unique(groups):
    mask = (df["city"] == city).to_numpy()
    Xi, yi = X[mask], y[mask]
    acc = accuracy_score(yi, best_model.predict(Xi))
    print(f"{city:10s} → accuracy: {acc:.4f}")
