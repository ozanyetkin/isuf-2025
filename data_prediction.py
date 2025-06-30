# data_prediction_with_city_details_fixed.py
import numpy as np
import warnings

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

# 1) Load and preprocess (no balancing here)
df = load_and_preprocess(Path("data/Selected Cities"), balance_method=None)
cities = df["city"].unique()
print("Cities in dataset:", cities)

# 2) Build feature matrix and labels, downcast to float32
X = df[FEATURES].to_numpy(dtype=np.float32)
le = LabelEncoder().fit(df["building_main"])
y = le.transform(df["building_main"])
groups = df["city"].values

# 3) Split off a hold-out set (20% stratified)
X_trainval, X_holdout, y_trainval, y_holdout, grp_trainval, grp_holdout = (
    train_test_split(X, y, groups, test_size=0.2, random_state=42, stratify=y)
)

# 4) Build a pipeline: scaler + RF
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "clf",
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1),
        ),
    ]
)

# 5) Randomized search over RF hyperparams with GroupKFold (one split per city)
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
    n_jobs=8,
    random_state=42,
)
search.fit(X_trainval, y_trainval, groups=grp_trainval)
best_model = search.best_estimator_
print("Best hyperparams:", search.best_params_, "\n")

# 6) Evaluate on the hold-out set
y_pred = best_model.predict(X_holdout)
print("=== Overall Hold-out Metrics ===")
print(f"Accuracy : {accuracy_score(y_holdout, y_pred):.4f}")
print(f"Macro-F1 : {f1_score(y_holdout, y_pred, average='macro'):.4f}\n")
print(
    classification_report(y_holdout, y_pred, target_names=le.classes_, zero_division=0)
)
print("Confusion matrix:\n", confusion_matrix(y_holdout, y_pred))

# 7) Per-city detailed diagnostics (model has seen every city in training)
print("\n=== Per-City Detailed Metrics ===")
for city in cities:
    mask = (df["city"] == city).to_numpy()
    Xi, yi = X[mask], y[mask]
    pi = best_model.predict(Xi)

    # only include labels actually in this city
    present_labels = np.unique(yi)
    present_names = [le.classes_[i] for i in present_labels]

    acc = accuracy_score(yi, pi)
    f1m = f1_score(yi, pi, average="macro", zero_division=0)
    report = classification_report(
        yi, pi, labels=present_labels, target_names=present_names, zero_division=0
    )
    cm = confusion_matrix(yi, pi, labels=present_labels)

    print(f"\n--- City: {city} ({len(yi)} samples) ---")
    print(f"Accuracy: {acc:.4f}   Macro-F1: {f1m:.4f}\n")
    print(report)
    print("Confusion matrix (rows/cols =", present_names, "):")
    print(cm)
