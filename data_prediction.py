# data_prediction.py
import warnings
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_preprocessing import load_and_preprocess, FEATURES, balance_classes

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 1) Load raw data
df = load_and_preprocess(Path("data/Selected Cities"), balance_method=None)
print("Cities loaded:", df["city"].unique())

# 2) Build an overall test‐set by sampling a fraction of each class
TEST_FRAC = 0.3
# make sure we never drop entire class: require at least 1 left
test_df = df.groupby("building_main", group_keys=False).apply(
    lambda g: g.sample(frac=TEST_FRAC, random_state=42)
)
train_df = df.drop(test_df.index)

# 3) Oversample training
train_bal = balance_classes(train_df, target_col="building_main", method="oversample")

# 4) Encode
le_all = LabelEncoder().fit(train_bal["building_main"])
X_tr_all = train_bal[FEATURES].values
y_tr_all = le_all.transform(train_bal["building_main"])
X_te_all = test_df[FEATURES].values
y_te_all = le_all.transform(test_df["building_main"])

# 5) Candidate models (with simple grid‐search for RF)
rf = RandomForestClassifier(class_weight="balanced", random_state=42)
param_grid = {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
rf_search = GridSearchCV(rf, param_grid, scoring="f1_macro", cv=3, n_jobs=-1)
candidates = {
    "RandomForest": rf_search,
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    ),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

results = {}
for name, clf in candidates.items():
    clf.fit(X_tr_all, y_tr_all)
    # if GridSearchCV, grab best_estimator_
    model = getattr(clf, "best_estimator_", clf)
    pred = model.predict(X_te_all)
    results[name] = {
        "model": model,
        "accuracy": accuracy_score(y_te_all, pred),
        "f1_macro": f1_score(y_te_all, pred, average="macro", zero_division=0),
    }
    print(
        f"{name:20s} → acc: {results[name]['accuracy']:.4f}, f1: {results[name]['f1_macro']:.4f}"
    )

best_name = max(results, key=lambda k: results[k]["f1_macro"])
best_model = results[best_name]["model"]
print(f"\nSelected best: {best_name} (macro-F1={results[best_name]['f1_macro']:.4f})\n")

print("=== Overall classification report ===")
print(
    classification_report(
        y_te_all,
        best_model.predict(X_te_all),
        labels=range(len(le_all.classes_)),
        target_names=le_all.classes_,
        zero_division=0,
    )
)
print("Confusion matrix:\n", confusion_matrix(y_te_all, best_model.predict(X_te_all)))


# 6) Leave-one-city-out with the same TEST_FRAC logic
for city in df["city"].unique():
    city_df = df[df["city"] == city]
    if city_df["building_main"].nunique() < 2:
        continue

    # sample TEST_FRAC of each class in this city
    test_city = city_df.groupby("building_main", group_keys=False).apply(
        lambda g: g.sample(frac=TEST_FRAC, random_state=42)
    )
    train_city = df[df["city"] != city]
    train_city_bal = balance_classes(
        train_city, target_col="building_main", method="oversample"
    )

    # encode per‐city
    le_city = LabelEncoder().fit(train_city_bal["building_main"])
    X_tr = train_city_bal[FEATURES].values
    y_tr = le_city.transform(train_city_bal["building_main"])
    X_te = test_city[FEATURES].values
    y_te = le_city.transform(test_city["building_main"])

    # clone the best_model (works for pipelines too)
    m = clone(best_model)
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)

    print(f"\n=== City‐holdout: {city} ({len(city_df)} samples) ===")
    print(f"Test fraction per class: {TEST_FRAC:.0%}")
    print("acc:", accuracy_score(y_te, pred))
    print("f1-macro:", f1_score(y_te, pred, average="macro", zero_division=0))
    print(
        classification_report(
            y_te,
            pred,
            labels=range(len(le_city.classes_)),
            target_names=le_city.classes_,
            zero_division=0,
        )
    )
    print("Confusion matrix:\n", confusion_matrix(y_te, pred))
    # feature_importances for tree models; for LR you'd need coefs instead
    if hasattr(m, "feature_importances_"):
        fi = m.feature_importances_
        print("Feature importances:")
        for feat, imp in zip(FEATURES, fi):
            print(f"  {feat}: {imp:.4f}")
