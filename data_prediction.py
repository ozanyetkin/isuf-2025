# data_prediction.py
import warnings
from pathlib import Path

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from data_preprocessing import load_and_preprocess, FEATURES

# suppress warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# load & preprocess once
df = load_and_preprocess(Path("data/Selected Cities"))

print("Cities loaded:", df["city"].unique())

# overall split
y_all = LabelEncoder().fit_transform(df["building_main"])
X_all = df[FEATURES].values
X_tr_all, X_te_all, y_tr_all, y_te_all = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# candidate models
candidates = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    # "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    # "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

results = {}
for name, model in candidates.items():
    m = model
    m.fit(X_tr_all, y_tr_all)
    pred = m.predict(X_te_all)
    results[name] = {
        "model": m,
        "accuracy": accuracy_score(y_te_all, pred),
        "f1_macro": f1_score(y_te_all, pred, average="macro", zero_division=0),
    }
    print(
        f"{name:20s} → acc: {results[name]['accuracy']:.4f}, f1: {results[name]['f1_macro']:.4f}"
    )

best_name = max(results, key=lambda k: results[k]["f1_macro"])
best_model = results[best_name]["model"]
print(f"\nSelected best: {best_name} (macro-F1={results[best_name]['f1_macro']:.4f})\n")

# detailed overall report
print("=== Overall classification report ===")
print(
    classification_report(
        y_te_all,
        best_model.predict(X_te_all),
        target_names=LabelEncoder().fit(df["building_main"]).classes_,
        zero_division=0,
    )
)
print("Confusion matrix:\n", confusion_matrix(y_te_all, best_model.predict(X_te_all)))

# city-wise using best_model class
for city, sub in df.groupby("city"):
    vc = sub["building_main"].value_counts()
    valid = vc[vc >= 2].index
    sub = sub[sub["building_main"].isin(valid)]
    if sub["building_main"].nunique() < 2:
        continue

    X = sub[FEATURES].values
    y = LabelEncoder().fit_transform(sub["building_main"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    m = best_model.__class__(**best_model.get_params())
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)

    print(f"\n=== City: {city} ({len(sub)} samples) ===")
    print("Accuracy:", accuracy_score(y_te, pred))
    print("Macro-F1:", f1_score(y_te, pred, average="macro", zero_division=0))
    print(classification_report(y_te, pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_te, pred))
    feature_importances = zip(FEATURES, m.feature_importances_)
    print("Feature importances:")
    for feature, importance in feature_importances:
        print(f"  {feature}: {importance:.4f}")
