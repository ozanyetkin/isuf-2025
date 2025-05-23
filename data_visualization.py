# data_visualization.py
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from data_preprocessing import load_and_preprocess, FEATURES

# ─── suppress warnings ──────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# reader-friendly labels & colors
category_display = {
    "multi-family": "Multi-family",
    "single-family": "Single-family",
    "commercial": "Commercial",
    "industrial": "Industrial",
    "public": "Public",
    "infrastructure": "Infrastructure",
    "other": "Other",
}

feature_display = {
    "compactnes": "Compactness",
    "global_int": "Global Integration",
    "local_inte": "Local Integration",
    "building_c": "Building Coverage",
    "rbox_width": "Building X-Dimension",
    "rbox_height": "Building Y-Dimension",
}

# Color mapping
color_map = {
    k: v
    for k, v in {
        "multi-family": "#1F4E79",
        "single-family": "#D55E00",
        "commercial": "#3C7A3F",
        "industrial": "#A11D21",
        "public": "#7A4A91",
        "infrastructure": "#FF5C6A",
        "other": "#6B4A3F",
    }.items()
}

light_gray = "#e0e0e0"
gray_patch = Patch(color=light_gray, label="Train set")

output_dir = Path("images")
output_dir.mkdir(exist_ok=True)

# ─── load & preprocess ─────────────────────────────────────────────────────
df = load_and_preprocess(Path("data/Selected Cities"))

# ─── per-city maps, matrices, importances ───────────────────────────────────
for city, sub in df.groupby("city"):
    vc = sub["building_main"].value_counts()
    valid = vc[vc >= 2].index
    sub = sub[sub["building_main"].isin(valid)]
    if sub["building_main"].nunique() < 2:
        continue

    idx = sub.index.to_numpy()
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=sub.loc[idx, "building_main"]
    )

    X_tr = sub.loc[idx_tr, FEATURES].values
    y_tr = LabelEncoder().fit_transform(sub.loc[idx_tr, "building_main"])
    X_te = sub.loc[idx_te, FEATURES].values
    y_true = (
        LabelEncoder()
        .fit(sub["building_main"])
        .transform(sub.loc[idx_te, "building_main"])
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_true, y_pred)

    # … (then exactly the same plotting code you already have) …

# ─── overall on full dataset ─────────────────────────────────────────────────
# split, train, compute y_pred_all, plot confusion & importances as before

print(f"Saved all plots to {output_dir.resolve()}")
