import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

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

# ─── suppress the undefined‐metric warning ──────────────────────────────────
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ─── 1. load & tag by city ─────────────────────────────────────────────────
shp_paths = list(Path("data/Selected Cities").rglob("*.shp"))
gdfs = []
for fp in shp_paths:
    city = Path(fp).parent.name
    g = gpd.read_file(fp)
    g["city"] = city
    gdfs.append(g)
df = pd.concat(gdfs, ignore_index=True)

# ─── 2. normalize, drop missing, reproject ────────────────────────────────
df.columns = df.columns.str.lower()
df = df.dropna(subset=["building_t"]).to_crs(epsg=3857)


# ─── 3. compute only rotated‐bbox dims ────────────────────────────────────
def rotated_dims(geom: Polygon):
    r = geom.minimum_rotated_rectangle
    pts = np.array(r.exterior.coords)[:4]
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return min(e1, e2), max(e1, e2)


r_dims = np.array([rotated_dims(g) for g in df.geometry])
df["rbox_width"], df["rbox_height"] = r_dims[:, 0], r_dims[:, 1]

# ─── 4. map into main categories (split residential) ──────────────────────
group_map = {}
group_map["apartments"] = "multi-family"
for t in [
    "house",
    "detached",
    "semidetached_house",
    "bungalow",
    "terrace",
    "allotment_house",
]:
    group_map[t] = "single-family"
for t in ["industrial", "warehouse"]:
    group_map[t] = "industrial"
for t in ["retail", "office", "supermarket", "kiosk", "garage", "garages", "parking"]:
    group_map[t] = "commercial"
for t in [
    "hospital",
    "school",
    "university",
    "college",
    "kindergarten",
    "civic",
    "government",
    "fire_station",
    "train_station",
    "sports_centre",
    "sports_hall",
    "museum",
    "chapel",
    "church",
    "cathedral",
    "castle",
]:
    group_map[t] = "public"
for t in ["bridge", "viaduct", "railway", "transportation"]:
    group_map[t] = "infrastructure"

df["building_main"] = df["building_t"].map(group_map).fillna("other")

# ─── 5. select & clean features ────────────────────────────────────────────
features = [
    "compactnes",
    "global_int",
    "local_inte",
    "building_c",
    "rbox_width",
    "rbox_height",
]
for c in features:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=features + ["building_main"])

# ─── 6. encode & split overall ────────────────────────────────────────────
le_global = LabelEncoder()
y_all = le_global.fit_transform(df["building_main"])
X_all = df[features].values

X_tr_all, X_te_all, y_tr_all, y_te_all = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# ─── 7. define candidate models ───────────────────────────────────────────
candidates = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

# ─── 8. evaluate each on the overall split ────────────────────────────────
results = {}
for name, model in candidates.items():
    m = model  # fresh instance
    m.fit(X_tr_all, y_tr_all)
    pred = m.predict(X_te_all)
    acc = accuracy_score(y_te_all, pred)
    f1 = f1_score(y_te_all, pred, average="macro", zero_division=0)
    results[name] = {"model": m, "accuracy": acc, "f1_macro": f1}
    print(f"{name:20s} → acc: {acc:.4f},  f1_macro: {f1:.4f}")

# ─── 9. pick best by macro‐F1 ──────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["f1_macro"])
best_model = results[best_name]["model"]
print(
    f"\n▶ Selected best overall: {best_name!r} (macro-F1={results[best_name]['f1_macro']:.4f})\n"
)

# print its full overall report
print("=== Overall classification report ===")
print(
    classification_report(
        y_te_all,
        best_model.predict(X_te_all),
        target_names=le_global.classes_,
        zero_division=0,
    )
)
print(
    "Overall confusion matrix:\n",
    confusion_matrix(y_te_all, best_model.predict(X_te_all)),
)
print("\nOverall feature importances:")
fi = pd.Series(best_model.feature_importances_, index=features).sort_values(
    ascending=False
)
print(fi.to_string())
