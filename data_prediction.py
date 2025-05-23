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
gdfs = []
base = Path("data/Selected Cities")

for fp in base.rglob("*.shp"):
    # if the .shp is directly in 'Selected Cities', use the filename;
    # otherwise use its parent folder name
    if fp.parent == base:
        city = fp.stem
    else:
        city = fp.parent.name

    gdf = gpd.read_file(fp)
    gdf["city"] = city
    gdfs.append(gdf)

df = pd.concat(gdfs, ignore_index=True)
print("Unique cities loaded:", df["city"].unique())


df = pd.concat(gdfs, ignore_index=True)

# ─── 2. normalize, drop missing, reproject ────────────────────────────────
df.columns = df.columns.str.lower()
df = df.dropna(subset=["building_t"]).to_crs(epsg=3857)

print("Unique cities loaded:", df["city"].unique())


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
    "residential",
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
for t in ["retail", "office", "supermarket", "kiosk"]:
    group_map[t] = "commercial"
for t in [
    "public",
    "commercial",
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
for t in ["bridge", "viaduct", "railway", "transportation", "parking"]:
    group_map[t] = "infrastructure"

df["building_main"] = df["building_t"].map(group_map).fillna("other")

# Print the names of "other" building categories
other_categories = df.loc[df["building_main"] == "other", "building_t"].unique()
print("Building categories classified as 'other':", other_categories)

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

print("Total number of cities:", len(df["city"].unique()))
# ─── 10. city-wise training with the chosen model ─────────────────────────
for city, sub in df.groupby("city"):
    # drop rare labels (per-city) to allow stratify
    counts = sub["building_main"].value_counts()
    valid = counts[counts >= 2].index
    sub = sub[sub["building_main"].isin(valid)]
    if sub["building_main"].nunique() < 2:
        continue  # not enough diversity to train

    X = sub[features].values
    y = LabelEncoder().fit_transform(sub["building_main"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    m = best_model.__class__(**best_model.get_params())  # fresh instance
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)

    print(f"\n=== City: {city} ({len(sub)} samples) ===")
    print("Accuracy:", accuracy_score(y_te, pred))
    print("Macro-F1:", f1_score(y_te, pred, average="macro", zero_division=0))
    print("\nClassification Report:")
    print(
        classification_report(
            y_te,
            pred,
            target_names=LabelEncoder()
            .fit(sub["building_main"])
            .inverse_transform(np.unique(y_te)),
            zero_division=0,
        )
    )
    print("Confusion Matrix:\n", confusion_matrix(y_te, pred))

    fi_city = pd.Series(m.feature_importances_, index=features).sort_values(
        ascending=False
    )
    print("\nFeature Importances:")
    print(fi_city.to_string())
