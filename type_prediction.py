import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Read all .shp files recursively
shp_paths = list(Path("data/Selected Cities").rglob("*.shp"))
gdfs = [gpd.read_file(fp) for fp in shp_paths]
df = pd.concat(gdfs, ignore_index=True)

# 2. Lower-case all column names
df.columns = df.columns.str.lower()

# 3. Drop rows missing the original fine-grained type
df = df.dropna(subset=["building_t"])

# 4. **Project to a metric CRS** (so area/length are in meters).  
#    Adjust the EPSG if your data uses a different local projection.
df = df.to_crs(epsg=3857)

# 5. Compute geometry‐derived features
df["area"]      = df.geometry.area            # m²
df["perimeter"]= df.geometry.length           # m
bounds = df.geometry.bounds                    # DataFrame with minx, miny, maxx, maxy
df["bbox_width"]  = bounds["maxx"] - bounds["minx"]
df["bbox_height"] = bounds["maxy"] - bounds["miny"]

# 6. Define mapping to main categories
group_map = {
    **{t: "residential" for t in [
        "apartments", "house", "detached", "semidetached_house",
        "bungalow", "terrace", "allotment_house"
    ]},
    **{t: "industrial" for t in ["industrial", "warehouse"]},
    **{t: "commercial" for t in [
        "retail", "office", "supermarket", "kiosk", "garage", "garages", "parking"
    ]},
    **{t: "public" for t in [
        "hospital", "school", "university", "college", "kindergarten",
        "civic", "government", "fire_station", "train_station",
        "sports_centre", "sports_hall", "museum", "chapel", "church",
        "cathedral", "castle"
    ]},
    **{t: "infrastructure" for t in [
        "bridge", "viaduct", "railway", "transportation"
    ]},
}
df["building_main"] = df["building_t"].map(group_map).fillna("other")

# 7. Pick out _all_ numeric features (your originals + geometry)
feature_cols = [
    "compactnes", "global_int", "local_inte", "building_c",
    "area", "perimeter", "bbox_width", "bbox_height"
]

# 8. Coerce to numeric & drop any rows that have become NaN
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=feature_cols + ["building_main"])

# 9. Encode target, split, and train
le = LabelEncoder()
y = le.fit_transform(df["building_main"])
X = df[feature_cols].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 10. Evaluate
y_pred = clf.predict(X_test)
print("Overall Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 11. Feature importances
importances = pd.Series(clf.feature_importances_, index=feature_cols) \
                 .sort_values(ascending=False)

print("\nFeature Importances:")
for feat, imp in importances.items():
    print(f"  {feat:12s}: {imp:.4f}")
