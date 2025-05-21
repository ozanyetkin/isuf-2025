import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Read all .shp files recursively
shp_paths = list(Path("data/Selected Cities").rglob("*.shp"))
gdfs = [gpd.read_file(fp) for fp in shp_paths]
df = pd.concat(gdfs, ignore_index=True)

# 2. Lower-case columns, drop missing target, reproject to metric CRS
df.columns = df.columns.str.lower()
df = df.dropna(subset=["building_t"]).to_crs(epsg=3857)

# 3. Compute geometry‐derived features:
#    - area, perimeter (axis‐aligned)
#    - bbox_x/bbox_y (axis‐aligned)
#    - rbox_x/rbox_y (minimum rotated rectangle)
df["area"] = df.geometry.area
df["perimeter"] = df.geometry.length

# rotated minimum‐area rectangle dims:
def rotated_dims(geom):
    # get the min‐area rectangle (Polygon)
    rbox: Polygon = geom.minimum_rotated_rectangle
    # its exterior coords: [p0, p1, p2, p3, p0]
    xs, ys = zip(*list(rbox.exterior.coords)[:4])
    pts = np.column_stack([xs, ys])
    # edge lengths
    edge1 = np.linalg.norm(pts[1] - pts[0])
    edge2 = np.linalg.norm(pts[2] - pts[1])
    # return (width, height) as the smaller/larger
    return (min(edge1, edge2), max(edge1, edge2))


# apply to each row
r_dims = np.array([rotated_dims(g) for g in df.geometry])
df["rbox_x"], df["rbox_y"] = r_dims[:, 0], r_dims[:, 1]

# 4. Map into main categories
group_map = {
    **{
        t: "residential"
        for t in [
            "apartments",
            "house",
            "detached",
            "semidetached_house",
            "bungalow",
            "terrace",
            "allotment_house",
        ]
    },
    **{t: "industrial" for t in ["industrial", "warehouse"]},
    **{
        t: "commercial"
        for t in [
            "retail",
            "office",
            "supermarket",
            "kiosk",
            "garage",
            "garages",
            "parking",
        ]
    },
    **{
        t: "public"
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
        ]
    },
    **{t: "infrastructure" for t in ["bridge", "viaduct", "railway", "transportation"]},
}
df["building_main"] = df["building_t"].map(group_map).fillna("other")

# 5. Feature list now includes the rotated dims
feature_cols = [
    "compactnes",
    "global_int",
    "local_inte",
    "building_c",
    "area",
    "perimeter",
    "rbox_x",
    "rbox_y",
]

# 6. Coerce & drop any NaNs
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=feature_cols + ["building_main"])

# 7. Encode, split, train
le = LabelEncoder()
y = le.fit_transform(df["building_main"])
X = df[feature_cols].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 8. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 9. Feature importances
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)
print("\nFeature Importances:")
print(importances.to_string())
