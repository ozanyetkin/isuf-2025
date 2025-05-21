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

# 1. Read all .shp files recursively, tagging each with its city
shp_paths = list(Path("data/Selected Cities").rglob("*.shp"))
gdfs = []
for fp in shp_paths:
    city = Path(fp).parent.name
    gdf = gpd.read_file(fp)
    gdf["city"] = city
    gdfs.append(gdf)
df = pd.concat(gdfs, ignore_index=True)

# 2. Normalize column names & drop missing fine-grained type
df.columns = df.columns.str.lower()
df = df.dropna(subset=["building_t"])

# 3. Reproject to a metric CRS (so areas/lengths are in meters)
df = df.to_crs(epsg=3857)


# 4. Compute only rotated‐bbox geometry features
def rotated_dims(geom: Polygon):
    rbox = geom.minimum_rotated_rectangle
    # first four points of the rectangle
    pts = np.array(rbox.exterior.coords)[:4]
    edge1 = np.linalg.norm(pts[1] - pts[0])
    edge2 = np.linalg.norm(pts[2] - pts[1])
    # return (width, height) = (shorter, longer)
    return min(edge1, edge2), max(edge1, edge2)


df["area"] = df.geometry.area
df["perimeter"] = df.geometry.length
r_dims = np.array([rotated_dims(g) for g in df.geometry])
df["rbox_width"], df["rbox_height"] = r_dims[:, 0], r_dims[:, 1]

# 5. Map fine‐grained types into main categories
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

# 6. Define feature columns (no axis-aligned bbox!)
feature_cols = [
    "compactnes",
    "global_int",
    "local_inte",
    "building_c",
    "area",
    "perimeter",
    "rbox_width",
    "rbox_height",
]

# 7. Coerce to numeric and drop any rows with NaNs in features or target
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=feature_cols + ["building_main"])

# 8. For each city: train/test, fit RF, and print metrics + feature importances
cities = df["city"].unique()
for city in cities:
    df_city = df[df["city"] == city]
    # skip cities with too few samples or too few classes
    if df_city.shape[0] < 20 or df_city["building_main"].nunique() < 2:
        continue

    # filter out classes with fewer than 2 samples in this city
    counts = df_city["building_main"].value_counts()
    valid = counts[counts >= 2].index
    df_city = df_city[df_city["building_main"].isin(valid)]

    X = df_city[feature_cols].values
    le = LabelEncoder()
    y = le.fit_transform(df_city["building_main"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"\n=== City: {city} ===")
    print(f"Samples: {df_city.shape[0]}, Classes: {len(le.classes_)}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(
        classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    )

    # feature importance
    imps = pd.Series(clf.feature_importances_, index=feature_cols)
    print("Feature Importance:")
    for feat, imp in imps.sort_values(ascending=False).items():
        print(f"  {feat:12s}: {imp:.4f}")
