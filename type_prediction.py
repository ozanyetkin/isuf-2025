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

# 2. Normalize column names
df.columns = df.columns.str.lower()

# 3. Drop rows missing the original fine-grained type
df = df.dropna(subset=["building_t"])

# 4. Define a mapping from detailed types → main categories
group_map = {
    # residential
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
    # industrial
    **{t: "industrial" for t in ["industrial", "warehouse"]},
    # commercial / retail
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
    # public / civic
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
    # infrastructure
    **{t: "infrastructure" for t in ["bridge", "viaduct", "railway", "transportation"]},
    # everything else → other
}

# 5. Create the new target column and drop any leftover NaNs
df["building_main"] = df["building_t"].map(group_map).fillna("other")
df = df.dropna(subset=["building_main"])

# 6. Pick the numeric features and coerce them
feature_cols = ["compactnes", "global_int", "local_inte", "building_c"]
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 7. Drop rows missing any of our chosen features
df = df.dropna(subset=feature_cols)

# 8. Encode the new target
le = LabelEncoder()
y = le.fit_transform(df["building_main"])
X = df[feature_cols].values

# 9. Split (stratifying on the new main categories)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 10. Train
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 11. Evaluate
y_pred = clf.predict(X_test)
print("Overall Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report (by main category):")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 12. Feature importances
importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)

print("\nFeature Importances:")
for feat, imp in importances.items():
    print(f"  {feat:12s}: {imp:.4f}")
