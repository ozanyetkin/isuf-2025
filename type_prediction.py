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

# 3. Drop rows missing the target
df = df.dropna(subset=["building_t"])

# 4. Explicitly pick the numeric features you care about
feature_cols = ["compactnes", "global_int", "local_inte", "building_c"]

# 5. Coerce those columns to floats (invalid parsing → NaN)
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 6. Drop any rows where we now lost a feature
df = df.dropna(subset=feature_cols)

# 7. (Optional) Filter out ultra-rare building types
counts = df["building_t"].value_counts()
valid = counts[counts >= 2].index
df = df[df["building_t"].isin(valid)]

# 8. Encode target and build X, y
le = LabelEncoder()
y = le.fit_transform(df["building_t"])
X = df[feature_cols].values

# 9. Stratified split (each class now has ≥2 samples)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 10. Fit Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 11. Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 12. Restrict to only the labels present in the test set
test_labels = np.unique(y_test)
test_names = le.inverse_transform(test_labels)

print("\nClassification Report:")
print(
    classification_report(y_test, y_pred, labels=test_labels, target_names=test_names)
)

# 13. Correlate each feature with the encoded target
df["building_t_enc"] = y
corrs = (
    df[feature_cols + ["building_t_enc"]]
    .corr()["building_t_enc"]
    .drop("building_t_enc")
    .sort_values(ascending=False)
)

print("\nFeature ↔ Target Correlations:")
print(corrs)
