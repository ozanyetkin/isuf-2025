import geopandas as gpd
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Read all .shp files recursively
shp_paths = list(Path("data/Selected Cities").rglob("*.shp"))
gdfs = [gpd.read_file(fp) for fp in shp_paths]
df = pd.concat(gdfs, ignore_index=True)

# Lower-case all column names
df.columns = df.columns.str.lower()

# Drop rows missing the target
df = df.dropna(subset=["building_t"])

# Explicitly pick the numeric features you care about
feature_cols = ["compactnes", "global_int", "local_inte", "building_c"]

# Coerce those columns to floats (invalid parsing → NaN)
for c in feature_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop any rows where we now lost a feature
df = df.dropna(subset=feature_cols)

# Filter out ultra-rare building types
counts = df["building_t"].value_counts()
valid = counts[counts >= 2].index
df = df[df["building_t"].isin(valid)]

# Encode target and build X, y
le = LabelEncoder()
y = le.fit_transform(df["building_t"])
X = df[feature_cols].values

# Stratified split (each class now has ≥2 samples)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Restrict to only the labels present in the test set
test_labels = np.unique(y_test)
test_names = le.inverse_transform(test_labels)

print("\nClassification Report:")
print(
    classification_report(y_test, y_pred, labels=test_labels, target_names=test_names)
)

# Feature Importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\nFeature Importances:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {feature_cols[indices[f]]}: {importances[indices[f]]:.4f}")

# Correlation matrix (encode building_t to numeric codes first)
df["building_t_code"] = le.transform(df["building_t"])
corrs = df[feature_cols + ["building_t_code"]].corr()
corrs = corrs.loc[feature_cols, "building_t_code"].sort_values(ascending=False)

print("\nFeature Correlations:")
print(corrs)
