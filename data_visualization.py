import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

# ─── suppress the undefined-metric warning ────────────────────────────────
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ─── 1. load & tag by city ────────────────────────────────────────────────
shp_paths = list(Path("data/Selected Cities").rglob("*.shp"))
gdfs = []
for fp in shp_paths:
    city = Path(fp).parent.name
    g = gpd.read_file(fp)
    g["city"] = city
    gdfs.append(g)
if not gdfs:
    raise RuntimeError("No shapefiles found under data/Selected Cities")
df = pd.concat(gdfs, ignore_index=True)

# ─── 2. normalize, drop missing, reproject ────────────────────────────────
df.columns = df.columns.str.lower()
df = df.dropna(subset=["building_t"]).to_crs(epsg=3857)


# ─── 3. compute rotated-bbox dims ─────────────────────────────────────────
def rotated_dims(geom: Polygon):
    r = geom.minimum_rotated_rectangle
    pts = np.array(r.exterior.coords)[:4]
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return min(e1, e2), max(e1, e2)


r_dims = np.array([rotated_dims(g) for g in df.geometry])
df["rbox_width"], df["rbox_height"] = r_dims[:, 0], r_dims[:, 1]

# ─── 4. map into main categories ──────────────────────────────────────────
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

# ─── 5. select & clean features ───────────────────────────────────────────
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

# ─── 6. for each city: train & plot ───────────────────────────────────────
for city, sub in df.groupby("city"):
    # filter out tiny classes to allow stratify
    vc = sub["building_main"].value_counts()
    valid = vc[vc >= 2].index
    sub = sub[sub["building_main"].isin(valid)]
    if sub["building_main"].nunique() < 2:
        continue

    # split indices (so geometry stays aligned)
    idx = sub.index.to_numpy()
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=sub.loc[idx, "building_main"]
    )

    # prepare data
    X_tr = sub.loc[idx_tr, features].values
    y_tr = LabelEncoder().fit_transform(sub.loc[idx_tr, "building_main"])
    X_te = sub.loc[idx_te, features].values
    le_city = LabelEncoder().fit(sub["building_main"])
    y_true = le_city.transform(sub.loc[idx_te, "building_main"])

    # train and predict
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = le_city.classes_

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(f"{city}: Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.tight_layout()

    # feature importances
    fi = pd.Series(model.feature_importances_, index=features).sort_values(
        ascending=False
    )
    plt.figure(figsize=(6, 4))
    fi.plot.bar()
    plt.title(f"{city}: Feature Importances")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # map of predictions
    sub_test = sub.loc[idx_te].copy()
    sub_test["pred"] = le_city.inverse_transform(y_pred)
    plt.figure(figsize=(6, 6))
    ax = sub_test.plot(
        column="pred",
        categorical=True,
        legend=True,
        legend_kwds={"loc": "best"},
        linewidth=0.1,
        edgecolor="gray",
    )
    ax.set_title(f"{city}: Predicted Building Types")
    ax.set_axis_off()
    plt.tight_layout()

# show all
plt.show()
