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
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Suppress warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Consistent categories & colors
categories = [
    "multi-family",
    "single-family",
    "commercial",
    "industrial",
    "public",
    "infrastructure",
    "other",
]
color_map = {
    "multi-family": "#1F4E79",
    "single-family": "#D55E00",
    "commercial": "#3C7A3F",
    "industrial": "#A11D21",
    "public": "#7A4A91",
    "infrastructure": "#FF5C6A",
    "other": "#6B4A3F",
}

# lighter gray for train set
light_gray = "#e0e0e0"
gray_patch = Patch(color=light_gray, label="train set")

# Output folder
output_dir = Path("images")
output_dir.mkdir(exist_ok=True)

# 1. Load all shapefiles, tag by city
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

# 2. Preprocess
df.columns = df.columns.str.lower()
df = df.dropna(subset=["building_t"]).to_crs(epsg=3857)


# 3. Rotated‐bbox dims
def rotated_dims(geom: Polygon):
    r = geom.minimum_rotated_rectangle
    pts = np.array(r.exterior.coords)[:4]
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return min(e1, e2), max(e1, e2)


r_dims = np.array([rotated_dims(g) for g in df.geometry])
df["rbox_width"], df["rbox_height"] = r_dims[:, 0], r_dims[:, 1]

# 4. Main category mapping
group_map = {"apartments": "multi-family"}
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

# 5. Numeric features
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

# 6. Per-city train & side-by-side plots
for city, sub in df.groupby("city"):
    # require at least two samples per class
    vc = sub["building_main"].value_counts()
    valid = vc[vc >= 2].index
    sub = sub[sub["building_main"].isin(valid)]  # "isin" is a valid pandas method
    if sub["building_main"].nunique() < 2:
        continue

    # split indices
    idx = sub.index.to_numpy()
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=sub.loc[idx, "building_main"]
    )

    # training & test data
    X_tr = sub.loc[idx_tr, features].values
    y_tr = LabelEncoder().fit_transform(sub.loc[idx_tr, "building_main"])
    X_te = sub.loc[idx_te, features].values
    le_city = LabelEncoder().fit(sub["building_main"])
    y_true = le_city.transform(sub.loc[idx_te, "building_main"])

    # train & predict
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_true, y_pred)

    # prepare GeoDataFrames
    sub_test = sub.loc[idx_te].copy()
    sub_test["pred"] = le_city.inverse_transform(y_pred)
    sub_train = sub.loc[idx_tr]

    # === 1) side-by-side comparison (with lighter gray) ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    # ground truth
    sub_train.plot(ax=axes[0], color=light_gray, linewidth=0.1, edgecolor="gray")
    for cat in categories:
        mask = sub_test["building_main"] == cat
        if not sub_test[mask].empty:
            sub_test[mask].plot(
                ax=axes[0], color=color_map[cat], linewidth=0.1, edgecolor="gray"
            )
    axes[0].set_title(f"{city} Ground Truth")
    axes[0].legend(
        handles=[gray_patch] + [Patch(color=color_map[c], label=c) for c in categories],
        loc="lower left",
    )
    axes[0].axis("off")

    # predictions
    sub_train.plot(ax=axes[1], color=light_gray, linewidth=0.1, edgecolor="gray")
    for cat in categories:
        mask = sub_test["pred"] == cat  # ← fix here
        if not sub_test[mask].empty:
            sub_test[mask].plot(
                ax=axes[1], color=color_map[cat], linewidth=0.1, edgecolor="gray"
            )
    axes[1].set_title(f"{city} Predictions (Acc: {acc:.2f})")
    axes[1].legend(
        handles=[gray_patch] + [Patch(color=color_map[c], label=c) for c in categories],
        loc="lower left",
    )
    axes[1].axis("off")

    plt.tight_layout(pad=2.0)
    (output_dir / f"{city}_comparison.png").write_bytes(
        fig.canvas.tostring_argb()
    )  # or fig.savefig(...)
    fig.savefig(output_dir / f"{city}_comparison.png", dpi=300)
    plt.close(fig)

    # === 2) confusion‐matrix heatmap (percent) ===
    # numeric true/pred arrays are y_true, y_pred
    labels = range(len(le_city.classes_))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(cm_pct, interpolation="nearest", cmap="Blues")
    ax2.set_xticks(labels)
    ax2.set_yticks(labels)
    ax2.set_xticklabels(le_city.classes_, rotation=45, ha="right")
    ax2.set_yticklabels(le_city.classes_)
    for i in labels:
        for j in labels:
            ax2.text(
                j,
                i,
                f"{cm_pct[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if cm_pct[i, j] > 50 else "black",
            )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title(f"{city} Confusion Matrix (%)")
    fig2.colorbar(im, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(output_dir / f"{city}_confusion_matrix.png", dpi=300)
    plt.close(fig2)

    # === 3) feature‐importance bar chart ===
    importances = model.feature_importances_
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.barh(features, importances)
    ax3.set_xlabel("Relative Importance")
    ax3.set_title(f"{city} Feature Importances")
    fig3.tight_layout()
    fig3.savefig(output_dir / f"{city}_feature_importance.png", dpi=300)
    plt.close(fig3)

print(
    f"Saved comparison, confusion matrix, and feature‐importance plots to {output_dir.resolve()}"
)
