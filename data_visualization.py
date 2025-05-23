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

# --- 0) Reader-friendly display names ---
category_display = {
    "multi-family": "Multi-family",
    "single-family": "Single-family",
    "commercial": "Commercial",
    "industrial": "Industrial",
    "public": "Public",
    "infrastructure": "Infrastructure",
    "other": "Other",
}

feature_display = {
    "compactnes": "Compactness",
    "global_int": "Global Integration",
    "local_inte": "Local Integration",
    "building_c": "Building Coverage",
    "rbox_width": "Building X-Dimension",
    "rbox_height": "Building Y-Dimension",
}

# Color mapping remains the same
color_map = {
    k: v
    for k, v in {
        "multi-family": "#1F4E79",
        "single-family": "#D55E00",
        "commercial": "#3C7A3F",
        "industrial": "#A11D21",
        "public": "#7A4A91",
        "infrastructure": "#FF5C6A",
        "other": "#6B4A3F",
    }.items()
}
light_gray = "#e0e0e0"
gray_patch = Patch(color=light_gray, label="Train set")

output_dir = Path("images")
output_dir.mkdir(exist_ok=True)

# --- Load & preprocess ---
shp_paths = list(Path("data/Selected Cities").rglob("*.shp"))
gdfs = []
for fp in shp_paths:
    city = fp.parent.name
    g = gpd.read_file(fp)
    g["city"] = city
    gdfs.append(g)
if not gdfs:
    raise RuntimeError("No shapefiles found under data/Selected Cities")
df = pd.concat(gdfs, ignore_index=True)

df.columns = df.columns.str.lower()
df = df.dropna(subset=["building_t"]).to_crs(epsg=3857)


def rotated_dims(geom: Polygon):
    r = geom.minimum_rotated_rectangle
    pts = np.array(r.exterior.coords)[:4]
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return min(e1, e2), max(e1, e2)


r_dims = np.array([rotated_dims(g) for g in df.geometry])
df["rbox_width"], df["rbox_height"] = r_dims[:, 0], r_dims[:, 1]

# Map OSM building tags to your main categories
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

# --- Per-city visualizations ---
for city, sub in df.groupby("city"):
    vc = sub["building_main"].value_counts()
    valid = vc[vc >= 2].index
    sub = sub[sub["building_main"].isin(valid)]
    if sub["building_main"].nunique() < 2:
        continue

    idx = sub.index.to_numpy()
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=sub.loc[idx, "building_main"]
    )

    X_tr = sub.loc[idx_tr, features].values
    y_tr = LabelEncoder().fit_transform(sub.loc[idx_tr, "building_main"])
    X_te = sub.loc[idx_te, features].values

    le_city = LabelEncoder().fit(sub["building_main"])
    y_true = le_city.transform(sub.loc[idx_te, "building_main"])

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_true, y_pred)

    sub_test = sub.loc[idx_te].copy()
    sub_test["pred"] = le_city.inverse_transform(y_pred)
    sub_train = sub.loc[idx_tr]

    # 1) Side-by-side maps
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))
    sub_train.plot(ax=ax0, color=light_gray, linewidth=0.1, edgecolor="gray")
    for cat in le_city.classes_:
        display = category_display[cat]
        mask = sub_test["building_main"] == cat
        if mask.any():
            sub_test[mask].plot(
                ax=ax0, color=color_map[cat], linewidth=0.1, edgecolor="gray"
            )
    ax0.set_title(f"{city} – Ground Truth")
    ax0.legend(
        handles=[gray_patch]
        + [
            Patch(color=color_map[c], label=category_display[c])
            for c in le_city.classes_
        ],
        loc="lower left",
    )
    ax0.axis("off")

    sub_train.plot(ax=ax1, color=light_gray, linewidth=0.1, edgecolor="gray")
    for cat in le_city.classes_:
        mask = sub_test["pred"] == cat
        if mask.any():
            sub_test[mask].plot(
                ax=ax1, color=color_map[cat], linewidth=0.1, edgecolor="gray"
            )
    ax1.set_title(f"{city} – Predictions (Acc: {acc:.2f})")
    ax1.legend(
        handles=[gray_patch]
        + [
            Patch(color=color_map[c], label=category_display[c])
            for c in le_city.classes_
        ],
        loc="lower left",
    )
    ax1.axis("off")

    plt.tight_layout(pad=2.0)
    fig.savefig(output_dir / f"{city}_comparison.png", dpi=300)
    plt.close(fig)

    # 2) City-wise confusion matrix (%)
    labels = range(len(le_city.classes_))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im = ax2.imshow(cm_pct, interpolation="nearest", cmap="Blues")
    ax2.set_xticks(labels)
    ax2.set_yticks(labels)
    ax2.set_xticklabels(
        [category_display[c] for c in le_city.classes_], rotation=45, ha="right"
    )
    ax2.set_yticklabels([category_display[c] for c in le_city.classes_])
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
    ax2.set_xlabel("Predicted Category")
    ax2.set_ylabel("Actual Category")
    ax2.set_title(f"{city} – Confusion Matrix (%)")
    fig2.colorbar(im, ax=ax2)
    fig2.tight_layout()
    fig2.savefig(output_dir / f"{city}_confusion_matrix.png", dpi=300)
    plt.close(fig2)

    # 3) City-wise feature importance
    importances = model.feature_importances_
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.barh([feature_display[f] for f in reversed(features)], reversed(importances))
    ax3.set_xlabel("Relative Importance")
    ax3.set_title(f"{city} – Feature Importances")
    fig3.tight_layout()
    fig3.savefig(output_dir / f"{city}_feature_importance.png", dpi=300)
    plt.close(fig3)

# --- 4) Overall model on full data ---
idx_all = df.index.to_numpy()
idx_tr_all, idx_te_all = train_test_split(
    idx_all, test_size=0.2, random_state=42, stratify=df.loc[idx_all, "building_main"]
)
X_tr_all = df.loc[idx_tr_all, features].values
y_tr_all = LabelEncoder().fit_transform(df.loc[idx_tr_all, "building_main"])
X_te_all = df.loc[idx_te_all, features].values

le_all = LabelEncoder().fit(df["building_main"])
y_true_all = le_all.transform(df.loc[idx_te_all, "building_main"])

model_all = RandomForestClassifier(n_estimators=100, random_state=42)
model_all.fit(X_tr_all, y_tr_all)
y_pred_all = model_all.predict(X_te_all)
acc_all = accuracy_score(y_true_all, y_pred_all)

# 4a) Overall confusion matrix
cm_all = confusion_matrix(y_true_all, y_pred_all, labels=range(len(le_all.classes_)))
cm_all_pct = cm_all.astype(float) / cm_all.sum(axis=1, keepdims=True) * 100
fig4, ax4 = plt.subplots(figsize=(8, 6))
im4 = ax4.imshow(cm_all_pct, interpolation="nearest", cmap="Blues")
ax4.set_xticks(range(len(le_all.classes_)))
ax4.set_yticks(range(len(le_all.classes_)))
ax4.set_xticklabels(
    [category_display[c] for c in le_all.classes_], rotation=45, ha="right"
)
ax4.set_yticklabels([category_display[c] for c in le_all.classes_])
for i in range(len(le_all.classes_)):
    for j in range(len(le_all.classes_)):
        ax4.text(
            j,
            i,
            f"{cm_all_pct[i, j]:.1f}%",
            ha="center",
            va="center",
            color="white" if cm_all_pct[i, j] > 50 else "black",
        )
ax4.set_xlabel("Predicted Category")
ax4.set_ylabel("Actual Category")
ax4.set_title(f"Overall Confusion Matrix (%) (Acc: {acc_all:.2f})")
fig4.colorbar(im4, ax=ax4)
fig4.tight_layout()
fig4.savefig(output_dir / "overall_confusion_matrix.png", dpi=300)
plt.close(fig4)

# 4b) Overall feature importance
feat_imp_all = model_all.feature_importances_
fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.barh([feature_display[f] for f in reversed(features)], reversed(feat_imp_all))
ax5.set_xlabel("Relative Importance")
ax5.set_title("Overall Feature Importances")
fig5.tight_layout()
fig5.savefig(output_dir / "overall_feature_importance.png", dpi=300)
plt.close(fig5)

print(f"Saved all plots (city-wise + overall) to {output_dir.resolve()}")
