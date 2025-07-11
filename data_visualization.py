# visualize.py
import warnings
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Patch

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

from data_preprocessing import FEATURES
from data_prediction import (
    df,
    best_model,
    le,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Display mappings
category_display = {
    "multi-family": "Apartment",
    "single-family": "House",
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
    "rbox_width": "Building X-Dim",
    "rbox_height": "Building Y-Dim",
}
color_map = {
    "multi-family": "#96403f",
    "single-family": "#ffbf41",
    "commercial": "#fd3f3f",
    "industrial": "#d340ff",
    "public": "#ffcfcf",
    "infrastructure": "#41d4ff",
    "other": "#666666",
}
light_gray = "#b1b1b1"

# Output directory
out = Path("images")
out.mkdir(exist_ok=True)

# Helper to get feature importances whether model is pipeline or estimator


def get_feature_importances(model):
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if isinstance(model, Pipeline) and "clf" in model.named_steps:
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            return clf.feature_importances_
    return None


for city in df["city"].unique():
    sub = df[df["city"] == city]
    if sub["building_main"].nunique() < 2:
        continue

    # Predict on all buildings
    X = sub[FEATURES].values
    y_true = le.transform(sub["building_main"])
    y_pred = best_model.predict(X)
    acc = accuracy_score(y_true, y_pred)

    sub = sub.copy()
    sub["pred"] = le.inverse_transform(y_pred)

    # Side-by-side maps
    legend_patches = [
        Patch(color=color_map[c], label=category_display[c]) for c in le.classes_
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(16, 8))
    sub.plot(ax=ax0, color=light_gray, linewidth=0.1, edgecolor="silver")
    for cat in le.classes_:
        mask = sub["building_main"] == cat
        if mask.any():
            sub[mask].plot(
                ax=ax0, color=color_map[cat], linewidth=0.1, edgecolor="silver"
            )
    ax0.set_title(f"{city} – Ground Truth")
    ax0.legend(handles=legend_patches, loc="lower left")
    ax0.axis("off")

    sub.plot(ax=ax1, color=light_gray, linewidth=0.1, edgecolor="silver")
    for cat in le.classes_:
        mask = sub["pred"] == cat
        if mask.any():
            sub[mask].plot(
                ax=ax1, color=color_map[cat], linewidth=0.1, edgecolor="silver"
            )
    ax1.set_title(f"{city} – Predictions (Acc: {acc:.2f})")
    ax1.legend(handles=legend_patches, loc="lower left")
    ax1.axis("off")

    plt.tight_layout(pad=2.0)
    fig.savefig(out / f"{city}_comparison.png", dpi=300)
    plt.close(fig)

    # Confusion matrix (%)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.imshow(cm_pct, interpolation="nearest", cmap="Blues")
    ax2.set_xticks(range(len(le.classes_)))
    ax2.set_yticks(range(len(le.classes_)))
    ax2.set_xticklabels(
        [category_display[c] for c in le.classes_], rotation=45, ha="right"
    )
    ax2.set_yticklabels([category_display[c] for c in le.classes_])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(
                j,
                i,
                f"{cm_pct[i, j]:.1f}%",
                ha="center",
                va="center",
                color="white" if cm_pct[i, j] > 50 else "black",
            )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title(f"{city} – Confusion Matrix (%)")
    fig2.tight_layout()
    fig2.savefig(out / f"{city}_confusion_matrix.png", dpi=300)
    plt.close(fig2)

    # Feature importances
    fi = get_feature_importances(best_model)
    if fi is not None:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.barh(
            [feature_display[f] for f in reversed(FEATURES)],
            list(reversed(fi)),
            color="#6aadd5",
            height=0.4,
        )
        ax3.set_xlabel("Relative Importance")
        ax3.set_title(f"{city} – Feature Importances")
        fig3.tight_layout()
        fig3.savefig(out / f"{city}_feature_importance.png", dpi=300)
        plt.close(fig3)

# Overall confusion and feature importance

y_all = le.transform(df["building_main"])
y_pred_all = best_model.predict(df[FEATURES].values)
cm_all = confusion_matrix(y_all, y_pred_all, labels=range(len(le.classes_)))
cm_all_pct = cm_all.astype(float) / cm_all.sum(axis=1, keepdims=True) * 100

fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.imshow(cm_all_pct, interpolation="nearest", cmap="Blues")
ax4.set_xticks(range(len(le.classes_)))
ax4.set_yticks(range(len(le.classes_)))
ax4.set_xticklabels([category_display[c] for c in le.classes_], rotation=45, ha="right")
ax4.set_yticklabels([category_display[c] for c in le.classes_])
for i in range(cm_all.shape[0]):
    for j in range(cm_all.shape[1]):
        ax4.text(
            j,
            i,
            f"{cm_all_pct[i, j]:.1f}%",
            ha="center",
            va="center",
            color="white" if cm_all_pct[i, j] > 50 else "black",
        )
ax4.set_xlabel("Predicted")
ax4.set_ylabel("Actual")
ax4.set_title("Overall Confusion Matrix (%)")
fig4.tight_layout()
fig4.savefig(out / "overall_confusion_matrix.png", dpi=300)
plt.close(fig4)

fi_all = get_feature_importances(best_model)
if fi_all is not None:
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.barh(
        [feature_display[f] for f in reversed(FEATURES)],
        list(reversed(fi_all)),
        color="#6aadd5",
        height=0.4,
    )
    ax5.set_xlabel("Relative Importance")
    ax5.set_title("Overall Feature Importances")
    fig5.tight_layout()
    fig5.savefig(out / "overall_feature_importance.png", dpi=300)
    plt.close(fig5)

print(f"All figures saved to {out.resolve()}")
