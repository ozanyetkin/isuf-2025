# data_preprocessing.py
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from shapely.geometry import Polygon
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Shared settings
FEATURES = [
    "compactnes",
    "global_int",
    "local_inte",
    "building_c",
    "rbox_width",
    "rbox_height",
]

# Map OSM tags into broad categories
GROUP_MAP = {
    **{"apartments": "multi-family"},
    **dict.fromkeys(
        [
            "residential",
            "house",
            "detached",
            "semidetached_house",
            "bungalow",
            "terrace",
            "allotment_house",
        ],
        "single-family",
    ),
    **dict.fromkeys(["industrial", "warehouse"], "industrial"),
    **dict.fromkeys(["retail", "office", "supermarket", "kiosk"], "commercial"),
    **dict.fromkeys(
        [
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
        ],
        "public",
    ),
    **dict.fromkeys(
        ["bridge", "viaduct", "railway", "transportation", "parking"], "infrastructure"
    ),
}


# Function to compute the dimensions of the minimum rotated rectangle
def rotated_dims(geom: Polygon):
    """Return (min_edge, max_edge) of the geometry’s minimum rotated rectangle."""
    r = geom.minimum_rotated_rectangle
    pts = np.array(r.exterior.coords)[:4]
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return min(e1, e2), max(e1, e2)


# Function to balance classes in a DataFrame
def balance_classes(
    df: pd.DataFrame,
    target_col: str = "building_main",
    method: str = "oversample",  # options: "oversample", "undersample", "smote"
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Balance DataFrame `df` on `target_col` via:
      - random oversample of minority classes
      - random undersample of majority classes
      - SMOTE synthetic oversampling
    """
    # separate majority/minority
    counts = df[target_col].value_counts()
    majority_cls = counts.idxmax()
    minority_cls = counts.idxmin()

    df_majority = df[df[target_col] == majority_cls]
    df_minority = df[df[target_col] == minority_cls]
    other = df[~df[target_col].isin([majority_cls, minority_cls])]

    if method == "oversample":
        # upsample minority to match majority
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=random_state,
        )
        balanced = pd.concat([df_majority, df_minority_upsampled, other])

    elif method == "undersample":
        # downsample majority to match minority
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=random_state,
        )
        balanced = pd.concat([df_majority_downsampled, df_minority, other])

    elif method == "smote":
        # apply SMOTE; only works on numeric features
        X = df[FEATURES]
        y = df[target_col]
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        balanced = pd.concat([X_res, y_res.rename(target_col)], axis=1)

    else:
        raise ValueError(f"Unknown balance method: {method!r}")

    return balanced.sample(frac=1, random_state=random_state)  # shuffle


# Function to load and preprocess shapefiles
def load_and_preprocess(
    shp_dir: Path,
    epsg: int = 3857,
    balance_method: str | None = None,
):
    """
    Reads all .shp under shp_dir, tags by city, computes rbox dims,
    maps building_main, filters & numeric-coerces FEATURES, returns a clean DataFrame.
    """
    shp_paths = list(Path(shp_dir).rglob("*.shp"))
    gdfs = []
    for fp in shp_paths:
        city = fp.parent.name if fp.parent != shp_dir else fp.stem
        g = gpd.read_file(fp)
        g["city"] = city
        gdfs.append(g)
    if not gdfs:
        raise RuntimeError(f"No shapefiles found in {shp_dir!r}")
    df = pd.concat(gdfs, ignore_index=True)

    # standardize column names, drop missing tags, reproject
    df.columns = df.columns.str.lower()
    df = df.dropna(subset=["building_t"]).to_crs(epsg=epsg)

    # rotated bbox dims
    r = np.array([rotated_dims(g) for g in df.geometry])
    df["rbox_width"], df["rbox_height"] = r[:, 0], r[:, 1]

    # map to main categories
    df["building_main"] = df["building_t"].map(GROUP_MAP).fillna("other")

    # coerce numeric and drop any remaining missing
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURES + ["building_main"])

    # only balance if requested
    if balance_method:
        df = balance_classes(df, target_col="building_main", method=balance_method)

    return df
