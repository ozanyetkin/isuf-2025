# data_preprocessing.py
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon

# ─── Shared settings ────────────────────────────────────────────────────────
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


def rotated_dims(geom: Polygon):
    """Return (min_edge, max_edge) of the geometry’s minimum rotated rectangle."""
    r = geom.minimum_rotated_rectangle
    pts = np.array(r.exterior.coords)[:4]
    e1 = np.linalg.norm(pts[1] - pts[0])
    e2 = np.linalg.norm(pts[2] - pts[1])
    return min(e1, e2), max(e1, e2)


def load_and_preprocess(shp_dir: Path, epsg: int = 3857):
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

    return df
