import os
import geopandas as gpd

#!/usr/bin/env python3


def rename_files_in_selected_cities(base_dir="data/Selected Cities"):
    """
    For each subfolder in base_dir, rename all files inside it to
    {subfolder_name}_buildings.ext, preserving the original extension.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for filename in os.listdir(folder_path):
            old_path = os.path.join(folder_path, filename)
            if not os.path.isfile(old_path):
                continue

            _, ext = os.path.splitext(filename)
            new_filename = f"{folder_name}_buildings{ext}"
            new_path = os.path.join(folder_path, new_filename)

            if os.path.exists(new_path):
                print(f"Skipping {old_path}: {new_filename} already exists")
                continue

            os.rename(old_path, new_path)
            print(f"Renamed {old_path} -> {new_path}")


def print_shp_column_names(base_dir="data/Selected Cities"):
    """
    After renaming, find all .shp files under base_dir, read them with geopandas,
    and print their column names for each file.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".shp"):
                continue

            shp_path = os.path.join(folder_path, fname)
            try:
                gdf = gpd.read_file(shp_path)
            except Exception as e:
                print(f"Failed to read {shp_path}: {e}")
                continue

            cols = gdf.columns.tolist()
            print(f"Columns in {shp_path}:")
            for col in cols:
                print(f"  {col}")
            print()


def rename_mismatched_columns(base_dir="data/Selected Cities"):
    """
    Rename 'osm_type' column to 'Building_t' and 'b_coverage' column to 'building_c'
    in all .shp files under base_dir and overwrite them.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".shp"):
                continue

            shp_path = os.path.join(folder_path, fname)
            try:
                gdf = gpd.read_file(shp_path)
            except Exception as e:
                print(f"Failed to read {shp_path}: {e}")
                continue

            rename_map = {}
            if "osm_type" in gdf.columns:
                rename_map["osm_type"] = "Building_t"
            if "b_coverage" in gdf.columns:
                rename_map["b_coverage"] = "building_c"

            if not rename_map:
                print(f"No columns to rename in {shp_path}, skipping")
                continue

            gdf = gdf.rename(columns=rename_map)
            try:
                gdf.to_file(shp_path)
                print(f"Renamed columns in {shp_path}: {rename_map}")
            except Exception as e:
                print(f"Failed to write {shp_path}: {e}")
                continue


def lowercase_and_validate_columns(base_dir="data/Selected Cities"):
    """
    For each .shp under base_dir, rename all columns to lowercase,
    overwrite the file, and then verify that every file has the same set of columns.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    feature_sets = {}
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".shp"):
                continue

            shp_path = os.path.join(folder_path, fname)
            try:
                gdf = gpd.read_file(shp_path)
            except Exception as e:
                print(f"Failed to read {shp_path}: {e}")
                continue

            # lowercase columns
            lower_cols = [col.lower() for col in gdf.columns]
            if list(gdf.columns) != lower_cols:
                gdf.columns = lower_cols
                try:
                    gdf.to_file(shp_path)
                    print(f"Lowercased columns in {shp_path}")
                except Exception as e:
                    print(f"Failed to write {shp_path}: {e}")
                    continue

            feature_sets[shp_path] = set(lower_cols)

    # verify consistency
    unique_sets = {}
    for path, cols in feature_sets.items():
        key = tuple(sorted(cols))
        unique_sets.setdefault(key, []).append(path)

    if len(unique_sets) == 1:
        print("All shapefiles have consistent, lowercase column names.")
    else:
        print("Inconsistent column sets detected:")
        for cols_tuple, paths in unique_sets.items():
            print(f"Columns: {cols_tuple}")
            for p in paths:
                print(f"  {p}")


if __name__ == "__main__":
    rename_files_in_selected_cities()
    print_shp_column_names()
    rename_mismatched_columns()
    print_shp_column_names()
    lowercase_and_validate_columns()
