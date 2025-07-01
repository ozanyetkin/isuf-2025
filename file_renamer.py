import os
import geopandas as gpd

#!/usr/bin/env python3


def rename_files_in_selected_cities(base_dir="data"):
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


def print_shp_column_names(base_dir="data"):
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


def rename_mismatched_columns(base_dir="data"):
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
            """
            if "building" in gdf.columns:
                rename_map["building"] = "building_t"
            if "osm_type" in gdf.columns and (
                folder_name == "Amsterdam" or folder_name == "Berlin"
            ):
                rename_map["osm_type"] = "building_t"
            """
            if "B_Coverage" in gdf.columns:
                rename_map["B_Coverage"] = "Building_C"

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


def lowercase_and_validate_columns(base_dir="data"):
    """
    For each .shp under base_dir, rename all columns to lowercase,
    drop extra columns if there are any, overwrite the file, and then verify
    that every file has the same set of columns.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    feature_sets = {}
    reference_columns = None  # To store the reference set of columns

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

            # Lowercase columns
            lower_cols = [col.lower() for col in gdf.columns]
            gdf.columns = lower_cols

            # Initialize reference columns if not set
            if reference_columns is None:
                reference_columns = set(lower_cols)
                print(f"Reference columns set to: {reference_columns}")
            else:
                # Drop extra columns
                extra_columns = set(lower_cols) - reference_columns
                if extra_columns:
                    print(f"Dropping extra columns in {shp_path}: {extra_columns}")
                    gdf = gdf.drop(columns=list(extra_columns))

            # Ensure only reference columns are present
            missing_columns = reference_columns - set(gdf.columns)
            if missing_columns:
                print(f"Missing columns in {shp_path}: {missing_columns}")
                for col in missing_columns:
                    gdf[col] = None  # Add missing columns with None values

            try:
                gdf.to_file(shp_path)
                print(f"Processed {shp_path} with consistent columns.")
            except Exception as e:
                print(f"Failed to write {shp_path}: {e}")
                continue

            feature_sets[shp_path] = set(gdf.columns)

    # Verify consistency
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


def detect_column_differences(base_dir="data"):
    """
    Detect and report differences in column names across all .shp files under base_dir,
    treating column names as case-insensitive.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    column_sets = {}

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

            # Store columns as lowercase for case-insensitive comparison
            column_sets[shp_path] = {col.lower() for col in gdf.columns}

    # Compare column sets
    all_columns = set.union(*column_sets.values()) if column_sets else set()
    print("All detected columns across shapefiles (case-insensitive):")
    print(all_columns)

    for path, cols in column_sets.items():
        missing = all_columns - cols
        extra = cols - all_columns
        if missing:
            print(f"{path} is missing columns: {missing}")
        if extra:
            print(f"{path} has extra columns: {extra}")


if __name__ == "__main__":
    rename_files_in_selected_cities()
    # print_shp_column_names()
    rename_mismatched_columns()
    # print_shp_column_names()
    lowercase_and_validate_columns()
    detect_column_differences()
