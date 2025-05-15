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


def rename_shp_osm_type_to_building_t(base_dir="data/Selected Cities"):
    """
    Rename 'osm_type' column to 'Building_t' in all .shp files under base_dir and overwrite them.
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

            if "osm_type" not in gdf.columns:
                print(f"'osm_type' not found in {shp_path}, skipping")
                continue

            gdf = gdf.rename(columns={"osm_type": "Building_t"})
            try:
                gdf.to_file(shp_path)
                print(f"Renamed column in {shp_path}")
            except Exception as e:
                print(f"Failed to write {shp_path}: {e}")
                continue


if __name__ == "__main__":
    rename_files_in_selected_cities()
    print_shp_column_names()
    rename_shp_osm_type_to_building_t()
    print_shp_column_names()
