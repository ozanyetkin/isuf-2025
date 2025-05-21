import os
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def get_subfolder_names(file_path):
    """Get all subfolder names under the given file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    subfolders = [f.name for f in os.scandir(file_path) if f.is_dir()]
    return subfolders


# Load Data
def load_data(file_path):
    """Load spatial data into a GeoDataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    data = gpd.read_file(file_path)
    return data


# Visualize Data with Zoom Option
def visualize_data(geo_data, output_path="land_use_map.png", zoom_factor=0.5):
    """Create a visualization of the GeoDataFrame with zooming and save it to a file."""
    # Drop rows with NaN building types to avoid invalid RGBA arguments
    geo_data = geo_data.dropna(subset=["Building_t"]).copy()
    fig, ax = plt.subplots(1, 1, figsize=(50, 50))
    # Accumulate unique Building_t categories across all calls
    global _ALL_CLASSES
    if "_ALL_CLASSES" not in globals():
        _ALL_CLASSES = set()
    _ALL_CLASSES.update(geo_data["Building_t"].dropna().unique())

    # Build a consistent colormap over every category seen so far
    unique_classes = sorted(_ALL_CLASSES)
    cmap = plt.get_cmap("Dark2", len(unique_classes))
    palette = {cls: cmap(i) for i, cls in enumerate(unique_classes)}

    # Map each feature to its color and plot
    geo_data["color"] = geo_data["Building_t"].map(palette)
    geo_data.plot(color=geo_data["color"], ax=ax)

    # Calculate the bounding box
    bounds = geo_data.total_bounds  # [minx, miny, maxx, maxy]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    width = (bounds[2] - bounds[0]) * zoom_factor
    height = (bounds[3] - bounds[1]) * zoom_factor

    ax.set_xlim(center_x - width / 2, center_x + width / 2)
    ax.set_ylim(center_y - height / 2, center_y + height / 2)

    city_name = os.path.splitext(os.path.basename(output_path))[0].replace(
        "land_use_map_", ""
    )
    ax.set_title(city_name, fontsize=15)
    ax.set_axis_off()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# Visualize Predictions
def visualize_predictions(
    geo_data, predictions, output_path="predictions_map.png", zoom_factor=0.5
):
    """Visualize the model predictions with zoom and save it to a file."""
    geo_data["predictions"] = predictions
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))

    bounds = geo_data.total_bounds  # [minx, miny, maxx, maxy]
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2

    width = (bounds[2] - bounds[0]) * zoom_factor
    height = (bounds[3] - bounds[1]) * zoom_factor

    # Ground Truth
    geo_data.plot(column="Building_t", cmap="tab20", legend=True, ax=axes[0])
    axes[0].set_xlim(center_x - width / 2, center_x + width / 2)
    axes[0].set_ylim(center_y - height / 2, center_y + height / 2)
    axes[0].set_title("Ground Truth Land Use Map", fontsize=15)

    # Predictions
    geo_data.plot(column="predictions", cmap="tab20", legend=True, ax=axes[1])
    axes[1].set_xlim(center_x - width / 2, center_x + width / 2)
    axes[1].set_ylim(center_y - height / 2, center_y + height / 2)
    axes[1].set_title("Predicted Land Use Map", fontsize=15)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# Preprocess Data
def preprocess_data(geo_data):
    """Preprocess spatial data for ML."""
    geo_data = geo_data.dropna(subset=["Building_t"]).copy()
    geo_data = geo_data[geo_data.is_valid]
    geo_data = geo_data.to_crs(epsg=3395)
    geo_data["area"] = geo_data.geometry.area
    geo_data["centroid_x"] = geo_data.geometry.centroid.x
    geo_data["centroid_y"] = geo_data.geometry.centroid.y
    features = geo_data[["area", "centroid_x", "centroid_y"]]
    target = geo_data["Building_t"]
    return features, target


# Train ML Model
def train_model(features, target):
    """Train a random forest model to predict land use."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    return model, X_test, y_test, predictions


# Main Script
if __name__ == "__main__":
    base_folder = "./data/Selected Cities"
    zoom_factor = 1.0  # Adjust this to control zoom level

    for file_name in get_subfolder_names(base_folder):
        file_path = f"{base_folder}/{file_name}/{file_name}_buildings.shp"

        try:
            data = load_data(file_path)
            print("Data loaded successfully.")

            features, target = preprocess_data(data)
            print("Data preprocessing complete.")
            
            visualize_data(
                data,
                output_path=f"land_use_map_{file_name}.png",
                zoom_factor=zoom_factor,
            )

        except Exception as e:
            print(f"Error: {e}")
