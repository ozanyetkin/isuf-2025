import os
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Load Data
def load_data(file_path):
    """Load spatial data into a GeoDataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    data = gpd.read_file(file_path)
    return data


# Visualize Data
def visualize_data(geo_data, output_path="land_use_map.png"):
    """Create a simple visualization of the GeoDataFrame and save it to a file."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    geo_data.plot(column="type", cmap="tab20", legend=True, ax=ax)
    plt.title("Land Use Map", fontsize=15)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# Visualize Predictions
def visualize_predictions(geo_data, predictions, output_path="predictions_map.png"):
    """Visualize the model predictions alongside the ground truth on the GeoDataFrame and save it to a file."""
    geo_data["predictions"] = predictions
    fig, axes = plt.subplots(1, 2, figsize=(30, 10))

    # Ground Truth
    geo_data.plot(column="type", cmap="tab20", legend=True, ax=axes[0])
    axes[0].set_title("Ground Truth Land Use Map", fontsize=15)

    # Predictions
    geo_data.plot(column="predictions", cmap="tab20", legend=True, ax=axes[1])
    axes[1].set_title("Predicted Land Use Map", fontsize=15)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# Preprocess Data
def preprocess_data(geo_data):
    """Preprocess spatial data for ML."""
    # Example: Drop rows with missing values
    geo_data = geo_data.dropna(subset=["type"]).copy()
    geo_data = geo_data[geo_data.is_valid]

    # Convert geometry to features (e.g., area, centroid coordinates)
    geo_data = geo_data.to_crs(epsg=3395)  # Re-project to a projected CRS
    geo_data["area"] = geo_data.geometry.area
    geo_data["centroid_x"] = geo_data.geometry.centroid.x
    geo_data["centroid_y"] = geo_data.geometry.centroid.y

    # Drop geometry for ML
    features = geo_data[["area", "centroid_x", "centroid_y"]]
    target = geo_data["type"]
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
    # File path to your dataset
    file_path = "data/planet_7.66971,45.05372_7.71254,45.07629-shp/shape/buildings.shp"

    try:
        data = load_data(file_path)
        print("Data loaded successfully.")

        features, target = preprocess_data(data)
        print("Data preprocessing complete.")

        visualize_data(data)

        model, X_test, y_test, predictions = train_model(features, target)
        print("Model training complete.")

        # Create a GeoDataFrame for the test data
        test_geo_data = data.iloc[X_test.index].copy()
        visualize_predictions(test_geo_data, predictions)

    except Exception as e:
        print(f"Error: {e}")
