import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import folium


# Load Data
def load_data(file_path):
    """Load spatial data into a GeoDataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")
    data = gpd.read_file(file_path)
    return data


# Visualize Data
def visualize_data(geo_data):
    """Create a simple visualization of the GeoDataFrame."""
    geo_data.plot(column="type", cmap="viridis", legend=True)
    plt.title("Land Use Map")
    plt.show()


# Preprocess Data
def preprocess_data(geo_data):
    """Preprocess spatial data for ML."""
    # Example: Drop rows with missing values
    geo_data = geo_data.dropna(subset=["type"])

    # Convert geometry to features (e.g., area, centroid coordinates)
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
    return model


# Main Script
if __name__ == "__main__":
    # File path to your dataset
    file_path = "data/planet_13.046,52.361_13.731,52.672-shp/shape/buildings.shp"

    try:
        data = load_data(file_path)
        print("Data loaded successfully.")

        visualize_data(data)

        # features, target = preprocess_data(data)
        # print("Data preprocessing complete.")

        # model = train_model(features, target)
        # print("Model training complete.")

    except Exception as e:
        print(f"Error: {e}")
