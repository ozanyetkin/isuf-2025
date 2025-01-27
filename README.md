# Land Use Classification Pipeline

This project is a pipeline for loading, preprocessing, visualizing, and classifying land use data using a Random Forest model.

## Project Structure

```plaintext
.gitignore
data/
    AL001L1_TIRANA_UA2018_v013.gpkg
    planet_13.046,52.361_13.731,52.672-shp/
        CHECKSUM.txt
        logfile.txt
        README.html
        README.txt
        shape/
            buildings.cpg
            buildings.dbf
            buildings.prj
            buildings.shp
            buildings.shx
            landuse.cpg
            landuse.dbf
            landuse.prj
            landuse.shp
            landuse.shx
            natural.cpg
            ...
    planet_7.66971,45.05372_7.71254,45.07629-shp/
        CHECKSUM.txt
        logfile.txt
        README.html
        README.txt
        shape/
            ...


gpkg_reader.py




pipeline.py




README.md


```

## Requirements

- Python 3.x
- geopandas
- matplotlib
- scikit-learn

You can install the required packages using pip:

```sh
pip install geopandas matplotlib scikit-learn
```

## Usage

### Load Data

The

load_data

function loads spatial data into a GeoDataFrame.

```python
from pipeline import load_data

file_path = "data/planet_7.66971,45.05372_7.71254,45.07629-shp/shape/buildings.shp"
data = load_data(file_path)
print(data.head())
```

### Visualize Data

The

visualize_data

function creates a simple visualization of the GeoDataFrame and saves it to a file.

```python
from pipeline import visualize_data

visualize_data(data, output_path="land_use_map.png")
```

### Preprocess Data

The

preprocess_data

function preprocesses spatial data for machine learning.

```python
from pipeline import preprocess_data

features, target = preprocess_data(data)
```

### Train Model

The

train_model

function trains a Random Forest model to predict land use.

```python
from pipeline import train_model

model, X_test, y_test, predictions = train_model(features, target)
```

### Visualize Predictions

The

visualize_predictions

function visualizes the model predictions alongside the ground truth on the GeoDataFrame and saves it to a file.

```python
from pipeline import visualize_predictions

test_geo_data = data.iloc[X_test.index].copy()
visualize_predictions(test_geo_data, predictions, output_path="predictions_map.png")
```

### Main Script

The main script in

pipeline.py

runs the entire pipeline.

```sh
python pipeline.py
```

## License

This project is licensed under the MIT License.
