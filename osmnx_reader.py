import os
import osmnx as ox

location = "Rome, Italy"

# Get the data
G = ox.features_from_place(location, tags={"building": True})
G = G[G.geometry.type == "Polygon"]

# Save the data as a shapefile
location_filename = location.lower().replace(",", "").replace(" ", "_")
output_dir = "./data/3d/"
os.makedirs(output_dir, exist_ok=True)
G.to_file(
    f"{output_dir}{location_filename}_buildings.shp",
    driver="ESRI Shapefile",
    geometry_type="Polygon",
)

print("Data saved as buildings.shp")
