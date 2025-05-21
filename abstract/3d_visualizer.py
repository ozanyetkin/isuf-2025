import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Load the data
gdf = gpd.read_file("./data/3d/rome_italy_buildings.shp")

# Use only the first 100 buildings
gdf = gdf.head(200)
# print(list(gdf.columns))

# Extract coordinates and height
gdf["height"] = (
    gdf["height"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
)

print(gdf["height"].describe())

# Use the "level" column to calculate height
# gdf["height"] = gdf["level"] * 3

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot buildings
for geom, height in zip(gdf.geometry, gdf["height"]):
    if geom.geom_type == "Polygon":  # Single building footprint
        x, y = np.array(geom.exterior.xy)
        ax.plot_trisurf(x, y, np.full_like(x, height), color="grey", alpha=0.7)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Height (m)")
# plt.show()
plt.savefig("3d_buildings.png", bbox_inches="tight")