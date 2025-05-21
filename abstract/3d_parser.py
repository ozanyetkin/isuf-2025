import overpy
import pyvista as pv
import numpy as np
from pyproj import Proj, transform


# Convert lat/lon to UTM coordinates
def latlon_to_utm(lon, lat):
    proj_utm = Proj(
        proj="utm", zone=33, ellps="WGS84", datum="WGS84"
    )  # Adjust zone as needed
    x, y = proj_utm(lon, lat)
    return x, y


# Fetch 3D building data from OpenStreetMap
def get_osm_3d_buildings(bbox):
    api = overpy.Overpass()
    query = f"""
    [out:json];
    (
        way["building"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body; >; out skel qt;
    """
    return api.query(query)


# Extract building height
def get_building_height(tags):
    if "height" in tags:
        try:
            return float(tags["height"])
        except:
            return np.nan
    elif "building:levels" in tags:
        try:
            return float(tags["building:levels"]) * 3  # Assuming ~3 meters per level
        except:
            return np.nan
    else:
        return np.nan


# Extract building geometry
def extract_building_geometry(osm_data):
    buildings = []
    for way in osm_data.ways:
        nodes = [(float(node.lon), float(node.lat)) for node in way.nodes]
        height = get_building_height(way.tags)
        if np.isnan(height):
            height = 10  # Default height if not available

        # Convert lat/lon to UTM (XY)
        nodes = np.array([latlon_to_utm(lon, lat) for lon, lat in nodes])

        # Create bottom and top layers
        base = np.column_stack((nodes, np.zeros(len(nodes))))
        top = np.column_stack((nodes, np.full(len(nodes), height)))

        buildings.append((base, top))
    return buildings


# Create and plot 3D buildings
def plot_buildings(buildings):
    plotter = pv.Plotter()

    for base, top in buildings:
        num_points = len(base)

        # Create vertices
        vertices = np.vstack([base, top])

        # Create face connectivity list
        faces = []

        # Walls
        for i in range(num_points):
            j = (i + 1) % num_points  # Connect last to first
            faces.append([4, i, j, num_points + j, num_points + i])  # Quad face

        # Bottom face
        faces.append([num_points] + list(range(num_points)))

        # Top face
        faces.append([num_points] + list(range(num_points, 2 * num_points)))

        # Convert to numpy array
        faces = np.hstack(faces).astype(np.int32)

        # Create mesh
        mesh = pv.PolyData(vertices, faces)
        plotter.add_mesh(mesh, color="gray", opacity=0.8)

    plotter.show()


# Define bounding box (latitude_min, longitude_min, latitude_max, longitude_max)
bbox = (40.7120, -74.0100, 40.7180, -74.0000)  # Larger area around New York City, near the World Trade Center

osm_data = get_osm_3d_buildings(bbox)
buildings = extract_building_geometry(osm_data)
plot_buildings(buildings)
