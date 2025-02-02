import overpy
import pyvista as pv
import numpy as np
from pyproj import Transformer


# Function to determine UTM zone based on longitude
def get_utm_zone(lon):
    return int((lon + 180) / 6) + 1  # UTM zones are 6 degrees wide


# Convert lat/lon to UTM coordinates dynamically
def latlon_to_utm(lon, lat):
    utm_zone = get_utm_zone(lon)
    transformer = Transformer.from_crs(
        f"epsg:4326", f"epsg:326{utm_zone}", always_xy=True
    )  # EPSG:4326 is WGS84
    x, y = transformer.transform(lon, lat)
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


# Extract building type
def get_building_type(tags):
    return tags.get("building", "unknown")


# Extract building geometry
def extract_building_geometry(osm_data):
    buildings = []
    for way in osm_data.ways:
        nodes = [(float(node.lon), float(node.lat)) for node in way.nodes]
        height = get_building_height(way.tags)
        building_type = get_building_type(way.tags)
        if np.isnan(height):
            height = 10  # Default height if not available

        # Convert lat/lon to UTM (XY)
        nodes = np.array([latlon_to_utm(lon, lat) for lon, lat in nodes])

        # Create bottom and top layers
        base = np.column_stack((nodes, np.zeros(len(nodes))))
        top = np.column_stack((nodes, np.full(len(nodes), height)))

        buildings.append((base, top, building_type))
    return buildings


# Define colors for different building types
def get_building_color(building_type):
    color_map = {
        "residential": "blue",
        "commercial": "red",
        "industrial": "gray",
        "education": "green",
        "hospital": "yellow",
        "church": "purple",
        "unknown": "white",
    }
    return color_map.get(building_type, "white")


# Create and plot 3D buildings
def plot_buildings(buildings):
    plotter = pv.Plotter()

    for base, top, building_type in buildings:
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
        plotter.add_mesh(mesh, color=get_building_color(building_type), opacity=0.8)

    plotter.show()


# Define bounding box (latitude_min, longitude_min, latitude_max, longitude_max)
bbox = (52.5160, 13.3777, 52.5200, 13.3827)  # Example: Berlin Brandenburg Gate

osm_data = get_osm_3d_buildings(bbox)
buildings = extract_building_geometry(osm_data)
plot_buildings(buildings)
