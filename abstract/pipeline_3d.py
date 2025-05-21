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


# Create and plot 3D buildings with individual colors
def plot_buildings(buildings):
    plotter = pv.Plotter()
    multi_block = pv.MultiBlock()  # Store all buildings in one dataset

    for idx, (base, top, building_type) in enumerate(buildings):
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

        # Add the mesh with a unique color
        multi_block.append(mesh)

        # Apply individual colors to each building
        plotter.add_mesh(mesh, color=get_building_color(building_type), opacity=0.8)

    plotter.show()


# Define bounding box (latitude_min, longitude_min, latitude_max, longitude_max)
# bbox = (37.7740, -122.4310, 37.7840, -122.4110)  # Larger area in San Francisco

# Define bounding boxes for multiple cities
cities_bbox = {
    "New York": (40.7128, -74.0060, 40.7228, -73.9960),
    "London": (51.5074, -0.1278, 51.5174, -0.1178),
    "Paris": (48.8566, 2.3522, 48.8666, 2.3622),
    "Tokyo": (35.6895, 139.6917, 35.6995, 139.7017),
    "Sydney": (-33.8688, 151.2093, -33.8588, 151.2193),
    "Berlin": (52.5200, 13.4050, 52.5300, 13.4150),
    "Moscow": (55.7558, 37.6173, 55.7658, 37.6273),
    "Rio de Janeiro": (-22.9068, -43.1729, -22.8968, -43.1629),
    "Cape Town": (-33.9249, 18.4241, -33.9149, 18.4341),
    "Singapore": (1.3521, 103.8198, 1.3621, 103.8298),
}

city = "London"
bbox = cities_bbox[city]

print(f"Processing city: {city}")
osm_data = get_osm_3d_buildings(bbox)
buildings = extract_building_geometry(osm_data)
plot_buildings(buildings)
