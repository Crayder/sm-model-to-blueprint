import numpy as np
import json
from tqdm import tqdm
from itertools import product
from scipy.spatial import KDTree
from time import time
import trimesh

# Import constants
from constants import BLOCK_IDS, SM_COLORS_RGB, SM_COLORS_KDTREE, FALLBACK_MATERIAL, CONFIG_VALUES

# Stats container
stats = {
    "num_materials": 0,
    "num_vertices": 0,
    "num_faces": 0,
    "num_prisms": 0,
    "total_bounding_box": {
        "min": [0.0, 0.0, 0.0],
        "max": [0.0, 0.0, 0.0]
    },
    "voxel_grid_shape": [0, 0, 0],
    "num_voxels": 0,
    "num_minimal_prisms": 0,
    "largest_prism_aabb_radius": 0.0,

    "skipped_faces_zero_area": 0,
    "invalid_normals": 0,
    "skipped_faces_invalid_prism": 0,

    "obj_rotation_time": 0.0,
    "voxel_grid_creation_time": 0.0,
    "kd_tree_building_time": 0.0,
    "voxel_preparation_time": 0.0,
    "voxel_marking_time": 0.0,
    "voxel_decomposition_time": 0.0,
    "output_save_time": 0.0
}

# Function to triangulate a face with more than 3 vertices
def triangulate_face(face):
    triangles = []
    for i in range(1, len(face) - 1):
        triangles.append([face[0], face[i], face[i + 1]])
    return triangles

# Function to find the closest color
def find_closest_color(input_mtl_color):
    # Perform the KDTree query directly with normalized colors
    distance, idx = SM_COLORS_KDTREE.query([input_mtl_color])  # Input should be [0, 1] already
    
    # Since idx could be a scalar or array, access the correct element
    if isinstance(idx, np.ndarray):
        return SM_COLORS_RGB[idx[0]]  # Access the correct element
    else:
        raise ValueError("KDTree query did not return a valid index.")

# Function to parse the MTL file and extract material colors and names
def parse_mtl_file(filepath, use_scrap_colors):
    materials = {}
    current_material_full = None

    # check if exists, if so return -1
    try:
        with open(filepath, 'r') as file:
            for line in file:
                if line.startswith('newmtl'):
                    current_material_full = line.split()[1]  # Store the full name
                    current_material_short = current_material_full.split('.')[0]  # Shortened name
                elif line.startswith('Kd'):
                    if current_material_full:
                        color = [float(value) for value in line.split()[1:]]
                        shape_id = BLOCK_IDS.get(current_material_short, BLOCK_IDS["plastic"])  # Default to plastic if material name is not in BLOCK_IDS
                        materials[current_material_full] = {
                            "color": color,
                            "shapeId": shape_id,
                            "shortName": current_material_short  # Store the shortened name
                        }
                        if use_scrap_colors:  # Use the closest Scrap Mechanic color
                            materials[current_material_full]["color"] = find_closest_color(color)
                            print(f"Color {color} replaced with {materials[current_material_full]['color']} for material {current_material_full}")
                            
                        print(f"Material parsed: {current_material_full} with color {color} and shapeId {shape_id}")

                        stats["num_materials"] = len(materials)
        return materials
    except FileNotFoundError:
        print("MTL file not found. Using default color.")
        return materials

# Function to parse the OBJ file and extract vertices, faces, and face materials
def parse_obj_file(filepath, obj_scale, obj_offset, use_scrap_colors, use_set_color, set_color, use_set_block, set_block):
    vertices = []
    faces = []
    face_materials = []  # List for face materials
    current_material_full = None

    # If using one color AND one material, skip the MTL file parsing
    if use_set_color is False or use_set_block is False:
        print("Parsing MTL file...")
        mtl_filepath = filepath.replace('.obj', '.mtl')
        materials = parse_mtl_file(mtl_filepath, use_scrap_colors)
    else:
        print("Not using MTL file.")
        materials = {}

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                # First add the obj_offset to the values, then scale the vertices with obj_scale
                vertices.append((np.array([float(parts[i]) for i in range(1, 4)]) + obj_offset) * obj_scale)
            elif line.startswith('usemtl') and (use_set_color is False or use_set_block is False):
                current_material_full = line.split()[1]  # Use the full name for retrieval
                print(f"Using material: {current_material_full}")
            elif line.startswith('f '):
                parts = line.split()
                face = [int(idx.split('/')[0]) - 1 for idx in parts[1:]]
                
                # If no material is specified, use the default material
                if current_material_full is None:
                    material = FALLBACK_MATERIAL
                else:
                    material = materials.get(current_material_full, FALLBACK_MATERIAL)

                # If use_set_color is True, use the set_color
                if use_set_color is True:
                    material["color"] = set_color
                
                # If use_set_block is True, use the set_block
                if use_set_block is True:
                    material["shapeId"] = BLOCK_IDS.get(set_block, FALLBACK_MATERIAL["shapeId"])

                if len(face) == 3:
                    faces.append(face)
                    face_materials.append(material)  # Store face material
                elif len(face) > 3:
                    triangles = triangulate_face(face)
                    for triangle in triangles:
                        faces.append(triangle)
                        face_materials.append(material)  # Store material for each triangle
                else:
                    print(f"Skipping invalid face: {face}")
    
    # Add a fixed variation to all vertices to avoid whole numbers
    variation = np.random.uniform(0, 0.00025, 3) + 0.00025
    vertices = np.array(vertices) + variation

    stats["num_vertices"] = len(vertices)
    stats["num_faces"] = len(faces)

    return np.array(vertices), np.array(faces), face_materials  # Return face materials

# Function to create a rotation matrix for rotation around the X, Y, or Z axis
def create_rotation_matrix(axis, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    if axis == 'x':
        matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    return matrix

# Function to apply rotation to all vertices
def rotate_vertices(vertices, rotation_matrix):
    print("Rotating vertices...")
    start_time = time()
    rotated_vertices = np.dot(vertices, rotation_matrix.T)
    print(f"Rotation completed in {time() - start_time:.4f} seconds.")
    return rotated_vertices

# Function to calculate the bounding box for the model
def calculate_model_bounding_box(vertices):
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    stats["total_bounding_box"]["min"] = min_coords.tolist()
    stats["total_bounding_box"]["max"] = max_coords.tolist()
    return min_coords, max_coords

# Function to create the 3D Voxel Grid
def create_voxel_grid(min_coords, max_coords, voxel_scale):
    print("Creating voxel grid...")
    start_time = time()
    min_grid_coords = np.floor(min_coords / voxel_scale).astype(int)
    max_grid_coords = np.ceil(max_coords / voxel_scale).astype(int)
    grid_size = max_grid_coords - min_grid_coords + 1
    voxel_grid = np.zeros(grid_size, dtype=int)
    voxel_colors = np.zeros((grid_size[0], grid_size[1], grid_size[2], 3), dtype=float)
    voxel_materials = np.full(grid_size, BLOCK_IDS["plastic"], dtype=object)
    print(f"Voxel grid created in {time() - start_time:.4f} seconds.")
    stats["voxel_grid_shape"] = grid_size.tolist()
    return voxel_grid, voxel_colors, voxel_materials, min_grid_coords

# Function to check if a triangle is intersecting a voxel
def is_triangle_intersecting_voxel(triangle_points, voxel_corner):
    # Move triangle to the voxel's coordinate system where the voxel is centered at the origin
    box_center = [voxel_corner[0] + 0.5,
                  voxel_corner[1] + 0.5,
                  voxel_corner[2] + 0.5]
    
    V0 = [triangle_points[0][i] - box_center[i] for i in range(3)]
    V1 = [triangle_points[1][i] - box_center[i] for i in range(3)]
    V2 = [triangle_points[2][i] - box_center[i] for i in range(3)]

    # Edges of the triangle
    E0 = [V1[i] - V0[i] for i in range(3)]
    E1 = [V2[i] - V1[i] for i in range(3)]
    E2 = [V0[i] - V2[i] for i in range(3)]

    # Box half-size
    box_half_size = [0.5, 0.5, 0.5]

    # Test 1: AABB overlap test between the triangle and the voxel
    min_v = [min(V0[i], V1[i], V2[i]) for i in range(3)]
    max_v = [max(V0[i], V1[i], V2[i]) for i in range(3)]
    for i in range(3):
        if min_v[i] > box_half_size[i] or max_v[i] < -box_half_size[i]:
            return False  # No overlap, so no intersection

    # Test 2: Plane-box overlap test
    normal = cross_product(E0, E1)
    if not plane_box_overlap(normal, V0, box_half_size):
        return False  # The triangle is completely outside the voxel

    # Test 3: Separating Axis Theorem for triangle edges and voxel axes
    # Edge vectors and their absolute values
    edges = [E0, E1, E2]
    for edge in edges:
        fex = abs(edge[0])
        fey = abs(edge[1])
        fez = abs(edge[2])

        # Test axis L = edge x A0
        if not axis_test_X(edge, V0, V1, V2, fey, fez, box_half_size):
            return False

        # Test axis L = edge x A1
        if not axis_test_Y(edge, V0, V1, V2, fex, fez, box_half_size):
            return False

        # Test axis L = edge x A2
        if not axis_test_Z(edge, V0, V1, V2, fex, fey, box_half_size):
            return False

    return True  # No separating axis found; the triangle intersects the voxel

# Math functions for intersection tests
def cross_product(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]
def dot_product(a, b):
    return sum(a[i]*b[i] for i in range(3))

# Function to check if a plane and box overlap
def plane_box_overlap(normal, vert, maxbox):
    vmin = [0.0]*3
    vmax = [0.0]*3
    for q in range(3):
        if normal[q] > 0.0:
            vmin[q] = -maxbox[q] - vert[q]
            vmax[q] = maxbox[q] - vert[q]
        else:
            vmin[q] = maxbox[q] - vert[q]
            vmax[q] = -maxbox[q] - vert[q]
    if dot_product(normal, vmin) > 0.0:
        return False
    if dot_product(normal, vmax) >= 0.0:
        return True
    return False

# Functions for the Separating Axis Theorem (SAT) tests
def axis_test_X(edge, V0, V1, V2, fey, fez, box_half_size):
    p0 = edge[2]*V0[1] - edge[1]*V0[2]
    p1 = edge[2]*V1[1] - edge[1]*V1[2]
    p2 = edge[2]*V2[1] - edge[1]*V2[2]
    min_p = min(p0, p1, p2)
    max_p = max(p0, p1, p2)
    rad = fez * box_half_size[1] + fey * box_half_size[2]
    return not (min_p > rad or max_p < -rad)
def axis_test_Y(edge, V0, V1, V2, fex, fez, box_half_size):
    p0 = -edge[2]*V0[0] + edge[0]*V0[2]
    p1 = -edge[2]*V1[0] + edge[0]*V1[2]
    p2 = -edge[2]*V2[0] + edge[0]*V2[2]
    min_p = min(p0, p1, p2)
    max_p = max(p0, p1, p2)
    rad = fez * box_half_size[0] + fex * box_half_size[2]
    return not (min_p > rad or max_p < -rad)
def axis_test_Z(edge, V0, V1, V2, fex, fey, box_half_size):
    p0 = edge[1]*V0[0] - edge[0]*V0[1]
    p1 = edge[1]*V1[0] - edge[0]*V1[1]
    p2 = edge[1]*V2[0] - edge[0]*V2[1]
    min_p = min(p0, p1, p2)
    max_p = max(p0, p1, p2)
    rad = fey * box_half_size[0] + fex * box_half_size[1]
    return not (min_p > rad or max_p < -rad)

# Function to mark voxels based on triangle-voxel intersection
def mark_voxels(voxel_grid, voxel_colors, voxel_materials, min_grid_coords, voxel_scale, vertices, faces, face_materials):
    import multiprocessing as mp

    print("Marking voxels...")
    start_time = time()

    # Prepare arguments for parallel processing
    print("Preparing triangles for parallel processing...")
    args_list = []
    voxel_grid_shape = voxel_grid.shape[:3]

    for i, face in enumerate(faces):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        material = face_materials[i]
        args_list.append((i, v0, v1, v2, material, voxel_scale, min_grid_coords, voxel_grid_shape))

    print(f"Triangles prepared in {time() - start_time:.4f} seconds.")

    # Limit CPU usage by controlling the number of worker processes
    start_time = time()
    print("Marking voxels in parallel...")
    num_processes = max(1, mp.cpu_count() * 3 // 4)
    total_faces = len(faces)
    optimal_chunksize = max(1, min(total_faces // num_processes // 10, 10))
    print(f"Marking voxels in parallel (Chunksize: {optimal_chunksize})...")
    with mp.Pool(processes=num_processes) as pool:
        results = []
        for result in tqdm(pool.imap(mark_voxels_for_triangle, args_list, chunksize=optimal_chunksize), total=total_faces):
            results.append(result)
    print(f"Voxels marked in {time() - start_time:.4f} seconds.")

    # Apply the results to the voxel grid
    print("Applying marks to the voxel grid...")
    for voxel_indices, material in results:
        for x_idx, y_idx, z_idx in voxel_indices:
            # Avoid overwriting voxels already marked
            if voxel_grid[x_idx, y_idx, z_idx] == 0:
                voxel_grid[x_idx, y_idx, z_idx] = 1
                voxel_colors[x_idx, y_idx, z_idx, :] = material["color"]
                voxel_materials[x_idx, y_idx, z_idx] = material["shapeId"]

    stats["num_voxels"] = np.sum(voxel_grid)

    return voxel_grid, voxel_colors, voxel_materials

def mark_voxels_for_triangle(args):
    face_index, v0, v1, v2, material, voxel_scale, min_grid_coords, voxel_grid_shape = args

    # Calculate the axis-aligned bounding box (AABB) of the triangle
    min_coords = np.min([v0, v1, v2], axis=0)
    max_coords = np.max([v0, v1, v2], axis=0)

    # Convert AABB coordinates to voxel indices
    min_voxel_coords = (min_coords - min_grid_coords * voxel_scale) / voxel_scale
    max_voxel_coords = (max_coords - min_grid_coords * voxel_scale) / voxel_scale
    min_voxel_indices = np.floor(min_voxel_coords).astype(int)
    max_voxel_indices = np.ceil(max_voxel_coords).astype(int)

    # Clip indices to voxel grid size
    min_voxel_indices = np.maximum(min_voxel_indices, 0)
    max_voxel_indices = np.minimum(max_voxel_indices, np.array(voxel_grid_shape) - 1)

    voxel_indices = []
    # Loop over the voxels in the triangle's AABB
    for x in range(min_voxel_indices[0], max_voxel_indices[0]+1):
        for y in range(min_voxel_indices[1], max_voxel_indices[1]+1):
            for z in range(min_voxel_indices[2], max_voxel_indices[2]+1):
                voxel_corner = (min_grid_coords + np.array([x, y, z])) * voxel_scale
                triangle_points = [v0, v1, v2]
                if is_triangle_intersecting_voxel(triangle_points, voxel_corner):
                    voxel_indices.append((x, y, z))

    return voxel_indices, material

# Function to Detect Prism in the Voxel Grid
def detect_prism(voxel_grid, voxel_colors, voxel_materials, x, y, z):
    max_x, max_y, max_z = len(voxel_grid), len(voxel_grid[0]), len(voxel_grid[0][0])
    prism_color = voxel_colors[x, y, z]
    prism_material = voxel_materials[x, y, z]  # GET MATERIAL FOR THIS VOXEL
    
    # Start with the smallest possible prism
    bounds_x, bounds_y, bounds_z = 1, 1, 1

    # Expand the prism in the x direction
    while x + bounds_x < max_x and all(
        voxel_grid[x + bounds_x][y + i][z + j] and  # First check if it's not null (1 or 2)
        (voxel_grid[x + bounds_x][y + i][z + j] == 2 or  # Check if it's 2 (filled)
         (np.array_equal(voxel_colors[x + bounds_x][y + i][z + j], prism_color) and  # Or the color matches
          voxel_materials[x + bounds_x][y + i][z + j] == prism_material))  # And the material matches
        for i in range(bounds_y) for j in range(bounds_z)
    ):
        bounds_x += 1

    # Expand the prism in the y direction
    while y + bounds_y < max_y and all(
        voxel_grid[x + i][y + bounds_y][z + j] and  # First check if it's not null (1 or 2)
        (voxel_grid[x + i][y + bounds_y][z + j] == 2 or  # Check if it's 2 (filled)
         (np.array_equal(voxel_colors[x + i][y + bounds_y][z + j], prism_color) and  # Or the color matches
          voxel_materials[x + i][y + bounds_y][z + j] == prism_material))  # And the material matches
        for i in range(bounds_x) for j in range(bounds_z)
    ):
        bounds_y += 1

    # Expand the prism in the z direction
    while z + bounds_z < max_z and all(
        voxel_grid[x + i][y + j][z + bounds_z] and  # First check if it's not null (1 or 2)
        (voxel_grid[x + i][y + j][z + bounds_z] == 2 or  # Check if it's 2 (filled)
         (np.array_equal(voxel_colors[x + i][y + j][z + bounds_z], prism_color) and  # Or the color matches
          voxel_materials[x + i][y + j][z + bounds_z] == prism_material))  # And the material matches
        for i in range(bounds_x) for j in range(bounds_y)
    ):
        bounds_z += 1

    # Mark the voxels within this prism as processed
    for i in range(bounds_x):
        for j in range(bounds_y):
            for k in range(bounds_z):
                voxel_grid[x + i][y + j][z + k] = 0  # Mark as processed

    return {
        "bounds": {
            "x": bounds_x,
            "y": bounds_y,
            "z": bounds_z
        },
        "color": f"{int(prism_color[0]*255):02x}{int(prism_color[1]*255):02x}{int(prism_color[2]*255):02x}",  # OUTPUT COLOR IN HEX
        "pos": {
            "x": x,
            "y": y,
            "z": z
        },
        "shapeId": prism_material,  # USE THE MATERIAL SHAPE ID
        "xaxis": 1,
        "zaxis": 3
    }

# Function to ensure the mesh is watertight
def ensure_watertight(mesh):
    if not mesh.is_watertight:
        filled_mesh = mesh.fill_holes()
        if not isinstance(filled_mesh, trimesh.Trimesh):
            print(f"Mesh is not watertight after fill holes. Attempting repair... ({type(filled_mesh)} detected, value: {filled_mesh})")
            filled_mesh = trimesh.repair.fill_holes(mesh)
            if not isinstance(filled_mesh, trimesh.Trimesh):
                raise ValueError("Mesh is not valid after repair.")
        else:
            mesh = filled_mesh
            mesh.remove_unreferenced_vertices()
        if not mesh.is_watertight:
            raise ValueError("Mesh is not watertight after repair.")
    return mesh

# Function to get ray directions for each point
def get_ray_directions(point, grid_min, grid_size, voxel_scale):
    # Determine the closest boundary side for each point
    directions = np.zeros((point.shape[0], 3))
    
    for i in range(point.shape[0]):
        x, y, z = point[i]
        distances = {
            'x_min': abs(x - grid_min[0]),
            'x_max': abs(x - (grid_min[0] + grid_size[0] * voxel_scale)),
            'y_min': abs(y - grid_min[1]),
            'y_max': abs(y - (grid_min[1] + grid_size[1] * voxel_scale)),
            'z_min': abs(z - grid_min[2]),
            'z_max': abs(z - (grid_min[2] + grid_size[2] * voxel_scale))
        }
        closest_side = min(distances, key=distances.get)
        if 'x_min' in closest_side:
            directions[i] = [-1, 0, 0]
        elif 'x_max' in closest_side:
            directions[i] = [1, 0, 0]
        elif 'y_min' in closest_side:
            directions[i] = [0, -1, 0]
        elif 'y_max' in closest_side:
            directions[i] = [0, 1, 0]
        elif 'z_min' in closest_side:
            directions[i] = [0, 0, -1]
        elif 'z_max' in closest_side:
            directions[i] = [0, 0, 1]
    return directions

# Function to perform ray casting for voxels
def ray_cast_voxels(mesh, points, directions):
    # Perform ray intersections
    locations, index_ray, index_tri = mesh.ray.intersects_location(points, directions)
    
    # Count intersections per ray
    counts = np.bincount(index_ray, minlength=len(points))
    
    # Determine filled voxels: odd number of intersections
    filled = counts % 2 == 1
    return filled

# Function to classify voxels based on Ray Casting
def classify_voxels_raycast(mesh, voxel_grid, grid_min, grid_size, voxel_scale):
    print("Classifying voxels based on Ray Casting...")
    start_time = time()
    
    # Identify candidate voxels (not walls)
    candidate_indices = np.argwhere(voxel_grid == 0)
    if candidate_indices.size == 0:
        return np.zeros_like(voxel_grid, dtype=bool), np.zeros_like(voxel_grid, dtype=bool)
    
    # Compute voxel centers
    voxel_centers = grid_min + (candidate_indices + 0.5) * voxel_scale
    
    # Get ray directions
    ray_directions = get_ray_directions(voxel_centers, grid_min, grid_size, voxel_scale)
    
    # Perform ray casting in batches to manage memory
    batch_size = 100000  # Adjust based on available memory
    filled = np.zeros(candidate_indices.shape[0], dtype=bool)
    
    for i in tqdm(range(0, candidate_indices.shape[0], batch_size), desc="Ray Casting Batches"):
        batch_points = voxel_centers[i:i+batch_size]
        batch_dirs = ray_directions[i:i+batch_size]
        batch_filled = ray_cast_voxels(mesh, batch_points, batch_dirs)
        filled[i:i+batch_size] = batch_filled
    
    # Create filled and unfilled masks
    filled_mask = np.zeros_like(voxel_grid, dtype=bool)
    filled_mask[candidate_indices[filled, 0],
                candidate_indices[filled, 1],
                candidate_indices[filled, 2]] = True
    
    unfilled_mask = np.zeros_like(voxel_grid, dtype=bool)
    unfilled_mask[candidate_indices[~filled, 0],
                  candidate_indices[~filled, 1],
                  candidate_indices[~filled, 2]] = True
    
    print(f"Voxel classification completed in {time() - start_time:.4f} seconds.")
    stats["voxel_classification_time"] = time() - start_time
    return filled_mask, unfilled_mask

# Function to get closed mesh components with volume check
def get_closed_mesh_components(mesh, min_volume=1e-6):
    # Split mesh into connected components, only watertight if only_watertight=True
    connected_components = mesh.split(only_watertight=True)  # Set to True to get only watertight components
    print(f"Total connected components (watertight): {len(connected_components)}")
    
    # Filter out components below the minimum volume
    closed_components = [comp for comp in connected_components if comp.volume >= min_volume]
    print(f"Watertight components with volume >= {min_volume}: {len(closed_components)}")
    
    # Reconstruct mesh from closed components
    if len(closed_components) == 0:
        raise ValueError("No watertight components with sufficient volume found in the mesh.")
    
    mesh_closed = trimesh.util.concatenate(closed_components)
    return mesh_closed

# Function to Decompose Voxel Grid into Minimal Prisms
def minimal_prism_decomposition(voxel_grid, voxel_colors, voxel_materials):  # ADD voxel_materials PARAMETER
    print("Decomposing voxel grid into minimal prisms...")
    start_time = time()
    prisms = []
    for x in range(len(voxel_grid)):
        for y in range(len(voxel_grid[0])):
            for z in range(len(voxel_grid[0][0])):
                if voxel_grid[x][y][z]:
                    prism = detect_prism(voxel_grid, voxel_colors, voxel_materials, x, y, z)  # PASS voxel_materials
                    prisms.append(prism)
    
    stats["num_minimal_prisms"] = len(prisms)

    output = {
        "bodies": [
            {
                "childs": prisms
            }
        ],
        "version": 4
    }
    print(f"Voxel grid decomposed in {time() - start_time:.4f} seconds.")
    return output

# Function to Save Output to a JSON File
def save_to_json_file(data, filename="output.json"):
    print(f"Saving output to {filename}...")
    start_time = time()
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Output saved in {time() - start_time:.4f} seconds.")

# Main function
def main(
    input_file,
    output_file,
    voxel_scale,
    obj_scale,
    obj_offset,
    rotate_axis,
    rotate_angle,
    use_set_color,
    set_color,
    use_set_block,
    set_block,
    use_scrap_colors,
    vary_colors,
    interior_fill
):
    script_start_time = time()
    print("Starting main process.")

    # Parse the OBJ file and get vertices, faces, and face materials
    vertices, faces, face_materials = parse_obj_file(input_file, obj_scale, obj_offset, use_scrap_colors, use_set_color, set_color, use_set_block, set_block)
    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of faces: {len(faces)}")

    # Rotate vertices
    rotation_matrix = create_rotation_matrix(rotate_axis, rotate_angle)
    vertices = rotate_vertices(vertices, rotation_matrix)

    # Calculate the bounding box for the model
    min_coords, max_coords = calculate_model_bounding_box(vertices)
    print(f"Bounding box min coordinates: {min_coords}")
    print(f"Bounding box max coordinates: {max_coords}")

    # Create the voxel grid
    voxel_grid, voxel_colors, voxel_materials, min_grid_coords = create_voxel_grid(min_coords, max_coords, voxel_scale)
    print(f"Voxel grid shape: {voxel_grid.shape}")

    # Mark voxels based on triangle-voxel intersection
    voxel_grid, voxel_colors, voxel_materials = mark_voxels(
        voxel_grid, voxel_colors, voxel_materials, min_grid_coords, voxel_scale,
        vertices, faces, face_materials)
    print(f"Number of filled voxels (walls): {stats['num_voxels']}")

    # Interior filling
    if interior_fill:
        # Load mesh for Ray Casting
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Get closed mesh components
        try:
            mesh_closed = get_closed_mesh_components(mesh)
        except ValueError as e:
            print(f"Error: {e}")
            print("Skipping interior filling due to lack of closed components.")
            mesh_closed = None

        if mesh_closed:
            # Classify voxels using Ray Casting on closed mesh
            filled, unfilled = classify_voxels_raycast(mesh_closed, voxel_grid, min_grid_coords, voxel_grid.shape, voxel_scale)
            stats["num_filled_voxels"] = np.sum(filled)
            stats["num_unfilled_voxels"] = np.sum(unfilled)
            print(f"Number of filled voxels: {stats['num_filled_voxels']}")
            print(f"Number of unfilled voxels: {stats['num_unfilled_voxels']}")

            # Update voxel states
            voxel_grid[filled] = 2  # Filled state
            if use_set_color:
                voxel_colors[filled] = set_color  # Set color
            else:
                voxel_colors[filled] = FALLBACK_MATERIAL["color"]  # Default color
            if use_set_block:
                voxel_materials[filled] = BLOCK_IDS.get(set_block, FALLBACK_MATERIAL["shapeId"])
            else:
                voxel_materials[filled] = FALLBACK_MATERIAL["shapeId"]  # Default material

            # Optionally, set unfilled voxels to air
            # voxel_grid[unfilled] = 0  # Unfilled state (air)
            # voxel_colors[unfilled] = [0, 0, 0]  # Air color
            # voxel_materials[unfilled] = None

    # Perform minimal prism decomposition
    output = minimal_prism_decomposition(voxel_grid, voxel_colors, voxel_materials)
    print(f"Number of minimal prisms: {len(output['bodies'][0]['childs'])}")

    # Save the output to a JSON file
    save_to_json_file(output, output_file)
    print(f"Output saved to '{output_file}'")

    total_execution_time = time() - script_start_time
    print(f"Total execution time: {total_execution_time:.4f} seconds")

    # Print stats
    print("\nRuntime Stats:")
    for key, value in stats.items():
        print(f" - {key.replace('_', ' ').title()}: {value}")
    
    print(f"Main process completed in {total_execution_time:.4f} seconds.")
    return total_execution_time

if __name__ == "__main__":
    # If launching this script directly, use the default configuration values
    main(
        input_file=CONFIG_VALUES["input_file"],
        output_file=CONFIG_VALUES["output_file"],
        voxel_scale=CONFIG_VALUES["voxel_scale"],
        obj_scale=CONFIG_VALUES["obj_scale"],
        obj_offset=CONFIG_VALUES["obj_offset"],
        rotate_axis=CONFIG_VALUES["rotate_axis"],
        rotate_angle=CONFIG_VALUES["rotate_angle"],
        use_set_color=CONFIG_VALUES["use_set_color"],
        set_color=CONFIG_VALUES["set_color"],
        use_set_block=CONFIG_VALUES["use_set_block"],
        set_block=CONFIG_VALUES["set_block"],
        use_scrap_colors=CONFIG_VALUES["use_scrap_colors"],
        vary_colors=CONFIG_VALUES["vary_colors"],
        interior_fill=CONFIG_VALUES["interior_fill"]
    )
