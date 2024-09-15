import numpy as np
import json
import multiprocessing as mp
from tqdm import tqdm
from itertools import product
from scipy.spatial import KDTree
from time import time
import trimesh

INPUT_FILE = '..\\testStuff.obj'
OUTPUT_FILE = 'C:\\Users\\Corey\\AppData\\Roaming\\Axolot Games\\Scrap Mechanic\\User\\User_76561198805744844\\Blueprints\\51c6485e-c45d-47b0-a3c9-08d2db23aef9\\blueprint.json'

# Scaling constants
VOXEL_SCALE = 1.0 / 1.0  # Voxel grid resolution
OBJ_SCALE = 4 * VOXEL_SCALE  # Scaling factor for the model itself
OBJ_OFFSET = np.array([0.0, 0.0, 0.0])  # Offset for the model

# Rotation constants
ROTATE_AXIS = 'x'  # Axis to rotate around ('x', 'y', 'z')
ROTATE_ANGLE = 90  # Rotation angle in degrees

# Options
USE_ONE_COLOR = False  # Use a single color for all voxels. False to use the mesh colors, 3 decimals otherwise (e.g. [1.0, 0.0, 0.0]). True to use the default color.
USE_ONE_MATERIAL = False  # Use a single material for all voxels. False to use the mesh materials, name otherwise (e.g. "plastic"). True to use the default material.
USE_SCRAP_COLORS = True  # Use the original Scrap Mechanic colors (converts all colors to the closest match)
VARY_COLORS = False  # Randomly vary the colors of the voxels
INTERIOR_FILL = False  # Fill the mesh interior

# Mapping of block names to UUIDs
BLOCK_IDS = {
    "scrapwood": "1fc74a28-addb-451a-878d-c3c605d63811",
    "wood1": "df953d9c-234f-4ac2-af5e-f0490b223e71",
    "wood2": "1897ee42-0291-43e4-9645-8c5a5d310398",
    "wood3": "061b5d4b-0a6a-4212-b0ae-9e9681f1cbfb",
    "scrapmetal": "1f7ac0bb-ad45-4246-9817-59bdf7f7ab39",
    "metal1": "8aedf6c2-94e1-4506-89d4-a0227c552f1e",
    "metal2": "1016cafc-9f6b-40c9-8713-9019d399783f",
    "metal3": "c0dfdea5-a39d-433a-b94a-299345a5df46",
    "scrapstone": "30a2288b-e88e-4a92-a916-1edbfc2b2dac",
    "concrete1": "a6c6ce30-dd47-4587-b475-085d55c6a3b4",
    "concrete2": "ff234e42-5da4-43cc-8893-940547c97882",
    "concrete3": "e281599c-2343-4c86-886e-b2c1444e8810",
    "cardboard": "f0cba95b-2dc4-4492-8fd9-36546a4cb5aa",
    "sand": "c56700d9-bbe5-4b17-95ed-cef05bd8be1b",
    "plastic": "628b2d61-5ceb-43e9-8334-a4135566df7a",
    "glass": "5f41af56-df4c-4837-9b3c-10781335757f",
    "glasstile": "749f69e0-56c9-488c-adf6-66c58531818f",
    "armoredglass": "b5ee5539-75a2-4fef-873b-ef7c9398b3f5",
    "bubblewrap": "f406bf6e-9fd5-4aa0-97c1-0b3c2118198e",
    "restroom": "920b40c8-6dfc-42e7-84e1-d7e7e73128f6",
    "tiles": "8ca49bff-eeef-4b43-abd0-b527a567f1b7",
    "bricks": "0603b36e-0bdb-4828-b90c-ff19abcdfe34",
    "lights": "073f92af-f37e-4aff-96b3-d66284d5081c",
    "caution": "09ca2713-28ee-4119-9622-e85490034758",
    "crackedconcrete": "f5ceb7e3-5576-41d2-82d2-29860cf6e20e",
    "concretetiles": "cd0eff89-b693-40ee-bd4c-3500b23df44e",
    "metalbricks": "220b201e-aa40-4995-96c8-e6007af160de",
    "beam": "25a5ffe7-11b1-4d3e-8d7a-48129cbaf05e",
    "insulation": "9be6047c-3d44-44db-b4b9-9bcf8a9aab20",
    "drywall": "b145d9ae-4966-4af6-9497-8fca33f9aee3",
    "carpet": "febce8a6-6c05-4e5d-803b-dfa930286944",
    "plasticwall": "e981c337-1c8a-449c-8602-1dd990cbba3a",
    "metalnet": "4aa2a6f0-65a4-42e3-bf96-7dec62570e0b",
    "crossnet": "3d0b7a6e-5b40-474c-bbaf-efaa54890e6a",
    "tryponet": "ea6864db-bb4f-4a89-b9ec-977849b6713a",
    "stripednet": "a479066d-4b03-46b5-8437-e99fec3f43ee",
    "squarenet": "b4fa180c-2111-4339-b6fd-aed900b57093",
    "spaceshipmetal": "027bd4ec-b16d-47d2-8756-e18dc2af3eb6",
    "spaceshipfloor": "4ad97d49-c8a5-47f3-ace3-d56ba3affe50",
    "treadplate": "f7d4bfed-1093-49b9-be32-394c872a1ef4",
    "warehousefloor": "3e3242e4-1791-4f70-8d1d-0ae9ba3ee94c",
    "wornmetal": "d740a27d-cc0f-4866-9e07-6a5c516ad719",
    "framework": "c4a2ffa8-c245-41fb-9496-966c6ee4648b",
    "challenge01": "491b1d4f-3a00-403e-b64f-f9eb7bda7683",
    "challenge02": "cad3a585-2686-40e2-8eb1-02f5df20a021",
    "challengeglass": "17baf3ba-0b40-4eef-9823-119059d5c12d"
}

# Scrap Mechanic colors
SM_COLORS = [
    "EEEEEE", "7F7F7F", "4A4A4A", "222222", "F5F071", "E2DB13", "817C00", "323000",
    "CBF66F", "A0EA00", "577D07", "375000", "064023", "0E8031", "19E753", "68FF88",
    "7EEDED", "2CE6E6", "118787", "0F2E91", "0A1D5A", "0A4444", "0A3EE2", "4C6FE3",
    "AE79F0", "7514ED", "500AA6", "35086C", "472800", "520653", "560202", "673B00",
    "DF7F00", "EEAF5C", "EE7BF0", "F06767", "CF11D2", "720A74", "7C0000", "D02525"
]
SM_COLORS_RGB = [np.array([int(c[i:i+2], 16) / 255 for i in range(0, 6, 2)]) for c in SM_COLORS]
SM_COLORS_KDTREE = KDTree(SM_COLORS_RGB)

DEFAULT_MATERIAL = {"color": [0.0, 0.588235, 1.0], "shapeId": BLOCK_IDS["wood1"]}  # Default to blue plastic

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
def parse_mtl_file(filepath):
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
                        if USE_SCRAP_COLORS:  # Use the closest Scrap Mechanic color
                            materials[current_material_full]["color"] = find_closest_color(color)
                            print(f"Color {color} replaced with {materials[current_material_full]['color']} for material {current_material_full}")
                            
                        print(f"Material parsed: {current_material_full} with color {color} and shapeId {shape_id}")

                        stats["num_materials"] = len(materials)
        return materials
    except FileNotFoundError:
        print("MTL file not found. Using default color.")
        return materials

# Function to parse the OBJ file and extract vertices, faces, and face materials
def parse_obj_file(filepath):
    vertices = []
    faces = []
    face_materials = []  # List for face materials
    current_material_full = None

    # If using one color AND one material, skip the MTL file parsing
    if USE_ONE_COLOR is False or USE_ONE_MATERIAL is False:
        print("Parsing MTL file...")
        mtl_filepath = filepath.replace('.obj', '.mtl')
        materials = parse_mtl_file(mtl_filepath)
    else:
        print("Not using MTL file.")
        materials = {}

    with open(filepath, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                # First add the OBJ_OFFSET to the values, then scale the vertices with OBJ_SCALE
                vertices.append((np.array([float(parts[i]) for i in range(1, 4)]) + OBJ_OFFSET) * OBJ_SCALE)
            elif line.startswith('usemtl') and (USE_ONE_COLOR is False or USE_ONE_MATERIAL is False):
                current_material_full = line.split()[1]  # Use the full name for retrieval
                print(f"Using material: {current_material_full}")
            elif line.startswith('f '):
                parts = line.split()
                face = [int(idx.split('/')[0]) - 1 for idx in parts[1:]]
                
                # If no material is specified, use the default material
                if current_material_full is None:
                    material = DEFAULT_MATERIAL
                else:
                    material = materials.get(current_material_full, DEFAULT_MATERIAL)

                # If USE_ONE_COLOR is True, use the default color, else if is list, use that color
                if USE_ONE_COLOR is True:
                    material["color"] = DEFAULT_MATERIAL["color"]
                elif isinstance(USE_ONE_COLOR, list) and len(USE_ONE_COLOR) == 3:
                    material["color"] = USE_ONE_COLOR
                
                # If USE_ONE_MATERIAL is True, use the default material, else if is string, use that material if found in BLOCK_IDS
                if USE_ONE_MATERIAL is True:
                    material["shapeId"] = DEFAULT_MATERIAL["shapeId"]
                elif isinstance(USE_ONE_MATERIAL, str):
                    material["shapeId"] = BLOCK_IDS.get(USE_ONE_MATERIAL, DEFAULT_MATERIAL["shapeId"])

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
def main():
    script_start_time = time()
    print("Starting main process.")

    # Parse the OBJ file and get vertices, faces, and face materials
    vertices, faces, face_materials = parse_obj_file(INPUT_FILE)
    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of faces: {len(faces)}")

    # Rotate vertices
    rotation_matrix = create_rotation_matrix(ROTATE_AXIS, ROTATE_ANGLE)
    vertices = rotate_vertices(vertices, rotation_matrix)

    # Calculate the bounding box for the model
    min_coords, max_coords = calculate_model_bounding_box(vertices)
    print(f"Bounding box min coordinates: {min_coords}")
    print(f"Bounding box max coordinates: {max_coords}")

    # Create the voxel grid
    voxel_grid, voxel_colors, voxel_materials, min_grid_coords = create_voxel_grid(min_coords, max_coords, VOXEL_SCALE)
    print(f"Voxel grid shape: {voxel_grid.shape}")

    # Mark voxels based on triangle-voxel intersection
    voxel_grid, voxel_colors, voxel_materials = mark_voxels(
        voxel_grid, voxel_colors, voxel_materials, min_grid_coords, VOXEL_SCALE,
        vertices, faces, face_materials)
    print(f"Number of filled voxels (walls): {stats['num_voxels']}")

    # Interior filling
    if INTERIOR_FILL:
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
            filled, unfilled = classify_voxels_raycast(mesh_closed, voxel_grid, min_grid_coords, voxel_grid.shape, VOXEL_SCALE)
            stats["num_filled_voxels"] = np.sum(filled)
            stats["num_unfilled_voxels"] = np.sum(unfilled)
            print(f"Number of filled voxels: {stats['num_filled_voxels']}")
            print(f"Number of unfilled voxels: {stats['num_unfilled_voxels']}")

            # Update voxel states
            voxel_grid[filled] = 2  # Filled state
            if isinstance(USE_ONE_COLOR, list) and len(USE_ONE_COLOR) == 3:
                voxel_colors[filled] = USE_ONE_COLOR  # Use the specified color
            else:
                voxel_colors[filled] = DEFAULT_MATERIAL["color"]  # Default color
            if isinstance(USE_ONE_MATERIAL, str):
                voxel_materials[filled] = BLOCK_IDS.get(USE_ONE_MATERIAL, DEFAULT_MATERIAL["shapeId"])
            else:
                voxel_materials[filled] = DEFAULT_MATERIAL["shapeId"]  # Default material

            # Optionally, set unfilled voxels to air
            # voxel_grid[unfilled] = 0  # Unfilled state (air)
            # voxel_colors[unfilled] = [0, 0, 0]  # Air color
            # voxel_materials[unfilled] = None

    # Perform minimal prism decomposition
    output = minimal_prism_decomposition(voxel_grid, voxel_colors, voxel_materials)
    print(f"Number of minimal prisms: {len(output['bodies'][0]['childs'])}")

    # Save the output to a JSON file
    save_to_json_file(output, OUTPUT_FILE)
    print(f"Output saved to '{OUTPUT_FILE}'")

    total_execution_time = time() - script_start_time
    print(f"Total execution time: {total_execution_time:.4f} seconds")

    # Print stats
    print("\nRuntime Stats:")
    for key, value in stats.items():
        print(f" - {key.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()
