#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Dict, Any

def create_view_matrix(camera_pos: np.ndarray, look_at: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Create a view matrix for transforming world coordinates to camera space.
    
    Args:
        camera_pos: Camera position in world space (3D vector)
        look_at: Point the camera is looking at (3D vector)
        up: Up direction vector (3D vector)
    
    Returns:
        4x4 view matrix
    """
    # Convert inputs to numpy arrays
    camera_pos = np.array(camera_pos, dtype=np.float32)
    look_at = np.array(look_at, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    # Calculate forward vector (from camera to look_at)
    forward = look_at - camera_pos
    forward = forward / np.linalg.norm(forward)
    
    # Calculate right vector (cross product of forward and up)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Recalculate up vector to ensure orthogonality
    up_corrected = np.cross(right, forward)
    up_corrected = up_corrected / np.linalg.norm(up_corrected)
    
    # Create view matrix
    view_matrix = np.eye(4, dtype=np.float32)
    view_matrix[0, :3] = right
    view_matrix[1, :3] = up_corrected
    view_matrix[2, :3] = -forward  # Negative because we're looking down -Z in camera space
    
    # Add translation component
    view_matrix[0, 3] = -np.dot(right, camera_pos)
    view_matrix[1, 3] = -np.dot(up_corrected, camera_pos)
    view_matrix[2, 3] = np.dot(forward, camera_pos)
    
    return view_matrix


def create_projection_matrix(fov: float, aspect_ratio: float, near: float, far: float) -> np.ndarray:
    """
    Create a perspective projection matrix.
    
    Args:
        fov: Field of view in degrees
        aspect_ratio: Width / height ratio
        near: Near clipping plane distance
        far: Far clipping plane distance
    
    Returns:
        4x4 perspective projection matrix
    """
    # Convert FOV to radians
    fov_rad = np.radians(fov)
    
    # Calculate projection parameters
    f = 1.0 / np.tan(fov_rad / 2.0)
    
    # Create projection matrix
    proj_matrix = np.zeros((4, 4), dtype=np.float32)
    
    proj_matrix[0, 0] = f / aspect_ratio
    proj_matrix[1, 1] = f
    proj_matrix[2, 2] = (far + near) / (near - far)
    proj_matrix[2, 3] = (2.0 * far * near) / (near - far)
    proj_matrix[3, 2] = -1.0
    
    return proj_matrix


def apply_matrix(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a transformation matrix to a set of vertices.
    
    Args:
        vertices: Nx3 or Nx4 array of vertex positions
        matrix: 4x4 transformation matrix
    
    Returns:
        Transformed vertices (Nx4 with homogeneous coordinates)
    """
    # Ensure vertices are in homogeneous coordinates
    if vertices.shape[1] == 3:
        # Add w component (set to 1 for positions)
        ones = np.ones((vertices.shape[0], 1), dtype=np.float32)
        vertices_homogeneous = np.hstack([vertices, ones])
    else:
        vertices_homogeneous = vertices.copy()
    
    # Apply transformation: matrix @ vertex (column vector)
    # Note: vertices are in rows, so we transpose, multiply, then transpose back
    # BUG INTRODUCED: Wrong order - should be vertices @ matrix.T for row vectors
    transformed = vertices_homogeneous @ matrix
    
    return transformed


def vertex_to_screen(vertices: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Convert vertices from NDC (Normalized Device Coordinates) to screen space.
    
    Args:
        vertices: Nx4 array of vertices in clip space (homogeneous coordinates)
        width: Screen width in pixels
        height: Screen height in pixels
    
    Returns:
        Nx3 array of vertices in screen coordinates (x, y, depth)
    """
    # Perform perspective division to get NDC coordinates
    vertices_ndc = vertices.copy()
    
    # Divide by w component
    w = vertices[:, 3:4]
    w = np.where(w == 0, 1e-10, w)  # Avoid division by zero
    vertices_ndc[:, :3] = vertices[:, :3] / w
    
    # Convert from NDC [-1, 1] to screen coordinates [0, width/height]
    screen_vertices = np.zeros((vertices.shape[0], 3), dtype=np.float32)
    
    # X: [-1, 1] -> [0, width]
    screen_vertices[:, 0] = (vertices_ndc[:, 0] + 1.0) * 0.5 * width
    
    # Y: [-1, 1] -> [0, height] (flip Y axis for screen coordinates)
    screen_vertices[:, 1] = (1.0 - vertices_ndc[:, 1]) * 0.5 * height
    
    # Keep depth value for depth testing
    screen_vertices[:, 2] = vertices_ndc[:, 2]
    
    return screen_vertices


def transform_vertices(vertices: np.ndarray, camera: Dict[str, Any]) -> np.ndarray:
    """
    Transform vertices through the complete vertex shader pipeline.
    
    Args:
        vertices: Nx3 array of vertex positions in world space
        camera: Dictionary containing camera parameters:
                - 'position': Camera position [x, y, z]
                - 'look_at': Point camera is looking at [x, y, z]
                - 'up': Up direction vector [x, y, z]
                - 'fov': Field of view in degrees
                - 'aspect_ratio': Width / height ratio
                - 'near': Near clipping plane
                - 'far': Far clipping plane
    
    Returns:
        Nx4 array of transformed vertices in clip space
    """
    # Extract camera parameters
    camera_pos = np.array(camera.get('position', [0, 0, 5]), dtype=np.float32)
    look_at = np.array(camera.get('look_at', [0, 0, 0]), dtype=np.float32)
    up = np.array(camera.get('up', [0, 1, 0]), dtype=np.float32)
    fov = camera.get('fov', 60.0)
    aspect_ratio = camera.get('aspect_ratio', 1.0)
    near = camera.get('near', 0.1)
    far = camera.get('far', 100.0)
    
    # Convert vertices to numpy array if needed
    vertices = np.array(vertices, dtype=np.float32)
    
    # Create transformation matrices
    view_matrix = create_view_matrix(camera_pos, look_at, up)
    projection_matrix = create_projection_matrix(fov, aspect_ratio, near, far)
    
    # Combine view and projection matrices
    view_projection_matrix = projection_matrix @ view_matrix
    
    # Apply transformations to vertices
    transformed_vertices = apply_matrix(vertices, view_projection_matrix)
    
    return transformed_vertices


def batch_transform_vertices(vertex_data: List[np.ndarray], camera: Dict[str, Any]) -> List[np.ndarray]:
    """
    Transform multiple batches of vertices (useful for complex scenes with multiple objects).
    
    Args:
        vertex_data: List of Nx3 vertex arrays
        camera: Camera parameters dictionary
    
    Returns:
        List of transformed vertex arrays
    """
    transformed_batches = []
    
    for vertices in vertex_data:
        transformed = transform_vertices(vertices, camera)
        transformed_batches.append(transformed)
    
    return transformed_batches


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute per-vertex normals from vertex positions and face indices.
    
    Args:
        vertices: Nx3 array of vertex positions
        faces: Mx3 array of face indices (triangles)
    
    Returns:
        Nx3 array of vertex normals
    """
    normals = np.zeros_like(vertices, dtype=np.float32)
    
    # Compute face normals and accumulate to vertices
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # Compute face normal via cross product
        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normal = np.cross(edge1, edge2)
        
        # Accumulate to each vertex of the face
        normals[face[0]] += face_normal
        normals[face[1]] += face_normal
        normals[face[2]] += face_normal
    
    # Normalize all vertex normals
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    normals = normals / norms
    
    return normals