#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path


def load_scene(filepath):
    """
    Load scene data from a JSON file.
    
    Args:
        filepath: Path to the JSON scene file
        
    Returns:
        Dictionary containing parsed scene data with keys:
        'vertices', 'faces', 'material', 'lighting', 'camera'
    """
    try:
        with open(filepath, 'r') as f:
            scene_data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Scene file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in scene file: {e}")
    
    # Parse all components
    vertices = parse_vertices(scene_data)
    faces = parse_faces(scene_data)
    material = parse_material(scene_data)
    lighting = parse_lighting(scene_data)
    camera = parse_camera(scene_data)
    
    return {
        'vertices': vertices,
        'faces': faces,
        'material': material,
        'lighting': lighting,
        'camera': camera
    }


def parse_vertices(scene_data):
    """
    Extract vertex array from scene data.
    
    Args:
        scene_data: Dictionary containing scene JSON data
        
    Returns:
        Numpy array of vertices with shape (N, 3)
    """
    if 'vertices' not in scene_data:
        raise KeyError("Scene data missing 'vertices' field")
    
    vertices = scene_data['vertices']
    if not vertices:
        raise ValueError("Vertices array is empty")
    
    # Convert to numpy array and ensure correct shape
    vertices_array = np.array(vertices, dtype=np.float32)
    
    if len(vertices_array.shape) != 2 or vertices_array.shape[1] != 3:
        raise ValueError(f"Vertices must be Nx3 array, got shape {vertices_array.shape}")
    
    return vertices_array


def parse_faces(scene_data):
    """
    Extract triangle indices from scene data.
    
    Args:
        scene_data: Dictionary containing scene JSON data
        
    Returns:
        Numpy array of triangle indices with shape (M, 3)
    """
    if 'faces' not in scene_data:
        raise KeyError("Scene data missing 'faces' field")
    
    faces = scene_data['faces']
    if not faces:
        raise ValueError("Faces array is empty")
    
    # Convert to numpy array and ensure correct shape
    faces_array = np.array(faces, dtype=np.int32)
    
    if len(faces_array.shape) != 2 or faces_array.shape[1] != 3:
        raise ValueError(f"Faces must be Mx3 array, got shape {faces_array.shape}")
    
    return faces_array


def parse_material(scene_data):
    """
    Extract material properties from scene data.
    
    Args:
        scene_data: Dictionary containing scene JSON data
        
    Returns:
        Dictionary with material properties:
        'diffuse_color', 'ambient', 'specular', 'shininess'
    """
    if 'material' not in scene_data:
        raise KeyError("Scene data missing 'material' field")
    
    material = scene_data['material']
    
    # Extract diffuse color
    if 'diffuse_color' not in material:
        raise KeyError("Material missing 'diffuse_color'")
    diffuse_color = np.array(material['diffuse_color'], dtype=np.float32)
    if diffuse_color.shape != (3,):
        raise ValueError(f"diffuse_color must be length 3, got {diffuse_color.shape}")
    
    # Extract ambient coefficient
    if 'ambient' not in material:
        raise KeyError("Material missing 'ambient'")
    ambient = float(material['ambient'])
    
    # Extract specular coefficient
    if 'specular' not in material:
        raise KeyError("Material missing 'specular'")
    specular = float(material['specular'])
    
    # Extract shininess
    if 'shininess' not in material:
        raise KeyError("Material missing 'shininess'")
    shininess = float(material['shininess'])
    
    return {
        'diffuse_color': diffuse_color,
        'ambient': ambient,
        'specular': specular,
        'shininess': shininess
    }


def parse_lighting(scene_data):
    """
    Extract lighting information from scene data.
    
    Args:
        scene_data: Dictionary containing scene JSON data
        
    Returns:
        Dictionary with lighting information:
        'lights' (list of light dicts), 'ambient_light' (RGB array)
    """
    if 'lighting' not in scene_data:
        raise KeyError("Scene data missing 'lighting' field")
    
    lighting = scene_data['lighting']
    
    # Extract ambient light
    if 'ambient_light' not in lighting:
        raise KeyError("Lighting missing 'ambient_light'")
    ambient_light = np.array(lighting['ambient_light'], dtype=np.float32)
    if ambient_light.shape != (3,):
        raise ValueError(f"ambient_light must be length 3, got {ambient_light.shape}")
    
    # Extract lights array
    if 'lights' not in lighting:
        raise KeyError("Lighting missing 'lights'")
    
    lights = []
    for light_data in lighting['lights']:
        if 'position' not in light_data:
            raise KeyError("Light missing 'position'")
        if 'color' not in light_data:
            raise KeyError("Light missing 'color'")
        if 'intensity' not in light_data:
            raise KeyError("Light missing 'intensity'")
        
        light = {
            'position': np.array(light_data['position'], dtype=np.float32),
            'color': np.array(light_data['color'], dtype=np.float32),
            'intensity': float(light_data['intensity'])
        }
        lights.append(light)
    
    return {
        'lights': lights,
        'ambient_light': ambient_light
    }


def parse_camera(scene_data):
    """
    Extract camera parameters from scene data.
    
    Args:
        scene_data: Dictionary containing scene JSON data
        
    Returns:
        Dictionary with camera parameters:
        'position', 'look_at', 'up', 'fov', 'aspect_ratio', 'near', 'far'
    """
    if 'camera' not in scene_data:
        raise KeyError("Scene data missing 'camera' field")
    
    camera = scene_data['camera']
    
    # Extract position
    if 'position' not in camera:
        raise KeyError("Camera missing 'position'")
    position = np.array(camera['position'], dtype=np.float32)
    
    # Extract look_at
    if 'look_at' not in camera:
        raise KeyError("Camera missing 'look_at'")
    look_at = np.array(camera['look_at'], dtype=np.float32)
    
    # Extract up vector
    if 'up' not in camera:
        raise KeyError("Camera missing 'up'")
    up = np.array(camera['up'], dtype=np.float32)
    
    # Extract field of view
    if 'fov' not in camera:
        raise KeyError("Camera missing 'fov'")
    fov = float(camera['fov'])
    
    # Extract aspect ratio
    if 'aspect_ratio' not in camera:
        raise KeyError("Camera missing 'aspect_ratio'")
    aspect_ratio = float(camera['aspect_ratio'])
    
    # Extract near plane
    if 'near' not in camera:
        raise KeyError("Camera missing 'near'")
    near = float(camera['near'])
    
    # Extract far plane
    if 'far' not in camera:
        raise KeyError("Camera missing 'far'")
    far = float(camera['far'])
    
    return {
        'position': position,
        'look_at': look_at,
        'up': up,
        'fov': fov,
        'aspect_ratio': aspect_ratio,
        'near': near,
        'far': far
    }