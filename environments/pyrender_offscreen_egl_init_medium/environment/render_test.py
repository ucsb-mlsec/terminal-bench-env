#!/usr/bin/env python3

import os
import numpy as np
import trimesh
import pyrender
from PIL import Image

def create_textured_cube():
    """Create a cube mesh with a simple texture."""
    # Create cube geometry
    cube_trimesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    
    # Create a simple texture (checkerboard pattern)
    texture_size = 256
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
    square_size = texture_size // 8
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                texture[i*square_size:(i+1)*square_size, 
                       j*square_size:(j+1)*square_size] = [255, 100, 100]
            else:
                texture[i*square_size:(i+1)*square_size, 
                       j*square_size:(j+1)*square_size] = [100, 100, 255]
    
    # Create material with texture
    texture_image = Image.fromarray(texture)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorTexture=texture_image,
        metallicFactor=0.2,
        roughnessFactor=0.8
    )
    
    # Create mesh
    mesh = pyrender.Mesh.from_trimesh(cube_trimesh, material=material)
    return mesh

def main():
    """Main rendering function."""
    # Ensure output directory exists
    output_dir = '/workspace/output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create scene
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    
    # Add textured cube
    cube_mesh = create_textured_cube()
    scene.add(cube_mesh)
    
    # Add lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ]))
    
    # Add point light for better illumination
    point_light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=10.0)
    scene.add(point_light, pose=np.array([
        [1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ]))
    
    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=640.0/480.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    scene.add(camera, pose=camera_pose)
    
    # Initialize renderer (will attempt hardware acceleration by default)
    renderer = pyrender.OffscreenRenderer(640, 480)
    
    try:
        # Render the scene
        color, depth = renderer.render(scene)
        
        # Save rendered image
        output_path = os.path.join(output_dir, 'rendered_scene.png')
        img = Image.fromarray(color)
        img.save(output_path)
        
        print(f"Successfully rendered scene to {output_path}")
        
    finally:
        # Clean up renderer
        renderer.delete()

if __name__ == '__main__':
    main()