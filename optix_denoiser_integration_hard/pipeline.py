#!/usr/bin/env python3

import sys
import json
import numpy as np
from PIL import Image
import os

# Local module imports (simulated since we're creating the pipeline)
# In a real scenario, these would be separate files

class SceneLoader:
    """Handles loading and parsing of scene JSON files"""
    
    @staticmethod
    def load_scene(scene_path):
        """Load scene description from JSON file"""
        print(f"Loading scene from: {scene_path}")
        
        with open(scene_path, 'r') as f:
            scene_data = json.load(f)
        
        # Parse vertices
        vertices = np.array(scene_data.get('vertices', []), dtype=np.float32)
        
        # Parse faces/indices
        indices = np.array(scene_data.get('indices', []), dtype=np.int32)
        
        # Parse materials
        material = scene_data.get('material', {
            'ambient': [0.2, 0.2, 0.2],
            'diffuse': [0.8, 0.8, 0.8],
            'specular': [1.0, 1.0, 1.0],
            'shininess': 32.0
        })
        
        # Parse lighting
        lighting = scene_data.get('lighting', {
            'position': [5.0, 5.0, 5.0],
            'color': [1.0, 1.0, 1.0],
            'intensity': 1.0
        })
        
        # Parse camera
        camera = scene_data.get('camera', {
            'position': [0.0, 0.0, 5.0],
            'target': [0.0, 0.0, 0.0],
            'up': [0.0, 1.0, 0.0],
            'fov': 45.0
        })
        
        # Output resolution
        resolution = scene_data.get('resolution', [800, 600])
        
        print(f"  Loaded {len(vertices)} vertices")
        print(f"  Loaded {len(indices)} indices")
        
        return {
            'vertices': vertices,
            'indices': indices,
            'material': material,
            'lighting': lighting,
            'camera': camera,
            'resolution': resolution
        }


class VertexShader:
    """Processes vertex transformations"""
    
    @staticmethod
    def process_vertices(vertices, camera, resolution):
        """Transform vertices from world space to screen space"""
        print("Processing vertices through vertex shader...")
        
        # Create view matrix
        cam_pos = np.array(camera['position'], dtype=np.float32)
        cam_target = np.array(camera['target'], dtype=np.float32)
        cam_up = np.array(camera['up'], dtype=np.float32)
        
        # View transformation
        forward = cam_target - cam_pos
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, cam_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Create projection matrix (simple perspective)
        fov = np.radians(camera['fov'])
        aspect = resolution[0] / resolution[1]
        near = 0.1
        far = 100.0
        
        f = 1.0 / np.tan(fov / 2.0)
        
        # Transform vertices
        transformed_vertices = []
        
        for vertex in vertices:
            # World to view space
            v = vertex - cam_pos
            view_pos = np.array([
                np.dot(v, right),
                np.dot(v, up),
                np.dot(v, forward)
            ])
            
            # Perspective projection
            if view_pos[2] > 0:  # In front of camera
                proj_x = (f / aspect) * view_pos[0] / view_pos[2]
                proj_y = f * view_pos[1] / view_pos[2]
                
                # To screen space
                screen_x = (proj_x + 1.0) * 0.5 * resolution[0]
                screen_y = (1.0 - proj_y) * 0.5 * resolution[1]
                
                transformed_vertices.append([screen_x, screen_y, view_pos[2]])
            else:
                transformed_vertices.append([0, 0, -1])  # Behind camera
        
        print(f"  Transformed {len(transformed_vertices)} vertices")
        
        return np.array(transformed_vertices, dtype=np.float32)


class FragmentShader:
    """Processes fragments/pixels with lighting calculations"""
    
    @staticmethod
    def process_fragments(screen_vertices, indices, material, lighting, resolution):
        """Rasterize triangles and apply lighting"""
        print("Processing fragments through fragment shader...")
        
        width, height = resolution
        
        # Initialize buffers
        color_buffer = np.zeros((height, width, 3), dtype=np.float32)
        depth_buffer = np.full((height, width), float('inf'), dtype=np.float32)
        
        # Light properties
        light_pos = np.array(lighting['position'], dtype=np.float32)
        light_color = np.array(lighting['color'], dtype=np.float32)
        light_intensity = lighting['intensity']
        
        # Material properties
        ambient = np.array(material['ambient'], dtype=np.float32)
        diffuse = np.array(material['diffuse'], dtype=np.float32)
        specular = np.array(material['specular'], dtype=np.float32)
        
        # Process each triangle
        num_triangles = len(indices) // 3
        
        for i in range(num_triangles):
            idx0 = indices[i * 3]
            idx1 = indices[i * 3 + 1]
            idx2 = indices[i * 3 + 2]
            
            v0 = screen_vertices[idx0]
            v1 = screen_vertices[idx1]
            v2 = screen_vertices[idx2]
            
            # Skip if behind camera
            if v0[2] <= 0 or v1[2] <= 0 or v2[2] <= 0:
                continue
            
            # Compute bounding box
            min_x = max(0, int(min(v0[0], v1[0], v2[0])))
            max_x = min(width - 1, int(max(v0[0], v1[0], v2[0])))
            min_y = max(0, int(min(v0[1], v1[1], v2[1])))
            max_y = min(height - 1, int(max(v0[1], v1[1], v2[1])))
            
            # Rasterize triangle
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    # Barycentric coordinates
                    p = np.array([x + 0.5, y + 0.5])
                    
                    # BUG: This division creates zero weights
                    # The denominator calculation is incorrect - should use cross product area
                    denom = 1.0  # INCORRECT: Should be twice the triangle area
                    
                    # When denom is 1.0 instead of actual area, barycentric coords are wrong
                    w0 = ((v1[0] - v2[0]) * (p[1] - v2[1]) - (v1[1] - v2[1]) * (p[0] - v2[0])) / denom
                    w1 = ((v2[0] - v0[0]) * (p[1] - v0[1]) - (v2[1] - v0[1]) * (p[0] - v0[0])) / denom
                    w2 = ((v0[0] - v1[0]) * (p[1] - v1[1]) - (v0[1] - v1[1]) * (p[0] - v1[0])) / denom
                    
                    # This check will almost always fail with wrong denom
                    if w0 >= 0 and w1 >= 0 and w2 >= 0:
                        # Interpolate depth
                        depth = w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
                        
                        if depth < depth_buffer[y, x]:
                            depth_buffer[y, x] = depth
                            
                            # Simple lighting calculation
                            # Ambient
                            color = ambient * light_intensity
                            
                            # Diffuse (simplified - using average normal)
                            normal = np.array([0, 0, 1])  # Simplified
                            light_dir = light_pos / np.linalg.norm(light_pos)
                            diff = max(0.0, np.dot(normal, light_dir))
                            color += diffuse * diff * light_color * light_intensity
                            
                            # Clamp color
                            color = np.clip(color, 0.0, 1.0)
                            
                            color_buffer[y, x] = color
        
        print(f"  Processed fragments for {num_triangles} triangles")
        
        return color_buffer


class OutputGenerator:
    """Handles output image generation"""
    
    @staticmethod
    def save_image(color_buffer, output_path):
        """Save color buffer as PNG image"""
        print(f"Generating output image: {output_path}")
        
        # Convert to 8-bit RGB
        image_data = (color_buffer * 255).astype(np.uint8)
        
        # Create PIL image
        image = Image.fromarray(image_data, mode='RGB')
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        
        print(f"  Image saved successfully")
        
        return output_path


def main():
    """Main rendering pipeline orchestration"""
    print("=" * 60)
    print("Rendering Pipeline Started")
    print("=" * 60)
    
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("ERROR: No scene file provided")
        print("Usage: python pipeline.py <scene_file.json>")
        sys.exit(1)
    
    scene_file = sys.argv[1]
    
    # Validate scene file exists
    if not os.path.exists(scene_file):
        print(f"ERROR: Scene file not found: {scene_file}")
        sys.exit(1)
    
    try:
        # Stage 1: Load scene
        print("\n[Stage 1] Loading Scene")
        print("-" * 60)
        scene_loader = SceneLoader()
        scene_data = scene_loader.load_scene(scene_file)
        
        # Stage 2: Vertex shader processing
        print("\n[Stage 2] Vertex Shader")
        print("-" * 60)
        vertex_shader = VertexShader()
        transformed_vertices = vertex_shader.process_vertices(
            scene_data['vertices'],
            scene_data['camera'],
            scene_data['resolution']
        )
        
        # Stage 3: Fragment shader processing
        print("\n[Stage 3] Fragment Shader")
        print("-" * 60)
        fragment_shader = FragmentShader()
        color_buffer = fragment_shader.process_fragments(
            transformed_vertices,
            scene_data['indices'],
            scene_data['material'],
            scene_data['lighting'],
            scene_data['resolution']
        )
        
        # Stage 4: Output generation
        print("\n[Stage 4] Output Generation")
        print("-" * 60)
        output_generator = OutputGenerator()
        
        # Determine output path
        scene_dir = os.path.dirname(os.path.abspath(scene_file))
        scene_name = os.path.splitext(os.path.basename(scene_file))[0]
        output_path = os.path.join(scene_dir, f"{scene_name}_output.png")
        
        output_generator.save_image(color_buffer, output_path)
        
        print("\n" + "=" * 60)
        print("Rendering Pipeline Completed Successfully")
        print("=" * 60)
        print(f"Output: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())