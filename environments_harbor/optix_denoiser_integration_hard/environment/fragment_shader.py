#!/usr/bin/env python3

import numpy as np

def process_fragments(triangles, material, lighting, width, height):
    """
    Process fragments (pixels) for all triangles and apply lighting.
    
    Args:
        triangles: List of triangle data (vertices with positions and normals)
        material: Material properties dict with 'ambient', 'diffuse', 'specular', 'shininess', 'color'
        lighting: Lighting configuration with 'ambient' and 'lights' list
        width: Output image width
        height: Output image height
    
    Returns:
        numpy array of shape (height, width, 3) with RGB values [0-255]
    """
    # Initialize framebuffer - BUG: initializing with zeros causes black output
    framebuffer = np.zeros((height, width, 3), dtype=np.float32)
    depth_buffer = np.full((height, width), float('inf'), dtype=np.float32)
    
    for triangle in triangles:
        v0 = triangle['vertices'][0]
        v1 = triangle['vertices'][1]
        v2 = triangle['vertices'][2]
        
        # Rasterize triangle to get pixel coordinates
        pixels = rasterize_triangle(v0, v1, v2, width, height)
        
        # Compute triangle normal
        normal = compute_normal(v0, v1, v2)
        
        # Get triangle center for lighting calculation
        center = {
            'x': (v0['x'] + v1['x'] + v2['x']) / 3.0,
            'y': (v0['y'] + v1['y'] + v2['y']) / 3.0,
            'z': (v0['z'] + v1['z'] + v2['z']) / 3.0
        }
        
        # Calculate lighting for this triangle
        lit_color = calculate_lighting(center, normal, material, lighting)
        
        # Fill pixels
        for px, py, depth in pixels:
            if 0 <= px < width and 0 <= py < height:
                if depth < depth_buffer[py, px]:
                    depth_buffer[py, px] = depth
                    framebuffer[py, px] = lit_color
    
    # Convert to 8-bit RGB
    framebuffer = np.clip(framebuffer * 255.0, 0, 255).astype(np.uint8)
    
    return framebuffer


def rasterize_triangle(v0, v1, v2, width, height):
    """
    Rasterize a triangle and return list of pixel coordinates inside it.
    Uses barycentric coordinates for triangle rasterization.
    
    Args:
        v0, v1, v2: Triangle vertices with x, y, z coordinates
        width: Image width
        height: Image height
    
    Returns:
        List of (x, y, depth) tuples for pixels inside triangle
    """
    pixels = []
    
    # Convert to screen space coordinates
    x0 = int((v0['x'] + 1.0) * 0.5 * width)
    y0 = int((1.0 - v0['y']) * 0.5 * height)
    z0 = v0['z']
    
    x1 = int((v1['x'] + 1.0) * 0.5 * width)
    y1 = int((1.0 - v1['y']) * 0.5 * height)
    z1 = v1['z']
    
    x2 = int((v2['x'] + 1.0) * 0.5 * width)
    y2 = int((1.0 - v2['y']) * 0.5 * height)
    z2 = v2['z']
    
    # Compute bounding box
    min_x = max(0, min(x0, x1, x2))
    max_x = min(width - 1, max(x0, x1, x2))
    min_y = max(0, min(y0, y1, y2))
    max_y = min(height - 1, max(y0, y1, y2))
    
    # Scan all pixels in bounding box
    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            # Compute barycentric coordinates
            w0, w1, w2 = barycentric_coordinates(x, y, x0, y0, x1, y1, x2, y2)
            
            # Check if point is inside triangle
            if w0 >= 0 and w1 >= 0 and w2 >= 0:
                # Interpolate depth
                depth = w0 * z0 + w1 * z1 + w2 * z2
                pixels.append((x, y, depth))
    
    return pixels


def barycentric_coordinates(px, py, x0, y0, x1, y1, x2, y2):
    """
    Calculate barycentric coordinates for a point relative to a triangle.
    
    Args:
        px, py: Point coordinates
        x0, y0, x1, y1, x2, y2: Triangle vertex coordinates
    
    Returns:
        Tuple of (w0, w1, w2) barycentric weights
    """
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    
    if abs(denom) < 1e-6:
        return -1, -1, -1
    
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    
    return w0, w1, w2


def calculate_lighting(position, normal, material, lighting):
    """
    Calculate final lighting color using Phong illumination model.
    
    Args:
        position: Fragment position dict with x, y, z
        normal: Surface normal as numpy array [x, y, z]
        material: Material properties
        lighting: Lighting configuration
    
    Returns:
        RGB color as numpy array [r, g, b] in range [0, 1]
    """
    # Start with ambient lighting
    base_color = np.array(material.get('color', [1.0, 1.0, 1.0]), dtype=np.float32)
    ambient_coeff = material.get('ambient', 0.1)
    ambient_light = np.array(lighting.get('ambient', [1.0, 1.0, 1.0]), dtype=np.float32)
    
    color = apply_ambient(base_color, ambient_coeff, ambient_light)
    
    # Add contribution from each light
    lights = lighting.get('lights', [])
    for light in lights:
        light_pos = np.array([light['x'], light['y'], light['z']], dtype=np.float32)
        light_color = np.array(light.get('color', [1.0, 1.0, 1.0]), dtype=np.float32)
        light_intensity = light.get('intensity', 1.0)
        
        # Calculate light direction
        frag_pos = np.array([position['x'], position['y'], position['z']], dtype=np.float32)
        light_dir = light_pos - frag_pos
        distance = np.linalg.norm(light_dir)
        
        if distance > 0:
            light_dir = light_dir / distance
        
        # Apply diffuse lighting
        diffuse_coeff = material.get('diffuse', 0.7)
        color = apply_diffuse(color, normal, light_dir, light_color, light_intensity * diffuse_coeff)
        
        # Apply specular lighting
        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        specular_coeff = material.get('specular', 0.3)
        shininess = material.get('shininess', 32.0)
        color = apply_specular(color, normal, light_dir, view_dir, light_color, 
                              specular_coeff, shininess, light_intensity)
    
    return np.clip(color, 0.0, 1.0)


def compute_normal(v0, v1, v2):
    """
    Compute triangle normal using cross product.
    
    Args:
        v0, v1, v2: Triangle vertices with x, y, z coordinates
    
    Returns:
        Normalized normal vector as numpy array [x, y, z]
    """
    p0 = np.array([v0['x'], v0['y'], v0['z']], dtype=np.float32)
    p1 = np.array([v1['x'], v1['y'], v1['z']], dtype=np.float32)
    p2 = np.array([v2['x'], v2['y'], v2['z']], dtype=np.float32)
    
    # Compute edge vectors
    edge1 = p1 - p0
    edge2 = p2 - p0
    
    # Cross product
    normal = np.cross(edge1, edge2)
    
    # Normalize
    length = np.linalg.norm(normal)
    if length > 1e-6:
        normal = normal / length
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    return normal


def apply_ambient(color, ambient_coeff, ambient_light):
    """
    Apply ambient lighting to base color.
    
    Args:
        color: Base material color as numpy array [r, g, b]
        ambient_coeff: Ambient reflection coefficient
        ambient_light: Ambient light color as numpy array [r, g, b]
    
    Returns:
        Color with ambient lighting applied
    """
    return color * ambient_coeff * ambient_light


def apply_diffuse(color, normal, light_dir, light_color, intensity):
    """
    Apply diffuse (Lambertian) lighting.
    
    Args:
        color: Current color as numpy array [r, g, b]
        normal: Surface normal as numpy array [x, y, z]
        light_dir: Direction to light as numpy array [x, y, z]
        light_color: Light color as numpy array [r, g, b]
        intensity: Light intensity multiplier
    
    Returns:
        Color with diffuse lighting added
    """
    # Calculate diffuse factor using Lambert's cosine law
    diffuse_factor = max(0.0, np.dot(normal, light_dir))
    
    # Add diffuse contribution
    diffuse_contribution = light_color * intensity * diffuse_factor
    
    return color + diffuse_contribution


def apply_specular(color, normal, light_dir, view_dir, light_color, 
                   specular_coeff, shininess, intensity):
    """
    Apply specular (Blinn-Phong) lighting.
    
    Args:
        color: Current color as numpy array [r, g, b]
        normal: Surface normal as numpy array [x, y, z]
        light_dir: Direction to light as numpy array [x, y, z]
        view_dir: Direction to viewer as numpy array [x, y, z]
        light_color: Light color as numpy array [r, g, b]
        specular_coeff: Specular reflection coefficient
        shininess: Specular shininess exponent
        intensity: Light intensity multiplier
    
    Returns:
        Color with specular lighting added
    """
    # Compute half vector for Blinn-Phong
    half_vector = light_dir + view_dir
    half_length = np.linalg.norm(half_vector)
    
    if half_length > 1e-6:
        half_vector = half_vector / half_length
    
    # Calculate specular factor
    spec_angle = max(0.0, np.dot(normal, half_vector))
    specular_factor = pow(spec_angle, shininess)
    
    # Add specular contribution
    specular_contribution = light_color * specular_coeff * specular_factor * intensity
    
    return color + specular_contribution