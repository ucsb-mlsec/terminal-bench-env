#!/usr/bin/env python3

import numpy as np


class VolumeRenderer:
    """
    A volume renderer that uses ray marching to visualize 3D scalar datasets.
    """
    
    def __init__(self):
        """
        Initialize the renderer with empty attributes.
        """
        self.volume_data = None
        self.width = None
        self.height = None
        self.depth = None
    
    def load_volume(self, filepath):
        """
        Load a .vol file containing volumetric data.
        Format: first line has three space-separated integers (width, height, depth),
        remaining lines contain float values (one per line) in row-major order.
        
        Args:
            filepath: Path to the .vol file
        """
        with open(filepath, 'r') as f:
            # Read dimensions from first line
            dimensions_line = f.readline().strip()
            self.width, self.height, self.depth = map(int, dimensions_line.split())
            
            # Read scalar values
            values = []
            for line in f:
                line = line.strip()
                if line:
                    values.append(float(line))
            
            # Reshape into 3D array (depth, height, width) for row-major order
            self.volume_data = np.array(values).reshape(self.depth, self.height, self.width)
    
    def get_dimensions(self):
        """
        Return the dimensions of the loaded volume.
        
        Returns:
            Tuple of (width, height, depth)
        """
        if self.width is None or self.height is None or self.depth is None:
            return None
        return (self.width, self.height, self.depth)
    
    def sample_volume(self, x, y, z):
        """
        Sample the scalar value at arbitrary 3D position using trilinear interpolation.
        
        Args:
            x, y, z: 3D coordinates to sample (in volume space)
            
        Returns:
            Interpolated scalar value
        """
        if self.volume_data is None:
            return 0.0
        
        # Clamp coordinates to volume bounds
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)
        z = np.clip(z, 0, self.depth - 1)
        
        # Get integer and fractional parts
        x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
        x1, y1, z1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1), min(z0 + 1, self.depth - 1)
        
        xd = x - x0
        yd = y - y0
        zd = z - z0
        
        # Trilinear interpolation
        c000 = self.volume_data[z0, y0, x0]
        c001 = self.volume_data[z0, y0, x1]
        c010 = self.volume_data[z0, y1, x0]
        c011 = self.volume_data[z0, y1, x1]
        c100 = self.volume_data[z1, y0, x0]
        c101 = self.volume_data[z1, y0, x1]
        c110 = self.volume_data[z1, y1, x0]
        c111 = self.volume_data[z1, y1, x1]
        
        c00 = c000 * (1 - xd) + c001 * xd
        c01 = c010 * (1 - xd) + c011 * xd
        c10 = c100 * (1 - xd) + c101 * xd
        c11 = c110 * (1 - xd) + c111 * xd
        
        c0 = c00 * (1 - yd) + c01 * yd
        c1 = c10 * (1 - yd) + c11 * yd
        
        return c0 * (1 - zd) + c1 * zd
    
    def ray_box_intersection(self, ray_origin, ray_direction):
        """
        Compute intersection of a ray with the volume bounding box [0, width] x [0, height] x [0, depth].
        
        Args:
            ray_origin: 3D point where ray starts
            ray_direction: 3D normalized direction vector
            
        Returns:
            (t_near, t_far) or None if no intersection
        """
        if self.volume_data is None:
            return None
        
        ray_origin = np.array(ray_origin, dtype=float)
        ray_direction = np.array(ray_direction, dtype=float)
        
        # Avoid division by zero
        inv_dir = np.where(np.abs(ray_direction) < 1e-10, 1e10, 1.0 / ray_direction)
        
        # Box bounds
        box_min = np.array([0.0, 0.0, 0.0])
        box_max = np.array([self.width, self.height, self.depth])
        
        t1 = (box_min - ray_origin) * inv_dir
        t2 = (box_max - ray_origin) * inv_dir
        
        t_min = np.minimum(t1, t2)
        t_max = np.maximum(t1, t2)
        
        t_near = np.max(t_min)
        t_far = np.min(t_max)
        
        if t_near > t_far or t_far < 0:
            return None
        
        return (max(t_near, 0.0), t_far)
    
    def ray_march(self, ray_origin, ray_direction, num_samples):
        """
        Perform ray marching through the volume, accumulating opacity and color.
        
        Args:
            ray_origin: 3D starting point of the ray
            ray_direction: 3D normalized direction vector
            num_samples: Number of samples to take along the ray
            
        Returns:
            Final accumulated pixel value
        """
        intersection = self.ray_box_intersection(ray_origin, ray_direction)
        if intersection is None:
            return 0.0
        
        t_near, t_far = intersection
        
        if t_near >= t_far:
            return 0.0
        
        step_size = (t_far - t_near) / num_samples
        accumulated_value = 0.0
        accumulated_opacity = 0.0
        
        for i in range(num_samples):
            t = t_near + (i + 0.5) * step_size
            sample_pos = ray_origin + t * ray_direction
            
            scalar_value = self.sample_volume(sample_pos[0], sample_pos[1], sample_pos[2])
            
            # Simple opacity based on scalar value
            opacity = scalar_value * step_size
            opacity = np.clip(opacity, 0.0, 1.0)
            
            # Accumulate using front-to-back compositing
            accumulated_value += (1.0 - accumulated_opacity) * scalar_value * opacity
            accumulated_opacity += (1.0 - accumulated_opacity) * opacity
            
            if accumulated_opacity >= 0.99:
                break
        
        return accumulated_value
    
    def render_image(self, width, height, camera_position, look_at, up_vector):
        """
        Generate a 2D image by casting rays through each pixel.
        
        Args:
            width, height: Image dimensions
            camera_position: 3D position of the camera
            look_at: 3D point the camera is looking at
            up_vector: 3D up direction vector
            
        Returns:
            2D numpy array of pixel values
        """
        camera_position = np.array(camera_position, dtype=float)
        look_at = np.array(look_at, dtype=float)
        up_vector = np.array(up_vector, dtype=float)
        
        # Construct camera basis
        forward = look_at - camera_position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, up_vector)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Create image
        image = np.zeros((height, width))
        
        aspect_ratio = width / height
        fov = 45.0 * np.pi / 180.0
        
        for j in range(height):
            for i in range(width):
                # Normalized device coordinates
                ndc_x = (2.0 * (i + 0.5) / width - 1.0) * aspect_ratio * np.tan(fov / 2)
                ndc_y = (1.0 - 2.0 * (j + 0.5) / height) * np.tan(fov / 2)
                
                ray_direction = forward + ndc_x * right + ndc_y * up
                ray_direction = ray_direction / np.linalg.norm(ray_direction)
                
                pixel_value = self.ray_march(camera_position, ray_direction, num_samples=100)
                image[j, i] = pixel_value
        
        return image