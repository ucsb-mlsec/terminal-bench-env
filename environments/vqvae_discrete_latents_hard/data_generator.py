#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_circle(center_x, center_y, radius, image_size=28):
    """Generate a circle image."""
    img = np.zeros((image_size, image_size), dtype=np.float32)
    y, x = np.ogrid[:image_size, :image_size]
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    img[dist_from_center <= radius] = 1.0
    return img


def generate_square(center_x, center_y, side_length, image_size=28):
    """Generate a square image."""
    img = np.zeros((image_size, image_size), dtype=np.float32)
    half_side = side_length // 2
    x_min = max(0, int(center_x - half_side))
    x_max = min(image_size, int(center_x + half_side))
    y_min = max(0, int(center_y - half_side))
    y_max = min(image_size, int(center_y + half_side))
    img[y_min:y_max, x_min:x_max] = 1.0
    return img


def generate_triangle(center_x, center_y, size, image_size=28):
    """Generate a triangle image."""
    img = np.zeros((image_size, image_size), dtype=np.float32)
    
    # Define vertices of an equilateral triangle centered at (center_x, center_y)
    height = size * np.sqrt(3) / 2
    v1 = (center_x, center_y - 2*height/3)  # Top vertex
    v2 = (center_x - size/2, center_y + height/3)  # Bottom left
    v3 = (center_x + size/2, center_y + height/3)  # Bottom right
    
    # Fill triangle using barycentric coordinates
    for i in range(image_size):
        for j in range(image_size):
            # Check if point (j, i) is inside triangle
            def sign(p1, p2, p3):
                return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
            
            d1 = sign((j, i), v1, v2)
            d2 = sign((j, i), v2, v3)
            d3 = sign((j, i), v3, v1)
            
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            
            if not (has_neg and has_pos):
                img[i, j] = 1.0
    
    return img


def generate_rectangle(center_x, center_y, width, height, image_size=28):
    """Generate a rectangle image."""
    img = np.zeros((image_size, image_size), dtype=np.float32)
    half_width = width // 2
    half_height = height // 2
    x_min = max(0, int(center_x - half_width))
    x_max = min(image_size, int(center_x + half_width))
    y_min = max(0, int(center_y - half_height))
    y_max = min(image_size, int(center_y + half_height))
    img[y_min:y_max, x_min:x_max] = 1.0
    return img


class ShapeDataset(Dataset):
    """Dataset that generates random geometric shapes."""
    
    def __init__(self, num_samples=1000, image_size=28):
        self.num_samples = num_samples
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Generate a random shape image on-the-fly."""
        # Use idx as seed for reproducibility if needed
        np.random.seed(idx)
        
        # Randomly select shape type
        shape_type = np.random.choice(['circle', 'square', 'triangle', 'rectangle'])
        
        # Random position (keep away from edges)
        center_x = np.random.randint(7, self.image_size - 7)
        center_y = np.random.randint(7, self.image_size - 7)
        
        if shape_type == 'circle':
            radius = np.random.randint(3, 8)
            img = generate_circle(center_x, center_y, radius, self.image_size)
        elif shape_type == 'square':
            side_length = np.random.randint(6, 14)
            img = generate_square(center_x, center_y, side_length, self.image_size)
        elif shape_type == 'triangle':
            size = np.random.randint(8, 16)
            img = generate_triangle(center_x, center_y, size, self.image_size)
        else:  # rectangle
            width = np.random.randint(6, 14)
            height = np.random.randint(6, 14)
            img = generate_rectangle(center_x, center_y, width, height, self.image_size)
        
        # Convert to tensor with channel dimension
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Shape: (1, 28, 28)
        
        return img_tensor