#!/usr/bin/env python3

import numpy as np
from scipy import ndimage

def predict(image_array):
    """
    Rule-based classifier for geometric shapes (circle, square, triangle).
    
    Args:
        image_array: numpy array of shape (28, 28) with values in [0.0, 1.0]
    
    Returns:
        str: one of 'circle', 'square', or 'triangle'
    """
    # Threshold the image to get binary mask
    threshold = 0.5
    binary = (image_array > threshold).astype(float)
    
    # Calculate various geometric features
    corner_score = detect_corners(image_array)
    circularity = calculate_circularity(binary)
    symmetry_score = calculate_symmetry(binary)
    edge_variance = calculate_edge_variance(image_array)
    center_mass_distance = calculate_center_mass_distance(binary)
    
    # Rule-based classification using thresholds
    # These thresholds are intentionally tight to be vulnerable to perturbations
    
    # Circle detection: high circularity, low corner score
    if circularity > 0.75 and corner_score < 0.35:
        return 'circle'
    
    # Square detection: high corner score, high symmetry, low edge variance
    if corner_score > 0.45 and symmetry_score > 0.80 and edge_variance < 0.25:
        return 'square'
    
    # Triangle detection: moderate corner score, asymmetric
    if corner_score > 0.30 and corner_score < 0.60 and center_mass_distance > 0.15:
        return 'triangle'
    
    # Additional rules for edge cases
    if edge_variance > 0.30 and corner_score > 0.35:
        return 'triangle'
    
    if circularity > 0.65:
        return 'circle'
    
    if symmetry_score > 0.85:
        return 'square'
    
    # Default fallback
    return 'triangle'


def detect_corners(image_array):
    """Detect corners using gradient-based approach."""
    # Sobel filters for edges
    sobel_x = ndimage.sobel(image_array, axis=1)
    sobel_y = ndimage.sobel(image_array, axis=0)
    
    # Gradient magnitude
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Harris corner detection approximation
    Ixx = sobel_x * sobel_x
    Iyy = sobel_y * sobel_y
    Ixy = sobel_x * sobel_y
    
    # Sum over neighborhood
    Sxx = ndimage.gaussian_filter(Ixx, sigma=1)
    Syy = ndimage.gaussian_filter(Iyy, sigma=1)
    Sxy = ndimage.gaussian_filter(Ixy, sigma=1)
    
    # Corner response
    det = Sxx * Syy - Sxy**2
    trace = Sxx + Syy
    
    k = 0.04
    corner_response = det - k * trace**2
    
    # Normalize and threshold
    corner_score = np.sum(corner_response > 0.01 * corner_response.max()) / (28 * 28)
    
    return corner_score


def calculate_circularity(binary):
    """Calculate circularity metric (1.0 = perfect circle)."""
    # Find perimeter and area
    area = np.sum(binary)
    
    if area == 0:
        return 0.0
    
    # Perimeter using edge detection
    edges = binary - ndimage.binary_erosion(binary)
    perimeter = np.sum(edges)
    
    if perimeter == 0:
        return 0.0
    
    # Circularity: 4π * area / perimeter^2
    circularity = (4 * np.pi * area) / (perimeter**2)
    
    return min(circularity, 1.0)


def calculate_symmetry(binary):
    """Calculate symmetry score (horizontal and vertical)."""
    h, w = binary.shape
    
    # Vertical symmetry
    left = binary[:, :w//2]
    right = np.fliplr(binary[:, w//2:])
    min_width = min(left.shape[1], right.shape[1])
    v_symmetry = np.mean(left[:, :min_width] == right[:, :min_width])
    
    # Horizontal symmetry
    top = binary[:h//2, :]
    bottom = np.flipud(binary[h//2:, :])
    min_height = min(top.shape[0], bottom.shape[0])
    h_symmetry = np.mean(top[:min_height, :] == bottom[:min_height, :])
    
    return (v_symmetry + h_symmetry) / 2


def calculate_edge_variance(image_array):
    """Calculate variance in edge distances from center."""
    h, w = image_array.shape
    center_y, center_x = h // 2, w // 2
    
    # Find edge pixels
    threshold = 0.5
    binary = image_array > threshold
    edges = binary - ndimage.binary_erosion(binary)
    
    edge_coords = np.argwhere(edges)
    
    if len(edge_coords) == 0:
        return 0.0
    
    # Calculate distances from center
    distances = np.sqrt((edge_coords[:, 0] - center_y)**2 + 
                       (edge_coords[:, 1] - center_x)**2)
    
    # Normalize variance
    variance = np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0.0
    
    return variance


def calculate_center_mass_distance(binary):
    """Calculate distance between geometric center and center of mass."""
    h, w = binary.shape
    geometric_center = np.array([h / 2, w / 2])
    
    # Calculate center of mass
    coords = np.argwhere(binary)
    
    if len(coords) == 0:
        return 0.0
    
    center_of_mass = np.mean(coords, axis=0)
    
    # Normalized distance
    distance = np.linalg.norm(center_of_mass - geometric_center)
    normalized_distance = distance / (h / 2)
    
    return normalized_distance