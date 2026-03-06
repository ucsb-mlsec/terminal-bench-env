#!/usr/bin/env python3
"""
output_generator.py - Handles image output for the rendering pipeline
"""

import numpy as np
from PIL import Image
import os


def normalize_pixels(pixel_data):
    """
    Normalize pixel values to 0-255 range by clamping.
    
    Args:
        pixel_data: numpy array of pixel values (may be any range)
        
    Returns:
        numpy array with values clamped to 0-255 range
    """
    # Clamp values to 0-255 range
    normalized = np.clip(pixel_data, 0, 255)
    return normalized


def convert_to_uint8(pixel_data):
    """
    Convert float pixel data to uint8 format.
    
    Args:
        pixel_data: numpy array of pixel values (float)
        
    Returns:
        numpy array converted to uint8 type
    """
    # Convert to uint8 type
    uint8_data = pixel_data.astype(np.uint8)
    return uint8_data


def save_image(pixel_data, output_path):
    """
    Save pixel data as a PNG image file.
    
    Args:
        pixel_data: numpy array of shape (height, width, 3) containing RGB values
        output_path: file path where the image should be saved
        
    Raises:
        IOError: if there's an issue writing the file
        ValueError: if pixel_data has invalid shape
    """
    try:
        # Validate input shape
        if len(pixel_data.shape) != 3 or pixel_data.shape[2] != 3:
            raise ValueError(f"Invalid pixel_data shape: {pixel_data.shape}. Expected (height, width, 3)")
        
        # Normalize pixels to 0-255 range
        normalized_data = normalize_pixels(pixel_data)
        
        # Convert to uint8
        uint8_data = convert_to_uint8(normalized_data)
        
        # Create PIL Image from array
        image = Image.fromarray(uint8_data, mode='RGB')
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save the image
        image.save(output_path, 'PNG')
        
    except IOError as e:
        raise IOError(f"Failed to write image to {output_path}: {str(e)}")
    except Exception as e:
        raise Exception(f"Error saving image: {str(e)}")


def validate_output_path(output_path):
    """
    Validate that the output path is writable.
    
    Args:
        output_path: path to validate
        
    Returns:
        True if path is valid and writable
    """
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            return False
        return True
    except Exception:
        return False