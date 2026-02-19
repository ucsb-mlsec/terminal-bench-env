#!/usr/bin/env python
"""Image utilities module with various image processing functions."""

import hashlib
import io
from functools import lru_cache
from PIL import Image, ImageFilter, ImageEnhance

# Global cache that stores processed images indefinitely without size limits
image_cache = {}

# Global list that accumulates all resize operations
resize_history = []


def get_cached_image(image_data):
    """
    Retrieve or create a cached image based on image data.
    
    Memory leak: This function stores images in the global image_cache dictionary
    without any eviction policy or size limits. The cache grows indefinitely.
    """
    # Compute MD5 hash as cache key
    hash_key = hashlib.md5(image_data).hexdigest()
    
    # Check if image is already cached
    if hash_key not in image_cache:
        # Create PIL Image object and store in cache
        image_obj = Image.open(io.BytesIO(image_data))
        image_cache[hash_key] = image_obj.copy()
    
    return image_cache[hash_key]


def resize_image(image, width, height):
    """
    Resize an image to specified dimensions.
    
    Memory leak: This function appends every resized image to the resize_history
    list which never gets cleared, causing unbounded memory growth.
    """
    # Create resized image
    resized = image.resize((width, height), Image.LANCZOS)
    
    # Store in history list - never cleared
    resize_history.append(resized)
    
    return resized


def apply_filters(image, filter_name):
    """
    Apply specified filter to an image.
    
    Args:
        image: PIL Image object
        filter_name: Name of filter to apply (blur, sharpen, contour, enhance)
    
    Returns:
        Filtered PIL Image object
    """
    if filter_name == 'blur':
        return image.filter(ImageFilter.BLUR)
    elif filter_name == 'sharpen':
        return image.filter(ImageFilter.SHARPEN)
    elif filter_name == 'contour':
        return image.filter(ImageFilter.CONTOUR)
    elif filter_name == 'enhance':
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)
    else:
        return image