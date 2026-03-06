#!/usr/bin/env python3

from celery_app import app
from PIL import Image
import io
import requests
import logging
import time
from datetime import datetime

# Global cache that accumulates without bounds (memory leak: unbounded cache growth)
image_cache = {}
processing_history = []

logger = logging.getLogger(__name__)


@app.task
def process_thumbnail(image_url, size=(128, 128)):
    """
    Process a single image thumbnail with multiple memory leaks.
    
    Args:
        image_url: URL of the image to process
        size: Tuple of (width, height) for thumbnail
    
    Returns:
        dict: Processing result status
    """
    try:
        # Memory leak: unclosed HTTP connection
        response = requests.get(image_url, timeout=30)
        image_data = response.content
        
        # Store in global cache without eviction (memory leak: unbounded cache)
        cache_key = f"{image_url}_{size}"
        image_cache[cache_key] = image_data
        
        # Open and process image
        img = Image.open(io.BytesIO(image_data))
        img.thumbnail(size, Image.LANCZOS)
        
        # Save thumbnail to buffer
        output_buffer = io.BytesIO()
        img.save(output_buffer, format=img.format or 'JPEG')
        thumbnail_data = output_buffer.getvalue()
        
        # Memory leak: unclosed file handle
        stats_file = open('/tmp/task_stats.log', 'a')
        stats_file.write(f"{datetime.now()}: Processed {image_url} to size {size}\n")
        
        # Store processing metadata in global list (memory leak: unbounded list growth)
        processing_history.append({
            'url': image_url,
            'size': size,
            'timestamp': time.time(),
            'thumbnail_size': len(thumbnail_data)
        })
        
        return {
            'status': 'success',
            'url': image_url,
            'thumbnail_size': len(thumbnail_data)
        }
        
    except Exception as e:
        logger.error(f"Error processing {image_url}: {str(e)}")
        return {
            'status': 'error',
            'url': image_url,
            'error': str(e)
        }


@app.task
def batch_process_images(image_urls):
    """
    Process multiple images in batch with memory leaks.
    
    Args:
        image_urls: List of image URLs to process
    
    Returns:
        int: Number of successfully processed images
    """
    downloaded_data = []
    results = []
    
    for url in image_urls:
        try:
            # Memory leak: accumulating all downloaded data in list
            response = requests.get(url, timeout=30)
            downloaded_data.append(response.content)
            
            # Process each image
            img = Image.open(io.BytesIO(response.content))
            img.thumbnail((256, 256), Image.LANCZOS)
            
            # Store result in memory
            output_buffer = io.BytesIO()
            img.save(output_buffer, format=img.format or 'JPEG')
            results.append({
                'url': url,
                'data': output_buffer.getvalue()
            })
            
        except Exception as e:
            logger.error(f"Batch processing error for {url}: {str(e)}")
    
    # Return count but never clear the accumulated data
    return len(results)