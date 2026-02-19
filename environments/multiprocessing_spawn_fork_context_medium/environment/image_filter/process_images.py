#!/usr/bin/env python3

import multiprocessing as mp
import time
import os
from datetime import datetime

# Configuration class with complex state that affects pickling behavior
class Configuration:
    """
    Image processing configuration with methods and complex state.
    This class holds processing parameters and runtime state.
    """
    def __init__(self, filter_type='blur', intensity=5, output_format='png'):
        self.filter_type = filter_type
        self.intensity = intensity
        self.output_format = output_format
        self.creation_time = datetime.now()
        self.process_id = os.getpid()
        # Store a reference to a method, which complicates serialization
        self._apply_filter = self._get_filter_function()
        
    def _get_filter_function(self):
        """Return the appropriate filter function based on filter type"""
        if self.filter_type == 'blur':
            return self._blur_filter
        elif self.filter_type == 'sharpen':
            return self._sharpen_filter
        else:
            return self._default_filter
    
    def _blur_filter(self, data):
        """Simulate blur filter computation"""
        result = sum(data['pixels']) / len(data['pixels'])
        return result * 0.9
    
    def _sharpen_filter(self, data):
        """Simulate sharpen filter computation"""
        result = sum(data['pixels']) / len(data['pixels'])
        return result * 1.1
    
    def _default_filter(self, data):
        """Default filter"""
        return sum(data['pixels']) / len(data['pixels'])
    
    def process(self, image_data):
        """Process image data using the configured filter"""
        # Apply intensity scaling
        base_result = self._apply_filter(image_data)
        return base_result * (self.intensity / 10.0)

# Global configuration object - workers will need to access this
config = Configuration(filter_type='blur', intensity=8, output_format='png')

def worker_process_image(image_data):
    """
    Worker function that processes a single image.
    This function is called by each worker process in the pool.
    It relies on the global configuration object.
    """
    try:
        # Access the global configuration object
        image_id = image_data['id']
        
        # Simulate some processing time
        time.sleep(0.1)
        
        # Use the configuration to process the image
        processed_value = config.process(image_data)
        
        # Simulate additional processing based on config
        if config.filter_type == 'blur':
            processed_value *= 0.95
        
        return {
            'id': image_id,
            'status': 'success',
            'processed_value': processed_value,
            'filter': config.filter_type
        }
    except Exception as e:
        return {
            'id': image_data.get('id', 'unknown'),
            'status': 'error',
            'error': str(e)
        }

def create_sample_images(count=15):
    """
    Create sample image data for processing.
    Each image is represented as a dictionary with mock pixel data.
    """
    images = []
    for i in range(count):
        images.append({
            'id': f'image_{i:03d}',
            'pixels': [float(x) for x in range(100 + i, 120 + i)],
            'width': 100,
            'height': 100
        })
    return images

def process_image_batch():
    """
    Main processing function that uses multiprocessing to handle images in parallel.
    Creates a pool of worker processes to process images concurrently.
    """
    print("Starting image processing pipeline...")
    print(f"Configuration: filter={config.filter_type}, intensity={config.intensity}")
    
    # Create sample images
    images = create_sample_images(15)
    print(f"Created {len(images)} images to process")
    
    # Process images using multiprocessing pool
    # Note: We're not setting the start method, relying on platform default
    with mp.Pool(processes=4) as pool:
        results = pool.map(worker_process_image, images)
    
    # Check results
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nProcessing complete: {successful} successful, {failed} failed")
    
    return results

if __name__ == '__main__':
    try:
        results = process_image_batch()
        print("\n✓ Pipeline completed successfully!")
        print(f"Processed {len(results)} images")
    except Exception as e:
        print(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()