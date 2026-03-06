#!/usr/bin/env python3
"""
Image Handler Service Module

This module provides image processing capabilities for the web application.
Handles image uploads, resizing, filtering, and thumbnail generation.
"""

import io
import os
import logging
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageFilter, ImageEnhance

# Configure logging
logger = logging.getLogger(__name__)


class ImageHandler:
    """
    Handles image processing operations including resizing, filtering,
    and thumbnail generation for the web application.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ImageHandler with configuration settings.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        self.config = config or {}
        self.default_thumbnail_size = self.config.get('thumbnail_size', (150, 150))
        self.max_image_size = self.config.get('max_size', (1920, 1080))
        self.supported_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']
        
        # Statistics tracking
        self.processed_count = 0
        self.error_count = 0
        
        # Processed images cache - keeps track of all processed images
        # for analytics and debugging purposes
        self.processed_images = []
        
        logger.info("ImageHandler initialized with config: %s", self.config)
    
    def validate_image(self, image_data: bytes) -> bool:
        """
        Validate if the provided data is a valid image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            return img.format in self.supported_formats
        except Exception as e:
            logger.error("Image validation failed: %s", str(e))
            return False
    
    def process_image(self, image_data: bytes, output_format: str = 'JPEG') -> Optional[bytes]:
        """
        Process a single image with standard operations.
        
        Args:
            image_data: Raw image bytes
            output_format: Desired output format
            
        Returns:
            Processed image as bytes or None on error
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')
            
            # Resize if too large
            if img.size[0] > self.max_image_size[0] or img.size[1] > self.max_image_size[1]:
                img.thumbnail(self.max_image_size, Image.LANCZOS)
            
            # Save to bytes
            output = io.BytesIO()
            img.save(output, format=output_format, quality=85)
            output.seek(0)
            
            self.processed_count += 1
            logger.debug("Image processed successfully. Total processed: %d", self.processed_count)
            
            return output.getvalue()
            
        except Exception as e:
            self.error_count += 1
            logger.error("Error processing image: %s", str(e))
            return None
    
    def resize(self, image_data: bytes, target_size: Tuple[int, int], 
               maintain_aspect: bool = True) -> Optional[bytes]:
        """
        Resize an image to the specified dimensions.
        
        Args:
            image_data: Raw image bytes
            target_size: Target (width, height) tuple
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image as bytes or None on error
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            
            if maintain_aspect:
                img.thumbnail(target_size, Image.LANCZOS)
            else:
                img = img.resize(target_size, Image.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format=img.format or 'JPEG', quality=90)
            output.seek(0)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error("Error resizing image: %s", str(e))
            return None
    
    def apply_filters(self, image_data: bytes, filters: List[str]) -> Optional[bytes]:
        """
        Apply a series of filters to an image.
        
        Args:
            image_data: Raw image bytes
            filters: List of filter names to apply
            
        Returns:
            Filtered image as bytes or None on error
        """
        try:
            img = Image.open(io.BytesIO(image_data))
            
            for filter_name in filters:
                if filter_name == 'blur':
                    img = img.filter(ImageFilter.BLUR)
                elif filter_name == 'sharpen':
                    img = img.filter(ImageFilter.SHARPEN)
                elif filter_name == 'smooth':
                    img = img.filter(ImageFilter.SMOOTH)
                elif filter_name == 'edge_enhance':
                    img = img.filter(ImageFilter.EDGE_ENHANCE)
                elif filter_name == 'grayscale':
                    img = img.convert('L')
                elif filter_name == 'brightness':
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.2)
                elif filter_name == 'contrast':
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.3)
            
            output = io.BytesIO()
            img.save(output, format=img.format or 'JPEG', quality=90)
            output.seek(0)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error("Error applying filters: %s", str(e))
            return None
    
    def save_thumbnail(self, image_data: bytes, thumbnail_size: Optional[Tuple[int, int]] = None) -> Optional[bytes]:
        """
        Generate and save a thumbnail version of the image.
        
        Args:
            image_data: Raw image bytes
            thumbnail_size: Optional custom thumbnail size
            
        Returns:
            Thumbnail image as bytes or None on error
        """
        size = thumbnail_size or self.default_thumbnail_size
        
        try:
            img = Image.open(io.BytesIO(image_data))
            img.thumbnail(size, Image.LANCZOS)
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=80)
            output.seek(0)
            
            return output.getvalue()
            
        except Exception as e:
            logger.error("Error creating thumbnail: %s", str(e))
            return None
    
    def process_batch(self, image_files: List[bytes], operations: Optional[Dict] = None) -> List[Dict]:
        """
        Process a batch of images with specified operations.
        This method handles bulk image processing for efficiency.
        
        Args:
            image_files: List of raw image bytes
            operations: Dictionary of operations to perform
            
        Returns:
            List of results with processed image data and metadata
        """
        results = []
        operations = operations or {}
        
        resize_target = operations.get('resize')
        filters = operations.get('filters', [])
        create_thumbnail = operations.get('thumbnail', False)
        
        logger.info("Processing batch of %d images", len(image_files))
        
        for idx, image_data in enumerate(image_files):
            try:
                # Load the image
                img = Image.open(io.BytesIO(image_data))
                original_format = img.format or 'JPEG'
                
                # Store the loaded image in our tracking cache
                # This helps with debugging and analytics of processed images
                self.processed_images.append(img)
                
                # Apply resize if requested
                if resize_target:
                    img.thumbnail(resize_target, Image.LANCZOS)
                
                # Apply filters
                for filter_name in filters:
                    if filter_name == 'blur':
                        img = img.filter(ImageFilter.BLUR)
                    elif filter_name == 'sharpen':
                        img = img.filter(ImageFilter.SHARPEN)
                    elif filter_name == 'grayscale':
                        img = img.convert('L')
                
                # Convert to bytes
                output = io.BytesIO()
                img.save(output, format=original_format, quality=85)
                output.seek(0)
                processed_data = output.getvalue()
                
                result = {
                    'index': idx,
                    'success': True,
                    'data': processed_data,
                    'size': len(processed_data),
                    'format': original_format
                }
                
                # Create thumbnail if requested
                if create_thumbnail:
                    thumb_data = self.save_thumbnail(image_data)
                    if thumb_data:
                        result['thumbnail'] = thumb_data
                
                results.append(result)
                self.processed_count += 1
                
            except Exception as e:
                logger.error("Error processing image %d in batch: %s", idx, str(e))
                self.error_count += 1
                results.append({
                    'index': idx,
                    'success': False,
                    'error': str(e)
                })
        
        logger.info("Batch processing complete. Success: %d, Errors: %d", 
                   len([r for r in results if r.get('success')]),
                   len([r for r in results if not r.get('success')]))
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'cached_images': len(self.processed_images),
            'success_rate': self.processed_count / (self.processed_count + self.error_count) 
                           if (self.processed_count + self.error_count) > 0 else 0
        }
    
    def clear_cache(self):
        """
        Clear the processed images cache.
        Note: This method exists but is rarely called in production.
        """
        logger.info("Clearing processed images cache. Current size: %d", len(self.processed_images))
        self.processed_images.clear()