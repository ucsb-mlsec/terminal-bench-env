#!/usr/bin/env python3

import os
import yaml
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the sanitizer module (will be the broken one initially, then fixed)
try:
    from sanitizer import sanitize_html
except ImportError:
    logger.warning("Sanitizer module not found, using placeholder")
    def sanitize_html(content):
        return content


def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        return {}


class ContentProcessor:
    """Main content processing service for the CMS"""
    
    def __init__(self, config_path='/opt/cms/cms_config.yaml'):
        """Initialize the content processor with configuration"""
        self.config = load_config(config_path)
        self.max_content_length = self.config.get('max_content_length', 1000000)
        self.allowed_tags = self.config.get('allowed_tags', [])
        logger.info("ContentProcessor initialized")
    
    def validate_content(self, content):
        """Validate content before processing"""
        if not isinstance(content, str):
            raise TypeError("Content must be a string")
        
        if len(content) == 0:
            raise ValueError("Content cannot be empty")
        
        if len(content) > self.max_content_length:
            raise ValueError(f"Content exceeds maximum length of {self.max_content_length}")
        
        logger.info(f"Content validated: {len(content)} characters")
        return True
    
    def process_content(self, content):
        """Process and sanitize content, adding metadata"""
        try:
            self.validate_content(content)
            
            # Sanitize the content
            sanitized = sanitize_html(content)
            
            # Add metadata
            processed = {
                'content': sanitized,
                'processed_at': datetime.now().isoformat(),
                'original_length': len(content),
                'sanitized_length': len(sanitized),
                'sanitized': True
            }
            
            logger.info(f"Content processed successfully: {len(content)} -> {len(sanitized)} chars")
            return processed
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")
            raise
    
    def load_sample_file(self, filepath):
        """Load HTML content from a file"""
        try:
            path = Path(filepath)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {filepath}")
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"Loaded file: {filepath} ({len(content)} bytes)")
            return content
            
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            raise
    
    def store_processed_content(self, content, filename):
        """Store processed content to file (placeholder implementation)"""
        output_dir = Path('/opt/cms/processed')
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / filename
        logger.info(f"Would store processed content to: {output_path}")
        # Actual storage implementation would go here
        return str(output_path)


def batch_process(file_list):
    """Process multiple files in batch"""
    processor = ContentProcessor()
    results = []
    
    logger.info(f"Starting batch process of {len(file_list)} files")
    
    for filepath in file_list:
        try:
            content = processor.load_sample_file(filepath)
            processed = processor.process_content(content)
            results.append({
                'file': filepath,
                'status': 'success',
                'data': processed
            })
        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")
            results.append({
                'file': filepath,
                'status': 'error',
                'error': str(e)
            })
    
    logger.info(f"Batch processing complete: {len(results)} files processed")
    return results


if __name__ == '__main__':
    # Example usage
    logger.info("Starting content processor demo")
    
    processor = ContentProcessor()
    
    # Process sample files
    sample_dir = Path('/opt/cms/samples/legitimate')
    if sample_dir.exists():
        sample_files = list(sample_dir.glob('*.html'))
        logger.info(f"Found {len(sample_files)} sample files")
        
        if sample_files:
            results = batch_process([str(f) for f in sample_files[:3]])
            for result in results:
                print(f"File: {result['file']}, Status: {result['status']}")
    else:
        logger.warning(f"Sample directory not found: {sample_dir}")
        
        # Demo with inline content
        demo_content = "<p>This is <b>bold</b> and <i>italic</i> text with a <a href='http://example.com'>link</a>.</p>"
        processed = processor.process_content(demo_content)
        print(f"Demo processed: {processed['sanitized_length']} chars")