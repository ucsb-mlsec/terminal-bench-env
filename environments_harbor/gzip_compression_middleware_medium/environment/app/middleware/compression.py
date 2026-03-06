import gzip
import io
from functools import wraps
from flask import request

class CompressionMiddleware:
    """
    Middleware to compress Flask responses with gzip when appropriate.
    Checks client Accept-Encoding header and response size before compressing.
    """
    
    def __init__(self, app=None, minimum_size=1024):
        self.minimum_size = minimum_size
        self.skip_content_types = {
            'image/jpeg', 'image/png', 'image/gif', 'image/webp',
            'video/mp4', 'video/mpeg', 'video/webm',
            'application/zip', 'application/gzip', 'application/x-gzip'
        }
        if app:
            self.init_app(app)
    
    def init_app(self, app):
        """Initialize the middleware with a Flask app."""
        app.after_request(self.compress_response)
    
    def should_compress(self, response):
        """
        Determine if response should be compressed based on:
        - Client acceptance of gzip encoding
        - Response content type
        - Response size
        """
        # Check if client accepts gzip encoding
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return False
        
        # Skip if already encoded
        if response.headers.get('Content-Encoding'):
            return False
        
        # Skip non-compressible content types
        content_type = response.headers.get('Content-Type', '')
        if any(ct in content_type for ct in self.skip_content_types):
            return False
        
        # BUG 2: Wrong comparison - checking if size is less than minimum
        # Should be checking if greater than or equal to minimum_size
        # This causes all responses to be skipped regardless of size
        response_data = response.get_data()
        if len(response_data) < self.minimum_size:
            return False
        
        return True
    
    def compress_response(self, response):
        """
        Compress the response data if conditions are met.
        """
        if not self.should_compress(response):
            return response
        
        # Get original response data
        original_data = response.get_data()
        
        # BUG 3: Using wrong mode for BytesIO buffer
        # Should use 'wb' mode but using 'w' which expects text not bytes
        # This causes the gzip compression to fail or produce corrupted data
        compressed_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode='w') as gzip_file:
            gzip_file.write(original_data)
        
        # Get compressed data
        compressed_data = compressed_buffer.getvalue()
        
        # Set the compressed data as response
        response.set_data(compressed_data)
        
        # BUG 1: Setting wrong header value
        # Should be 'gzip' but setting 'compressed' which is not a valid encoding
        response.headers['Content-Encoding'] = 'compressed'
        
        # Update Content-Length header
        response.headers['Content-Length'] = len(compressed_data)
        
        # Add Vary header to indicate response varies by Accept-Encoding
        vary = response.headers.get('Vary', '')
        if vary:
            if 'Accept-Encoding' not in vary:
                response.headers['Vary'] = f"{vary}, Accept-Encoding"
        else:
            response.headers['Vary'] = 'Accept-Encoding'
        
        return response


def compress(minimum_size=1024):
    """
    Decorator for compressing individual route responses.
    Alternative to using middleware for specific routes.
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            response = f(*args, **kwargs)
            
            accept_encoding = request.headers.get('Accept-Encoding', '')
            if 'gzip' not in accept_encoding.lower():
                return response
            
            if hasattr(response, 'get_data'):
                data = response.get_data()
            else:
                data = response.encode('utf-8')
            
            if len(data) < minimum_size:
                return response
            
            compressed_buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=compressed_buffer, mode='w') as gzip_file:
                gzip_file.write(data)
            
            compressed_data = compressed_buffer.getvalue()
            
            if hasattr(response, 'set_data'):
                response.set_data(compressed_data)
                response.headers['Content-Encoding'] = 'compressed'
                response.headers['Content-Length'] = len(compressed_data)
            
            return response
        return decorated_function
    return decorator