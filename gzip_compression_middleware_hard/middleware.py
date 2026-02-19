import gzip
import io

# Common content types that are already compressed
COMPRESSED_CONTENT_TYPES = [
    'image/jpeg',
    'image/png',
    'image/gif',
    'video/mp4',
    'application/zip',
    'application/gzip',
    'application/x-gzip'
]

class Response:
    """Simple response object for testing"""
    def __init__(self, body, status_code=200, headers=None):
        self.body = body if isinstance(body, bytes) else body.encode('utf-8')
        self.status_code = status_code
        self.headers = headers or {}


def compress_response(response, request_headers):
    """
    Middleware function to compress HTTP responses with gzip.
    
    Args:
        response: Response object with body, status_code, and headers
        request_headers: Dictionary of request headers
        
    Returns:
        Response object (compressed if appropriate)
    """
    # Check if client supports gzip
    accept_encoding = request_headers.get('Accept-Encoding', '')
    if 'gzip' not in accept_encoding.lower():
        return response
    
    # Get response body as bytes
    body = response.body
    if isinstance(body, str):
        body = body.encode('utf-8')
    
    # Check if content is already compressed
    content_type = response.headers.get('Content-Type', '')
    content_encoding = response.headers.get('Content-Encoding', '')
    
    # Don't compress if already compressed by content type
    for compressed_type in COMPRESSED_CONTENT_TYPES:
        if compressed_type in content_type.lower():
            return response
    
    # Don't compress if Content-Encoding already set
    if content_encoding:
        return response
    
    # Only compress if body is >= 1024 bytes (1KB)
    if len(body) < 1024:
        return response
    
    # Perform gzip compression
    try:
        compressed_buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=compressed_buffer, mode='wb') as gz_file:
            gz_file.write(body)
        
        compressed_body = compressed_buffer.getvalue()
        
        # Update response
        response.body = compressed_body
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = str(len(compressed_body))
        
        # Remove Vary header issue - add Vary: Accept-Encoding
        vary = response.headers.get('Vary', '')
        if vary:
            if 'Accept-Encoding' not in vary:
                response.headers['Vary'] = vary + ', Accept-Encoding'
        else:
            response.headers['Vary'] = 'Accept-Encoding'
        
        return response
        
    except Exception as e:
        # If compression fails, return original response
        return response


def gzip_middleware(request_headers):
    """
    Returns a middleware function that can be used to wrap response handling.
    
    Args:
        request_headers: Dictionary of request headers
        
    Returns:
        Function that takes a response and returns compressed response
    """
    def middleware(response):
        return compress_response(response, request_headers)
    return middleware