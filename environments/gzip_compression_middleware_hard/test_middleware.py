#!/usr/bin/env python3

import unittest
import sys
import os
import gzip
import io

# Mock Response class to simulate HTTP responses
class Response:
    def __init__(self, body, status_code=200, headers=None):
        self.status_code = status_code
        self.headers = headers if headers else {}
        self.body = body if isinstance(body, bytes) else body.encode('utf-8')
    
    def get_body(self):
        return self.body
    
    def set_body(self, body):
        self.body = body if isinstance(body, bytes) else body.encode('utf-8')
    
    def get_header(self, key):
        return self.headers.get(key)
    
    def set_header(self, key, value):
        self.headers[key] = value
    
    def remove_header(self, key):
        if key in self.headers:
            del self.headers[key]


# Import the middleware function
try:
    from middleware import compress_response
except ImportError:
    print("Error: Could not import compress_response from middleware.py")
    sys.exit(1)


class TestCompressionMiddleware(unittest.TestCase):
    
    def test_large_response_compression(self):
        """Test that responses >= 1KB are compressed when client supports gzip"""
        # Create a large response (2KB)
        large_body = "A" * 2048
        response = Response(large_body, headers={'Content-Type': 'text/html'})
        request_headers = {'Accept-Encoding': 'gzip, deflate'}
        
        # Apply compression middleware
        compress_response(response, request_headers)
        
        # Verify response is compressed
        compressed_body = response.get_body()
        self.assertTrue(len(compressed_body) < len(large_body), 
                       "Compressed body should be smaller than original")
        
        # Verify Content-Encoding header is set
        self.assertEqual(response.get_header('Content-Encoding'), 'gzip',
                        "Content-Encoding header should be set to gzip")
        
        # Verify we can decompress and get original data
        decompressed = gzip.decompress(compressed_body).decode('utf-8')
        self.assertEqual(decompressed, large_body,
                        "Decompressed data should match original")
    
    def test_small_response_not_compressed(self):
        """Test that responses < 1KB are NOT compressed"""
        # Create a small response (500 bytes)
        small_body = "B" * 500
        response = Response(small_body, headers={'Content-Type': 'text/html'})
        request_headers = {'Accept-Encoding': 'gzip, deflate'}
        
        # Apply compression middleware
        compress_response(response, request_headers)
        
        # Verify response is NOT compressed
        body = response.get_body()
        self.assertEqual(body.decode('utf-8'), small_body,
                        "Small response should not be compressed")
        
        # Verify Content-Encoding header is NOT set
        self.assertIsNone(response.get_header('Content-Encoding'),
                         "Content-Encoding should not be set for small responses")
    
    def test_no_gzip_support(self):
        """Test that large responses are not compressed when client doesn't support gzip"""
        # Create a large response
        large_body = "C" * 2048
        response = Response(large_body, headers={'Content-Type': 'text/plain'})
        request_headers = {'Accept-Encoding': 'deflate'}  # No gzip
        
        # Apply compression middleware
        compress_response(response, request_headers)
        
        # Verify response is NOT compressed
        body = response.get_body()
        self.assertEqual(body.decode('utf-8'), large_body,
                        "Response should not be compressed when client doesn't support gzip")
        
        # Verify Content-Encoding header is NOT set
        self.assertIsNone(response.get_header('Content-Encoding'),
                         "Content-Encoding should not be set when gzip not supported")
    
    def test_already_compressed_content(self):
        """Test that already-compressed content types are not compressed again"""
        # Test with image/jpeg
        large_body = "D" * 2048
        response = Response(large_body, headers={'Content-Type': 'image/jpeg'})
        request_headers = {'Accept-Encoding': 'gzip'}
        
        compress_response(response, request_headers)
        
        body = response.get_body()
        self.assertEqual(body.decode('utf-8'), large_body,
                        "JPEG images should not be compressed")
        self.assertIsNone(response.get_header('Content-Encoding'),
                         "Content-Encoding should not be set for JPEG")
        
        # Test with image/png
        response2 = Response(large_body, headers={'Content-Type': 'image/png'})
        compress_response(response2, request_headers)
        self.assertEqual(response2.get_body().decode('utf-8'), large_body,
                        "PNG images should not be compressed")
        
        # Test with application/zip
        response3 = Response(large_body, headers={'Content-Type': 'application/zip'})
        compress_response(response3, request_headers)
        self.assertEqual(response3.get_body().decode('utf-8'), large_body,
                        "ZIP files should not be compressed")
    
    def test_content_integrity(self):
        """Test that compressed data can be decompressed without corruption"""
        # Create response with specific pattern
        original_body = "Test data: " + "XYZ123" * 200
        response = Response(original_body, headers={'Content-Type': 'text/plain'})
        request_headers = {'Accept-Encoding': 'gzip'}
        
        # Apply compression
        compress_response(response, request_headers)
        
        # Decompress and verify
        compressed_body = response.get_body()
        decompressed = gzip.decompress(compressed_body).decode('utf-8')
        self.assertEqual(decompressed, original_body,
                        "Decompressed content must match original exactly")
    
    def test_content_length_header(self):
        """Test that Content-Length header is updated after compression"""
        large_body = "E" * 2048
        response = Response(large_body, headers={'Content-Type': 'text/html'})
        request_headers = {'Accept-Encoding': 'gzip'}
        
        # Apply compression
        compress_response(response, request_headers)
        
        # Verify Content-Length matches compressed body length
        compressed_body = response.get_body()
        content_length = response.get_header('Content-Length')
        
        if content_length:
            self.assertEqual(int(content_length), len(compressed_body),
                           "Content-Length should match compressed body length")
    
    def test_json_compression(self):
        """Test that JSON responses are compressed"""
        # Create large JSON response
        json_body = '{"data": "' + "F" * 2000 + '"}'
        response = Response(json_body, headers={'Content-Type': 'application/json'})
        request_headers = {'Accept-Encoding': 'gzip'}
        
        # Apply compression
        compress_response(response, request_headers)
        
        # Verify compression
        compressed_body = response.get_body()
        self.assertTrue(len(compressed_body) < len(json_body),
                       "JSON should be compressed")
        self.assertEqual(response.get_header('Content-Encoding'), 'gzip',
                        "Content-Encoding should be gzip for JSON")
        
        # Verify integrity
        decompressed = gzip.decompress(compressed_body).decode('utf-8')
        self.assertEqual(decompressed, json_body,
                        "Decompressed JSON should match original")
    
    def test_html_compression(self):
        """Test that HTML responses are compressed"""
        # Create large HTML response
        html_body = "<html><body>" + "G" * 2000 + "</body></html>"
        response = Response(html_body, headers={'Content-Type': 'text/html; charset=utf-8'})
        request_headers = {'Accept-Encoding': 'gzip'}
        
        # Apply compression
        compress_response(response, request_headers)
        
        # Verify compression
        compressed_body = response.get_body()
        self.assertTrue(len(compressed_body) < len(html_body),
                       "HTML should be compressed")
        self.assertEqual(response.get_header('Content-Encoding'), 'gzip',
                        "Content-Encoding should be gzip for HTML")
    
    def test_accept_encoding_case_insensitive(self):
        """Test that Accept-Encoding header is handled case-insensitively"""
        large_body = "H" * 2048
        response = Response(large_body, headers={'Content-Type': 'text/plain'})
        request_headers = {'accept-encoding': 'GZIP'}  # Different case
        
        # Apply compression
        compress_response(response, request_headers)
        
        # Should still compress
        self.assertEqual(response.get_header('Content-Encoding'), 'gzip',
                        "Should handle Accept-Encoding case-insensitively")
    
    def test_empty_response(self):
        """Test handling of empty responses"""
        empty_body = ""
        response = Response(empty_body, headers={'Content-Type': 'text/plain'})
        request_headers = {'Accept-Encoding': 'gzip'}
        
        # Apply compression
        compress_response(response, request_headers)
        
        # Empty response should not be compressed
        body = response.get_body()
        self.assertEqual(len(body), 0,
                        "Empty response should remain empty")
        self.assertIsNone(response.get_header('Content-Encoding'),
                         "Content-Encoding should not be set for empty response")
    
    def test_boundary_size_1kb(self):
        """Test compression at exactly 1KB boundary"""
        # Exactly 1024 bytes
        body_1kb = "I" * 1024
        response = Response(body_1kb, headers={'Content-Type': 'text/plain'})
        request_headers = {'Accept-Encoding': 'gzip'}
        
        # Apply compression
        compress_response(response, request_headers)
        
        # Should be compressed (>= 1KB)
        self.assertEqual(response.get_header('Content-Encoding'), 'gzip',
                        "1KB response should be compressed")


if __name__ == '__main__':
    unittest.main()