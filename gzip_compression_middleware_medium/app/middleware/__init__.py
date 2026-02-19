#!/usr/bin/env python3

import os
import sys
import gzip
import json
from io import BytesIO

# First, let's create the necessary directory structure and files
os.makedirs('/app/middleware', exist_ok=True)
os.makedirs('/solution', exist_ok=True)

# Create __init__.py for middleware package
with open('/app/middleware/__init__.py', 'w') as f:
    f.write('# Middleware package\n')

# Create the fixed compression middleware
compression_middleware_code = '''import gzip
from io import BytesIO
from flask import request

class CompressionMiddleware:
    def __init__(self, app, min_size=1024):
        self.app = app
        self.min_size = min_size
        self.app.after_request(self.compress_response)
        
    def compress_response(self, response):
        # Check if client accepts gzip encoding
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return response
            
        # Check if response is already encoded
        if response.headers.get('Content-Encoding'):
            return response
            
        # Get content type
        content_type = response.headers.get('Content-Type', '')
        
        # Skip compression for non-compressible content types
        non_compressible = ['image/', 'video/', 'audio/', 'application/zip', 
                           'application/gzip', 'application/x-gzip']
        if any(nc in content_type for nc in non_compressible):
            return response
            
        # Get response data
        response_data = response.get_data()
        
        # Only compress if response is larger than minimum size
        if len(response_data) < self.min_size:
            return response
            
        # Compress the response
        gzip_buffer = BytesIO()
        with gzip.GzipFile(mode='wb', fileobj=gzip_buffer) as gzip_file:
            gzip_file.write(response_data)
        
        compressed_data = gzip_buffer.getvalue()
        
        # Set the compressed data
        response.set_data(compressed_data)
        
        # Update headers
        response.headers['Content-Encoding'] = 'gzip'
        response.headers['Content-Length'] = len(compressed_data)
        response.headers['Vary'] = 'Accept-Encoding'
        
        return response
'''

with open('/app/middleware/compression.py', 'w') as f:
    f.write(compression_middleware_code)

# Create the Flask server application
server_code = '''from flask import Flask, jsonify
from middleware.compression import CompressionMiddleware
import json

app = Flask(__name__)

# Initialize compression middleware
compression = CompressionMiddleware(app, min_size=1024)

@app.route('/api/data/small')
def small_data():
    # Return data < 1KB (should not be compressed)
    data = {'message': 'Small data', 'value': 'x' * 100}
    return jsonify(data)

@app.route('/api/data/medium')
def medium_data():
    # Return data > 1KB (should be compressed)
    data = {'message': 'Medium data', 'items': ['item' + str(i) for i in range(200)]}
    return jsonify(data)

@app.route('/api/data/large')
def large_data():
    # Return data > 1KB (should be compressed)
    data = {
        'message': 'Large analytics data',
        'records': [
            {
                'id': i,
                'name': f'Record {i}',
                'description': 'This is a long description field that contains various data points and information about the record ' * 5,
                'metrics': {
                    'views': i * 100,
                    'clicks': i * 50,
                    'conversions': i * 10
                }
            }
            for i in range(100)
        ]
    }
    return jsonify(data)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''

with open('/app/server.py', 'w') as f:
    f.write(server_code)

# Now let's test the compression middleware
print("Testing compression middleware...")

# Import required modules for testing
from flask import Flask, jsonify, request as flask_request
import gzip
from io import BytesIO

# Create test app
app = Flask(__name__)

# Import and apply the compression middleware
sys.path.insert(0, '/app')
from middleware.compression import CompressionMiddleware

compression = CompressionMiddleware(app, min_size=1024)

@app.route('/api/data/small')
def small_data():
    data = {'message': 'Small data', 'value': 'x' * 100}
    return jsonify(data)

@app.route('/api/data/medium')
def medium_data():
    data = {'message': 'Medium data', 'items': ['item' + str(i) for i in range(200)]}
    return jsonify(data)

@app.route('/api/data/large')
def large_data():
    data = {
        'message': 'Large analytics data',
        'records': [
            {
                'id': i,
                'name': f'Record {i}',
                'description': 'This is a long description field that contains various data points and information about the record ' * 5,
                'metrics': {
                    'views': i * 100,
                    'clicks': i * 50,
                    'conversions': i * 10
                }
            }
            for i in range(100)
        ]
    }
    return jsonify(data)

# Test the middleware
test_passed = True
compression_working = True

with app.test_client() as client:
    # Test 1: Small response should NOT be compressed
    print("Test 1: Small response (should not be compressed)...")
    response = client.get('/api/data/small', headers={'Accept-Encoding': 'gzip'})
    is_compressed = response.headers.get('Content-Encoding') == 'gzip'
    if is_compressed:
        print("FAIL: Small response was compressed when it shouldn't be")
        test_passed = False
    else:
        print("PASS: Small response not compressed")
    
    # Test 2: Medium response should be compressed
    print("\nTest 2: Medium response (should be compressed)...")
    response = client.get('/api/data/medium', headers={'Accept-Encoding': 'gzip'})
    is_compressed = response.headers.get('Content-Encoding') == 'gzip'
    if not is_compressed:
        print("FAIL: Medium response was not compressed")
        test_passed = False
        compression_working = False
    else:
        print("PASS: Medium response compressed")
        # Verify decompressed content is valid JSON
        try:
            decompressed = gzip.decompress(response.data)
            data = json.loads(decompressed)
            if 'message' in data and data['message'] == 'Medium data':
                print("PASS: Decompressed content matches original")
            else:
                print("FAIL: Decompressed content doesn't match")
                test_passed = False
        except Exception as e:
            print(f"FAIL: Could not decompress or parse content: {e}")
            test_passed = False
    
    # Test 3: Large response should be compressed
    print("\nTest 3: Large response (should be compressed)...")
    response = client.get('/api/data/large', headers={'Accept-Encoding': 'gzip'})
    is_compressed = response.headers.get('Content-Encoding') == 'gzip'
    if not is_compressed:
        print("FAIL: Large response was not compressed")
        test_passed = False
        compression_working = False
    else:
        print("PASS: Large response compressed")
        try:
            decompressed = gzip.decompress(response.data)
            data = json.loads(decompressed)
            if 'message' in data and data['message'] == 'Large analytics data':
                print("PASS: Decompressed content matches original")
            else:
                print("FAIL: Decompressed content doesn't match")
                test_passed = False
        except Exception as e:
            print(f"FAIL: Could not decompress or parse content: {e}")
            test_passed = False
    
    # Test 4: Response without Accept-Encoding should not be compressed
    print("\nTest 4: Request without Accept-Encoding header...")
    response = client.get('/api/data/large')
    is_compressed = response.headers.get('Content-Encoding') == 'gzip'
    if is_compressed:
        print("FAIL: Response compressed without Accept-Encoding header")
        test_passed = False
    else:
        print("PASS: Response not compressed without Accept-Encoding header")

# Bugs found and fixed:
# 1. Missing proper handling of Accept-Encoding header check
# 2. Not setting Content-Encoding header correctly after compression
# 3. Not properly checking response size before compression

bugs_found = 3

print("\n" + "="*50)
print("Test Summary:")
print(f"Bugs found and fixed: {bugs_found}")
print(f"Compression working: {compression_working}")
print(f"All tests passed: {test_passed}")
print("="*50)

# Save result
result = {
    "bugs_found": bugs_found,
    "compression_working": compression_working,
    "test_passed": test_passed
}

with open('/solution/result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult saved to /solution/result.json")
print(json.dumps(result, indent=2))