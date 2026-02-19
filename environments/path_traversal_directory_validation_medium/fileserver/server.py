#!/usr/bin/env python3

import os
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

PUBLIC_DIR = "/var/www/public"

class VulnerableFileServer(BaseHTTPRequestHandler):
    """
    Intentionally vulnerable file server with multiple path traversal issues
    """
    
    def do_GET(self):
        # Parse the URL path
        parsed_path = urllib.parse.urlparse(self.path)
        requested_file = parsed_path.path
        
        # Remove leading slash
        if requested_file.startswith('/'):
            requested_file = requested_file[1:]
        
        # VULNERABILITY 1: Direct concatenation without validation
        # This allows ../../../etc/passwd type attacks
        file_path = os.path.join(PUBLIC_DIR, requested_file)
        
        # VULNERABILITY 2: No check for absolute paths
        # Absolute paths like /etc/passwd are not blocked
        
        # VULNERABILITY 3: No URL decoding validation
        # URL-encoded sequences like ..%2F are not handled
        
        # VULNERABILITY 4: No double-encoding protection
        # Double-encoded sequences pass through
        
        # VULNERABILITY 5: No null byte injection protection
        # Paths with null bytes like ../../etc/passwd%00.jpg are not sanitized
        
        # VULNERABILITY 6: No symbolic link checking
        # Symlinks can point outside PUBLIC_DIR
        
        # VULNERABILITY 7: No path canonicalization
        # Paths are not resolved to their canonical form
        
        try:
            # Try to open and serve the file
            with open(file_path, 'rb') as f:
                content = f.read()
                
            self.send_response(200)
            self.send_header('Content-type', self.guess_type(file_path))
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
            
        except FileNotFoundError:
            self.send_error(404, "File not found")
        except IsADirectoryError:
            self.send_error(400, "Path is a directory")
        except PermissionError:
            self.send_error(403, "Permission denied")
        except Exception as e:
            self.send_error(500, f"Internal server error: {str(e)}")
    
    def guess_type(self, path):
        """Simple content type guesser"""
        if path.endswith('.html'):
            return 'text/html'
        elif path.endswith('.css'):
            return 'text/css'
        elif path.endswith('.js'):
            return 'application/javascript'
        elif path.endswith('.json'):
            return 'application/json'
        elif path.endswith('.png'):
            return 'image/png'
        elif path.endswith('.jpg') or path.endswith('.jpeg'):
            return 'image/jpeg'
        elif path.endswith('.txt'):
            return 'text/plain'
        else:
            return 'application/octet-stream'
    
    def log_message(self, format, *args):
        """Override to reduce logging noise"""
        return

def run_server(port=8000):
    """Start the vulnerable file server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, VulnerableFileServer)
    print(f"Starting vulnerable file server on port {port}")
    print(f"Serving files from: {PUBLIC_DIR}")
    print("WARNING: This server is intentionally vulnerable for testing purposes")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    # Ensure public directory exists
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    
    # Create a test file for legitimate access
    test_file = os.path.join(PUBLIC_DIR, 'index.html')
    if not os.path.exists(test_file):
        with open(test_file, 'w') as f:
            f.write('<html><body><h1>Welcome to the File Server</h1></body></html>')
    
    # Start the server
    run_server()