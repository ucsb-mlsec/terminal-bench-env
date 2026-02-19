#!/usr/bin/env python3

import ssl
import http.server
import socketserver

class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        response = b'<html><body>Hello from HTTPS server</body></html>'
        self.wfile.write(response)
    
    def log_message(self, format, *args):
        return

def main():
    server_address = ('0.0.0.0', 8443)
    httpd = http.server.HTTPServer(server_address, SimpleHTTPRequestHandler)
    
    # Create SSL context for basic HTTPS (NOT mTLS)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # Load server certificate and key
    context.load_cert_chain('certs/server.crt', 'certs/server.key')
    
    # NOTE: Missing client certificate verification configuration
    # This is the bug - no verify_mode set, no CA certificate loaded
    
    # Wrap socket with SSL
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    print(f"Starting HTTPS server on port 8443...")
    print("WARNING: Server is not configured for mutual TLS authentication")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()

if __name__ == '__main__':
    main()