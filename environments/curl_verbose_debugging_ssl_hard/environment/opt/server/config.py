#!/usr/bin/env python3

import ssl
import http.server
import socketserver
from http import HTTPStatus

# Server configuration
HOST = 'localhost'
PORT = 8443
CERT_DIR = '/opt/certs/'

# SSL certificate configuration
# WARNING: Incomplete certificate chain configuration
# Only server certificate is specified, missing intermediate CA certificate
CERTFILE = '/opt/certs/server.pem'
KEYFILE = '/opt/certs/server-key.pem'

class StatusHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP handler that returns status"""
    
    def do_GET(self):
        if self.path == '/status':
            self.send_response(HTTPStatus.OK)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "message": "Server is running"}')
        else:
            self.send_response(HTTPStatus.NOT_FOUND)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Override to customize logging"""
        print(f"[{self.address_string()}] {format % args}")

def run_server():
    """Start the HTTPS server with SSL configuration"""
    
    # Create SSL context
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # Load certificate and key
    # PROBLEM: This only loads the server certificate without the intermediate CA
    # A complete chain should include both server cert and intermediate cert
    context.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)
    
    # Create server
    with socketserver.TCPServer((HOST, PORT), StatusHandler) as httpd:
        httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
        print(f"HTTPS Server running on https://{HOST}:{PORT}")
        print(f"Certificate: {CERTFILE}")
        print(f"Key: {KEYFILE}")
        httpd.serve_forever()

if __name__ == '__main__':
    run_server()