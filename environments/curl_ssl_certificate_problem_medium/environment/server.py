#!/usr/bin/env python3

import http.server
import ssl
import json
import time
from urllib.parse import urlparse

class StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/status':
            response_data = {
                "status": "operational",
                "timestamp": int(time.time())
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def log_message(self, format, *args):
        return

if __name__ == '__main__':
    server_address = ('localhost', 8443)
    httpd = http.server.HTTPServer(server_address, StatusHandler)
    
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(
        certfile='/home/agent/ssl/server.crt',
        keyfile='/home/agent/ssl/server.key'
    )
    
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    
    print(f"HTTPS server running on https://{server_address[0]}:{server_address[1]}")
    
    httpd.serve_forever()