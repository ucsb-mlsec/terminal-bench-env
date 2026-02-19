#!/usr/bin/env python3

import http.server
import socketserver
import sys

PORT = 3000
HOST = '127.0.0.1'

class BackendHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        message = 'Backend service running on localhost:3000\n'
        self.wfile.write(message.encode())
    
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        response = f'Backend received POST request. Body length: {content_length}\n'
        self.wfile.write(response.encode())
    
    def log_message(self, format, *args):
        sys.stdout.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format%args))

def run_server():
    with socketserver.TCPServer((HOST, PORT), BackendHandler) as httpd:
        httpd.allow_reuse_address = True
        print(f"Backend server running on {HOST}:{PORT}")
        sys.stdout.flush()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            httpd.shutdown()

if __name__ == '__main__':
    run_server()