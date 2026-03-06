#!/usr/bin/env python3

import http.server
import ssl
import socketserver
import json
import sys

from config import HOST, PORT, CERTFILE, KEYFILE

class StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"status": "ok", "message": "Server is running"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {"error": "Not found"}
            self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.address_string(),
                          self.log_date_time_string(),
                          format%args))

def main():
    try:
        server = http.server.HTTPServer((HOST, PORT), StatusHandler)
        
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=CERTFILE, keyfile=KEYFILE)
        
        server.socket = context.wrap_socket(server.socket, server_side=True)
        
        print(f"HTTPS server running on https://{HOST}:{PORT}")
        print(f"Using certificate: {CERTFILE}")
        print(f"Using key: {KEYFILE}")
        
        server.serve_forever()
        
    except FileNotFoundError as e:
        print(f"Error: Certificate or key file not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ssl.SSLError as e:
        print(f"SSL Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()