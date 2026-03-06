#!/usr/bin/env python3

import json
import os
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

CONFIG_PATH = "/config/app.json"
config = {}
config_lock = threading.RLock()
config_mtime = None


def load_config():
    """Load configuration from file."""
    global config, config_mtime
    
    try:
        with open(CONFIG_PATH, 'r') as f:
            new_config = json.load(f)
        
        # Validate required fields
        required_fields = ['timeout', 'feature_enabled', 'api_endpoint']
        for field in required_fields:
            if field not in new_config:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate types
        if not isinstance(new_config['timeout'], int):
            raise ValueError("timeout must be an integer")
        if not isinstance(new_config['feature_enabled'], bool):
            raise ValueError("feature_enabled must be a boolean")
        if not isinstance(new_config['api_endpoint'], str):
            raise ValueError("api_endpoint must be a string")
        
        with config_lock:
            config.clear()
            config.update(new_config)
            config_mtime = os.path.getmtime(CONFIG_PATH)
        
        print(f"Configuration loaded successfully: {config}")
        return True
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False


def watch_config():
    """Watch configuration file for changes and reload when modified."""
    global config_mtime
    
    while True:
        try:
            current_mtime = os.path.getmtime(CONFIG_PATH)
            
            if config_mtime is None or current_mtime != config_mtime:
                print(f"Configuration file change detected (mtime: {current_mtime})")
                load_config()
            
        except Exception as e:
            print(f"Error checking configuration file: {e}")
        
        time.sleep(1)


class RequestHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        """Override to reduce log noise."""
        pass
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path in ['/', '/status']:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            
            with config_lock:
                response = {
                    "status": "ok",
                    "timeout": config.get('timeout', 0),
                    "feature_enabled": config.get('feature_enabled', False),
                    "api_endpoint": config.get('api_endpoint', '')
                }
            
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')


def main():
    """Main server setup and execution."""
    # Load initial configuration
    if not load_config():
        print("Failed to load initial configuration. Exiting.")
        return
    
    # Start configuration watcher thread
    watcher_thread = threading.Thread(target=watch_config, daemon=True)
    watcher_thread.start()
    
    # Start HTTP server
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f"Server running on port 8080...")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == '__main__':
    main()