#!/usr/bin/env python3

import os
import json
import time
import threading
from flask import Flask, jsonify

app = Flask(__name__)

# Global variable to store current configuration
current_config = {}
config_dir = '/etc/config/'

def load_config():
    """
    Load configuration from files in the config directory.
    Reads all files in /etc/config/ and stores their contents.
    """
    global current_config
    new_config = {}
    
    try:
        if not os.path.exists(config_dir):
            print(f"Config directory {config_dir} does not exist")
            return
        
        files = os.listdir(config_dir)
        print(f"Found config files: {files}")
        
        for filename in files:
            filepath = os.path.join(config_dir, filename)
            
            # Skip directories and hidden files
            if os.path.isdir(filepath) or filename.startswith('.'):
                continue
            
            try:
                with open(filepath, 'r') as f:
                    content = f.read().strip()
                    mtime = os.path.getmtime(filepath)
                    new_config[filename] = {
                        'value': content,
                        'modified_time': mtime,
                        'modified_readable': time.ctime(mtime)
                    }
                    print(f"Loaded {filename}: {content[:50]}... (modified: {time.ctime(mtime)})")
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
        
        current_config = new_config
        print(f"Configuration loaded successfully: {len(new_config)} items")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")

def config_reloader():
    """
    Background thread that periodically reloads configuration.
    Checks for file changes every 10 seconds by comparing modification times.
    """
    print("Starting configuration reloader thread...")
    last_mtimes = {}
    
    while True:
        try:
            time.sleep(10)  # Check every 10 seconds
            
            if not os.path.exists(config_dir):
                continue
            
            files = os.listdir(config_dir)
            current_mtimes = {}
            
            for filename in files:
                filepath = os.path.join(config_dir, filename)
                if os.path.isfile(filepath) and not filename.startswith('.'):
                    current_mtimes[filename] = os.path.getmtime(filepath)
            
            # Check if any file has been modified
            if current_mtimes != last_mtimes:
                print("Configuration change detected! Reloading...")
                load_config()
                last_mtimes = current_mtimes
            else:
                print("No configuration changes detected")
                
        except Exception as e:
            print(f"Error in config reloader: {e}")

@app.route('/')
def get_config():
    """
    Return current configuration as JSON.
    """
    return jsonify({
        'config': current_config,
        'timestamp': time.time(),
        'message': 'Current application configuration'
    })

@app.route('/health')
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        'status': 'OK',
        'timestamp': time.time()
    })

if __name__ == '__main__':
    print("Starting application...")
    
    # Load initial configuration
    load_config()
    
    # Start background thread for config reloading
    reloader_thread = threading.Thread(target=config_reloader, daemon=True)
    reloader_thread.start()
    
    # Start Flask application
    print("Starting Flask server on 0.0.0.0:8080")
    app.run(host='0.0.0.0', port=8080)