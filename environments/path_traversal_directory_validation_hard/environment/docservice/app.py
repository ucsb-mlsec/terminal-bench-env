#!/usr/bin/env python3

from flask import Flask, request, send_file, jsonify
import os

app = Flask(__name__)

DOCUMENT_DIR = '/home/agent/docservice/documents/'

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download a file from the document directory"""
    try:
        file_path = os.path.join(DOCUMENT_DIR, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/view', methods=['GET'])
def view_file():
    """View file contents"""
    try:
        filename = request.args.get('file')
        if not filename:
            return jsonify({'error': 'No file parameter provided'}), 400
        
        file_path = os.path.join(DOCUMENT_DIR, filename)
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
            return jsonify({'content': content}), 200
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload a file to the document directory"""
    try:
        filename = request.form.get('filename')
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        file_path = os.path.join(DOCUMENT_DIR, filename)
        
        file.save(file_path)
        return jsonify({'message': 'File uploaded successfully', 'path': file_path}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Create documents directory if it doesn't exist
    os.makedirs(DOCUMENT_DIR, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)