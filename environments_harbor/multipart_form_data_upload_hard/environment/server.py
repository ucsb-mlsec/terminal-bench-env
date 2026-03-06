#!/usr/bin/env python3

from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
    
    video_file = request.files['video']
    
    if video_file.filename == '':
        return jsonify({'status': 'error', 'message': 'Empty filename'}), 400
    
    # BUG: Opening file in text mode instead of binary mode
    filepath = os.path.join(UPLOAD_FOLDER, video_file.filename)
    with open(filepath, 'w') as f:
        f.write(video_file.read().decode('utf-8', errors='replace'))
    
    return jsonify({'status': 'success', 'filename': video_file.filename}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)