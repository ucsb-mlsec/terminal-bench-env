#!/usr/bin/env python3

import requests
import os

# Define the test video path
test_video_path = 'test_data/sample.mp4'

# Read the original file in binary mode
with open(test_video_path, 'rb') as f:
    file_data = f.read()

# Get the filename
filename = os.path.basename(test_video_path)

# Send POST request with multipart/form-data
files = {'video': (filename, file_data, 'video/mp4')}
response = requests.post('http://localhost:5000/upload', files=files)

# Read the saved file
uploaded_file_path = 'uploads/sample.mp4'
with open(uploaded_file_path, 'rb') as f:
    uploaded_data = f.read()

# Perform byte-level comparison
print('Upload completed')
if file_data == uploaded_data:
    print('Integrity check: PASS')
else:
    print('Integrity check: FAIL')