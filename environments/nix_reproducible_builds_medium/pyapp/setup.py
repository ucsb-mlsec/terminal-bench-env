#!/usr/bin/env python3

import os
import json
import hashlib
import subprocess
import glob

# Change to project directory
os.chdir('/workspace/pyapp')

# Clean any previous builds
if os.path.exists('dist'):
    import shutil
    shutil.rmtree('dist')
if os.path.exists('build'):
    import shutil
    shutil.rmtree('build')
if os.path.exists('src/pyapp.egg-info'):
    import shutil
    shutil.rmtree('src/pyapp.egg-info')

# Build the wheel
subprocess.run(['python', 'setup.py', 'bdist_wheel'], check=True)

# Find the wheel file
wheel_files = glob.glob('dist/*.whl')
if not wheel_files:
    raise FileNotFoundError("No wheel file found in dist/")
wheel_file = wheel_files[0]

# Compute SHA256 hash of the wheel file
sha256_hash = hashlib.sha256()
with open(wheel_file, 'rb') as f:
    for chunk in iter(lambda: f.read(4096), b''):
        sha256_hash.update(chunk)
actual_hash = sha256_hash.hexdigest()

# Read expected hash from deployment-config.json
with open('/workspace/pyapp/deployment-config.json', 'r') as f:
    config = json.load(f)
expected_hash = config['expected_wheel_hash']

# Determine if hashes match
match = "YES" if expected_hash.lower() == actual_hash.lower() else "NO"

# Write results to verification_result.txt
with open('/workspace/verification_result.txt', 'w') as f:
    f.write(f"EXPECTED={expected_hash.lower()}\n")
    f.write(f"ACTUAL={actual_hash.lower()}\n")
    f.write(f"MATCH={match}\n")

print(f"Verification complete. Results written to /workspace/verification_result.txt")
print(f"Expected: {expected_hash.lower()}")
print(f"Actual: {actual_hash.lower()}")
print(f"Match: {match}")