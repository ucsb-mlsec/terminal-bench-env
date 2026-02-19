#!/usr/bin/env python3

import os
import json
import hashlib
import subprocess
import shutil
from pathlib import Path

def compute_sha256(filepath):
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    # Set up paths
    project_dir = Path("/workspace/pyapp")
    dist_dir = project_dir / "dist"
    config_file = project_dir / "deployment-config.json"
    result_file = Path("/workspace/verification_result.txt")
    
    # Clean dist directory if it exists
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    
    # Change to project directory
    os.chdir(project_dir)
    
    # Build the wheel
    print("Building wheel package...")
    subprocess.run(
        ["python", "setup.py", "bdist_wheel"],
        check=True,
        capture_output=True
    )
    
    # Find the wheel file
    wheel_files = list(dist_dir.glob("*.whl"))
    if len(wheel_files) != 1:
        raise RuntimeError(f"Expected exactly one wheel file, found {len(wheel_files)}")
    
    wheel_file = wheel_files[0]
    print(f"Found wheel file: {wheel_file.name}")
    
    # Compute hash of the wheel file
    actual_hash = compute_sha256(wheel_file)
    print(f"Computed hash: {actual_hash}")
    
    # Read expected hash from config
    with open(config_file, "r") as f:
        config = json.load(f)
    
    expected_hash = config["expected_wheel_hash"].lower()
    print(f"Expected hash: {expected_hash}")
    
    # Compare hashes
    match = "YES" if actual_hash == expected_hash else "NO"
    print(f"Match: {match}")
    
    # Write results
    with open(result_file, "w") as f:
        f.write(f"EXPECTED={expected_hash}\n")
        f.write(f"ACTUAL={actual_hash}\n")
        f.write(f"MATCH={match}\n")
    
    print(f"\nVerification results written to {result_file}")

if __name__ == "__main__":
    main()