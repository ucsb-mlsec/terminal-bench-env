#!/bin/bash
set -e

echo "Building numlib..."

# Remove old build artifacts
rm -rf build/ dist/ *.egg-info

# Build the extension module
python setup.py build_ext --inplace

echo "Build complete"