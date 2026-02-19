#!/usr/bin/env python3

import os
import subprocess
import sys

# Change to the workspace directory
os.chdir('/workspace/mathlib')

# The issue is that matrix_ops.pyx uses C++ features (libcpp.vector and libcpp.algorithm)
# but the distutils directive at the top says "language = c"
# We need to change it to "language = c++" or configure setup.py to use C++

# First, let's read the current matrix_ops.pyx
with open('matrix_ops.pyx', 'r') as f:
    content = f.read()

# Fix: Change the language directive from c to c++
content = content.replace('# distutils: language = c', '# distutils: language = c++')

# Write the fixed content back
with open('matrix_ops.pyx', 'w') as f:
    f.write(content)

# Clean any previous build artifacts
subprocess.run(['python', 'setup.py', 'clean', '--all'], capture_output=True)

# Build the extension
build_result = subprocess.run(
    ['python', 'setup.py', 'build_ext', '--inplace'],
    capture_output=True,
    text=True
)

if build_result.returncode != 0:
    print("Build failed:", file=sys.stderr)
    print(build_result.stdout, file=sys.stderr)
    print(build_result.stderr, file=sys.stderr)
    with open('/workspace/solution.txt', 'w') as f:
        f.write('language=cpp\n')
        f.write('status=failed\n')
    sys.exit(1)

# Test the import
test_result = subprocess.run(
    ['python', 'test_import.py'],
    capture_output=True,
    text=True
)

if test_result.returncode == 0:
    print("Test passed!")
    print(test_result.stdout)
    with open('/workspace/solution.txt', 'w') as f:
        f.write('language=cpp\n')
        f.write('status=success\n')
else:
    print("Test failed:", file=sys.stderr)
    print(test_result.stdout, file=sys.stderr)
    print(test_result.stderr, file=sys.stderr)
    with open('/workspace/solution.txt', 'w') as f:
        f.write('language=cpp\n')
        f.write('status=failed\n')
    sys.exit(1)