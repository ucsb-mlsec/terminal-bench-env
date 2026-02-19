#!/usr/bin/env python

import os
import subprocess
import sys

# Change to the workspace directory
os.chdir('/workspace/mathlib')

# The problem: The code uses C++ features (libcpp.vector, libcpp.algorithm)
# but the distutils directive says "language = c"
# This causes a compilation failure because C doesn't have these C++ libraries

# Solution: Change the language directive from c to cpp
print("Analyzing the issue...")
print("The matrix_ops.pyx file uses C++ standard library (vector, algorithm)")
print("but is configured to compile as C. This needs to be changed to C++.")

# Read the original file
with open('matrix_ops.pyx', 'r') as f:
    content = f.read()

# Fix: Change language from c to cpp
fixed_content = content.replace('# distutils: language = c', '# distutils: language = c++')

# Write the fixed file
with open('matrix_ops.pyx', 'w') as f:
    f.write(fixed_content)

print("\nFixed matrix_ops.pyx to use C++ compilation")

# Clean any previous builds
print("\nCleaning previous builds...")
subprocess.run(['rm', '-rf', 'build'], stderr=subprocess.DEVNULL)
subprocess.run(['rm', '-f', 'matrix_ops.c', 'matrix_ops.cpp'], stderr=subprocess.DEVNULL)
for file in os.listdir('.'):
    if file.startswith('matrix_ops') and file.endswith('.so'):
        subprocess.run(['rm', '-f', file])

# Build the extension
print("\nBuilding the extension...")
result = subprocess.run(
    [sys.executable, 'setup.py', 'build_ext', '--inplace'],
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("Build failed!")
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    status = "failed"
else:
    print("Build succeeded!")
    
    # Run the test
    print("\nRunning test...")
    test_result = subprocess.run(
        [sys.executable, 'test_import.py'],
        capture_output=True,
        text=True
    )
    
    if test_result.returncode == 0:
        print("Test passed!")
        print(test_result.stdout)
        status = "success"
    else:
        print("Test failed!")
        print("STDOUT:", test_result.stdout)
        print("STDERR:", test_result.stderr)
        status = "failed"

# Write the solution
with open('/workspace/solution.txt', 'w') as f:
    f.write(f"language=cpp\n")
    f.write(f"status={status}\n")

print(f"\nSolution written to /workspace/solution.txt")
print(f"Language: cpp")
print(f"Status: {status}")