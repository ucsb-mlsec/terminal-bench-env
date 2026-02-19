#!/usr/bin/env python3

import os
import subprocess
import sys

def find_file(filename, search_paths=['/opt', '/usr', '/home', '/var', '/data']):
    """Find a file in the filesystem."""
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        for root, dirs, files in os.walk(search_path):
            if filename in files:
                return os.path.join(root, filename)
    return None

def find_directory(dirname, search_paths=['/opt', '/usr', '/home', '/var', '/data']):
    """Find a directory in the filesystem."""
    results = []
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        for root, dirs, files in os.walk(search_path):
            if dirname in dirs:
                results.append(os.path.join(root, dirname))
    return results

def find_python_package(package_name, search_paths=['/opt', '/usr', '/home', '/var', '/data']):
    """Find a Python package by looking for its __init__.py."""
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        for root, dirs, files in os.walk(search_path):
            if package_name in dirs:
                init_file = os.path.join(root, package_name, '__init__.py')
                if os.path.exists(init_file):
                    return os.path.join(root, package_name)
    return None

def find_application_entry():
    """Find the analysis application entry point."""
    search_paths = ['/opt', '/usr', '/home', '/var', '/data']
    
    # Look for common application entry point names
    entry_names = ['analysis.py', 'run_analysis.py', 'main.py', 'app.py']
    
    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue
        for root, dirs, files in os.walk(search_path):
            for entry_name in entry_names:
                if entry_name in files:
                    full_path = os.path.join(root, entry_name)
                    # Check if it's likely the analysis application
                    with open(full_path, 'r') as f:
                        content = f.read()
                        if 'import' in content or 'from' in content:
                            return full_path
    return None

# Step 1: Find the core library (scicore)
print("Searching for core library (scicore)...")
core_lib_path = find_python_package('scicore')
if core_lib_path:
    print(f"Found scicore at: {core_lib_path}")
    core_lib_parent = os.path.dirname(core_lib_path)
else:
    print("ERROR: Could not find scicore package")
    sys.exit(1)

# Step 2: Find the data processing framework
print("\nSearching for data processing framework...")
framework_names = ['dataproc', 'framework', 'processing', 'dataframework']
framework_path = None
framework_parent = None

for name in framework_names:
    path = find_python_package(name)
    if path:
        framework_path = path
        framework_parent = os.path.dirname(path)
        print(f"Found framework '{name}' at: {framework_path}")
        break

if not framework_path:
    print("ERROR: Could not find framework package")
    sys.exit(1)

# Step 3: Find the analysis application
print("\nSearching for analysis application...")
app_entry = find_application_entry()
if app_entry:
    print(f"Found application entry at: {app_entry}")
    app_path = os.path.dirname(app_entry)
else:
    print("ERROR: Could not find application entry point")
    sys.exit(1)

# Step 4: Construct PYTHONPATH
print("\nConstructing PYTHONPATH...")
pythonpath_parts = []

# Add parent directories of packages to PYTHONPATH
if core_lib_parent not in pythonpath_parts:
    pythonpath_parts.append(core_lib_parent)

if framework_parent not in pythonpath_parts and framework_parent != core_lib_parent:
    pythonpath_parts.append(framework_parent)

if app_path not in pythonpath_parts:
    pythonpath_parts.append(app_path)

pythonpath_value = ':'.join(pythonpath_parts)
print(f"PYTHONPATH: {pythonpath_value}")

# Step 5: Test the application
print("\nTesting application execution...")
env = os.environ.copy()
env['PYTHONPATH'] = pythonpath_value

try:
    result = subprocess.run(
        ['python3', app_entry],
        env=env,
        capture_output=True,
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        print("Application executed successfully!")
        output_lines = result.stdout.strip().split('\n')
        verification_output = output_lines[0] if output_lines else ""
        print(f"First line of output: {verification_output}")
    else:
        print(f"Application failed with return code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        verification_output = "ERROR"
        
except subprocess.TimeoutExpired:
    print("Application execution timed out")
    verification_output = "TIMEOUT"
except Exception as e:
    print(f"Error running application: {e}")
    verification_output = "ERROR"

# Step 6: Save the solution
print("\nSaving solution...")
os.makedirs('/solution', exist_ok=True)

solution_content = f"""CORE_LIBRARY_PATH={core_lib_path}
FRAMEWORK_PATH={framework_path}
APPLICATION_PATH={app_path}
PYTHONPATH_VALUE={pythonpath_value}
VERIFICATION_OUTPUT={verification_output}
"""

with open('/solution/python_path_fix.txt', 'w') as f:
    f.write(solution_content)

print("Solution saved to /solution/python_path_fix.txt")
print("\n" + "="*60)
print("SOLUTION SUMMARY:")
print("="*60)
print(solution_content)