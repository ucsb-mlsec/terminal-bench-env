#!/bin/bash

# First, let's try to run the analyzer to see the exact error
echo "Testing analyzer before fix..."
/usr/local/bin/analyzer /data/sample.dat 2>&1 | head -20

# Check if the library exists
echo -e "\nChecking current library..."
ls -l /usr/local/lib/libmathcore.so.1

# Check library dependencies
echo -e "\nChecking analyzer dependencies..."
ldd /usr/local/bin/analyzer | grep libmathcore

# Navigate to source directory
cd /opt/mathcore-src/

# Check what's available
echo -e "\nChecking source files..."
ls -la

# Rebuild the library from source
echo -e "\nRebuilding library from source..."
make clean
make

# Install the newly built library
echo -e "\nInstalling updated library..."
make install

# Verify installation
echo -e "\nVerifying library installation..."
ls -l /usr/local/lib/libmathcore.so.1

# Update library cache
ldconfig

# Test the analyzer again
echo -e "\nTesting analyzer after rebuild..."
if /usr/local/bin/analyzer /data/sample.dat; then
    echo -e "\nSUCCESS: Analyzer ran successfully"
    cat > /home/user/solution.txt << 'EOF'
issue=shared library incompatible with rebuilt application binary
action=recompiled libmathcore.so.1 from source and reinstalled
status=PASS
EOF
else
    echo -e "\nFAILURE: Analyzer still failing"
    cat > /home/user/solution.txt << 'EOF'
issue=shared library incompatible with rebuilt application binary
action=recompiled libmathcore.so.1 from source and reinstalled
status=FAIL
EOF
fi

# Display solution
echo -e "\nSolution saved to /home/user/solution.txt:"
cat /home/user/solution.txt