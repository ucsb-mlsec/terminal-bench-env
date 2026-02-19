#!/bin/bash

# Create the test environment
mkdir -p /opt/symbolic-test

# Create the validation script
cat > /opt/symbolic-test/run_analysis.sh << 'EOF'
#!/bin/bash

CONFIG_FILE="/opt/symbolic-test/backend.conf"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found at $CONFIG_FILE"
    exit 1
fi

# Extract SOLVER_LIB value from backend.conf
SOLVER_LIB=$(grep "^SOLVER_LIB=" "$CONFIG_FILE" | cut -d'=' -f2)

if [ -z "$SOLVER_LIB" ]; then
    echo "Error: SOLVER_LIB not configured in $CONFIG_FILE"
    exit 1
fi

# Check if the library file exists
if [ -f "$SOLVER_LIB" ]; then
    echo "Solver backend configured correctly"
    exit 0
else
    echo "Error: Solver library not found at $SOLVER_LIB"
    exit 1
fi
EOF

chmod +x /opt/symbolic-test/run_analysis.sh

# Create the initial (incorrect) backend.conf
cat > /opt/symbolic-test/backend.conf << 'EOF'
SOLVER_LIB=/usr/lib/libz3solver.so
EOF

# Find the actual Z3 library on the system
Z3_LIB=$(find /usr/lib /usr/local/lib /lib -name "*z3*.so*" 2>/dev/null | grep -E '\.so(\.[0-9]+)*$' | head -n 1)

if [ -z "$Z3_LIB" ]; then
    # If not found in standard locations, try ldconfig
    Z3_LIB=$(ldconfig -p 2>/dev/null | grep "libz3" | awk '{print $NF}' | head -n 1)
fi

if [ -z "$Z3_LIB" ]; then
    # Last resort: check common alternative paths
    for path in /usr/lib/x86_64-linux-gnu/libz3.so /usr/lib64/libz3.so /usr/lib/libz3.so; do
        if [ -f "$path" ]; then
            Z3_LIB="$path"
            break
        fi
    done
fi

# Save the solution
if [ -n "$Z3_LIB" ]; then
    echo "$Z3_LIB" > /home/user/solution.txt
    echo "Solution saved to /home/user/solution.txt: $Z3_LIB"
else
    echo "Error: Could not find Z3 library on the system"
    exit 1
fi