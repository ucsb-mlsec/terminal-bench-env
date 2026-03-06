#!/bin/bash

# configure_jni_path.sh - Setup script for JNI native library paths
# This script organizes native libraries and configures java.library.path

set -e

# Directories to search for native libraries
SEARCH_DIRS=("/opt/native-libs" "/usr/local/lib/jni" "/tmp/legacy-libs")

# Target directory for organized libraries
TARGET_DIR="/opt/native-libs/x86_64"

# Create target directory
mkdir -p "$TARGET_DIR"

# Java application source directory
APP_SRC="/opt/app/src"

# Function to detect library architecture
get_library_arch() {
    local lib_file="$1"
    if file "$lib_file" | grep -q "x86-64\|x86_64"; then
        echo "x86_64"
    elif file "$lib_file" | grep -q "aarch64\|ARM aarch64"; then
        echo "aarch64"
    else
        echo "unknown"
    fi
}

# Function to extract required libraries from Java source
get_required_libraries() {
    if [ -d "$APP_SRC" ]; then
        # Search for System.loadLibrary calls in Java files
        grep -rh "System\.loadLibrary\|System\.load" "$APP_SRC" 2>/dev/null | \
            sed -n 's/.*System\.loadLibrary("\([^"]*\)").*/\1/p' | \
            sort -u
    fi
}

echo "Scanning for native libraries..."

# Find all .so files
declare -A FOUND_LIBS

for search_dir in "${SEARCH_DIRS[@]}"; do
    if [ -d "$search_dir" ]; then
        while IFS= read -r lib_path; do
            if [ -f "$lib_path" ] && [ ! -L "$lib_path" ]; then
                arch=$(get_library_arch "$lib_path")
                if [ "$arch" = "x86_64" ]; then
                    lib_name=$(basename "$lib_path")
                    base_name=$(echo "$lib_name" | sed 's/\.so.*//')
                    FOUND_LIBS["$base_name"]="$lib_path"
                fi
            fi
        done < <(find "$search_dir" -name "*.so*" -type f 2>/dev/null)
    fi
done

echo "Extracting required libraries from Java source..."
REQUIRED_LIBS=$(get_required_libraries)

# Copy required x86_64 libraries to target directory
echo "Organizing x86_64 libraries..."

for req_lib in $REQUIRED_LIBS; do
    # Try to find matching library
    for base_name in "${!FOUND_LIBS[@]}"; do
        if [[ "$base_name" == *"$req_lib"* ]] || [[ "$req_lib" == *"$base_name"* ]]; then
            src_path="${FOUND_LIBS[$base_name]}"
            lib_filename=$(basename "$src_path")
            
            # Copy library to target directory
            cp -f "$src_path" "$TARGET_DIR/"
            echo "Copied: $lib_filename"
            
            # Create standard symbolic link without version
            base_link=$(echo "$lib_filename" | sed 's/\.so.*/\.so/')
            if [ "$base_link" != "$lib_filename" ]; then
                ln -sf "$lib_filename" "$TARGET_DIR/$base_link"
            fi
        fi
    done
done

# Also copy all x86_64 libraries found (in case some dependencies were missed)
for base_name in "${!FOUND_LIBS[@]}"; do
    src_path="${FOUND_LIBS[$base_name]}"
    lib_filename=$(basename "$src_path")
    dest_path="$TARGET_DIR/$lib_filename"
    
    if [ ! -f "$dest_path" ]; then
        cp -f "$src_path" "$TARGET_DIR/"
    fi
    
    # Create standard symbolic link without version
    base_link=$(echo "$lib_filename" | sed 's/\.so.*/\.so/')
    if [ "$base_link" != "$lib_filename" ]; then
        ln -sf "$lib_filename" "$TARGET_DIR/$base_link"
    fi
done

# Count actual .so files (not symlinks)
LIBRARY_COUNT=$(find "$TARGET_DIR" -name "*.so*" -type f | wc -l)

# Set up java.library.path
JAVA_LIBRARY_PATH="$TARGET_DIR:/usr/lib/jni:/usr/lib/x86_64-linux-gnu"

# Export for current session
export LD_LIBRARY_PATH="$TARGET_DIR:$LD_LIBRARY_PATH"
export JAVA_LIBRARY_PATH

# Create configuration file
CONFIG_FILE="/solution/library_config.txt"
mkdir -p "$(dirname "$CONFIG_FILE")"

cat > "$CONFIG_FILE" << EOF
LIBRARY_DIR=$TARGET_DIR
JAVA_LIBRARY_PATH=$JAVA_LIBRARY_PATH
LIBRARY_COUNT=$LIBRARY_COUNT
EOF

echo "Configuration written to $CONFIG_FILE"

# Update start-services.sh script
START_SCRIPT="/opt/app/bin/start-services.sh"
if [ -f "$START_SCRIPT" ]; then
    echo "Updating $START_SCRIPT with java.library.path configuration..."
    
    # Backup original
    cp "$START_SCRIPT" "${START_SCRIPT}.backup"
    
    # Add library path configuration before java commands
    sed -i '/^JAVA_HOME=/a\\n# Configure JNI native library path\nexport LD_LIBRARY_PATH='"$TARGET_DIR"':$LD_LIBRARY_PATH\nJAVA_OPTS="-Djava.library.path='"$JAVA_LIBRARY_PATH"'"' "$START_SCRIPT"
    
    # Update java commands to include JAVA_OPTS
    sed -i 's/java -cp/java $JAVA_OPTS -cp/g' "$START_SCRIPT"
    
    chmod +x "$START_SCRIPT"
fi

echo "JNI library path configuration complete!"
echo "Library directory: $TARGET_DIR"
echo "Java library path: $JAVA_LIBRARY_PATH"
echo "Library count: $LIBRARY_COUNT"

# Make this script idempotent - can be sourced or executed
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    echo "Script executed directly. Configuration saved."
else
    echo "Script sourced. Environment variables exported to current shell."
fi