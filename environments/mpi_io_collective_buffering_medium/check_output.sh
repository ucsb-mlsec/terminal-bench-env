#!/bin/bash

# Check if output file exists
if [ ! -f /tmp/output.dat ]; then
    echo "ERROR: /tmp/output.dat does not exist"
    exit 1
fi

# Check file size is exactly 4096 bytes
FILESIZE=$(stat -c%s /tmp/output.dat 2>/dev/null || stat -f%z /tmp/output.dat 2>/dev/null)
if [ "$FILESIZE" != "4096" ]; then
    echo "ERROR: File size is $FILESIZE bytes, expected 4096 bytes"
    exit 2
fi

echo "File size check: PASSED (4096 bytes)"

# Create a temporary C program to verify data integrity
cat > /tmp/verify_data.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *fp = fopen("/tmp/output.dat", "rb");
    if (!fp) {
        fprintf(stderr, "ERROR: Cannot open /tmp/output.dat\n");
        return 1;
    }
    
    int value;
    int errors = 0;
    
    // Check all 4 processes' data (4 chunks of 256 integers each)
    for (int process = 0; process < 4; process++) {
        // Seek to the correct offset for this process
        fseek(fp, process * 1024, SEEK_SET);
        
        // Read and verify 256 integers
        for (int i = 0; i < 256; i++) {
            if (fread(&value, sizeof(int), 1, fp) != 1) {
                fprintf(stderr, "ERROR: Failed to read at process %d, index %d\n", process, i);
                fclose(fp);
                return 1;
            }
            
            // Each process writes values 0-255
            if (value != i) {
                fprintf(stderr, "ERROR: Process %d, index %d: expected %d, got %d\n", 
                        process, i, i, value);
                errors++;
                if (errors > 10) {
                    fprintf(stderr, "Too many errors, stopping verification\n");
                    fclose(fp);
                    return 1;
                }
            }
        }
    }
    
    fclose(fp);
    
    if (errors > 0) {
        fprintf(stderr, "Data verification FAILED with %d errors\n", errors);
        return 1;
    }
    
    printf("Data integrity check: PASSED\n");
    return 0;
}
EOF

# Compile the verification program
gcc -o /tmp/verify_data /tmp/verify_data.c 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to compile verification program"
    exit 3
fi

# Run the verification program
/tmp/verify_data
VERIFY_RESULT=$?

# Clean up
rm -f /tmp/verify_data /tmp/verify_data.c

if [ $VERIFY_RESULT -eq 0 ]; then
    echo "All verification checks PASSED"
    exit 0
else
    echo "Verification FAILED"
    exit 4
fi