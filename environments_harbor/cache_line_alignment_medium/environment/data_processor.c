// Legacy data processor - unoptimized structure layout

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// DataRecord structure - basic definition without alignment optimization
struct DataRecord {
    uint32_t id;
    char status;
    uint64_t timestamp;
    double value;
    uint16_t flags;
    char category;
    uint32_t count;
    uint64_t hash;
};

int main() {
    printf("Data Processor Started\n");
    
    // Declare a DataRecord variable to test
    struct DataRecord record;
    
    // Initialize fields to avoid warnings
    record.id = 0;
    record.status = 'A';
    record.timestamp = 0;
    record.value = 0.0;
    record.flags = 0;
    record.category = 'B';
    record.count = 0;
    record.hash = 0;
    
    // Print the size of the structure
    printf("Size of DataRecord: %zu bytes\n", sizeof(struct DataRecord));
    
    return 0;
}