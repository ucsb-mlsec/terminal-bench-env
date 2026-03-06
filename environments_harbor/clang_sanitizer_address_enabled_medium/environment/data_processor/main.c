#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BUFFER_SIZE 10
#define MAX_RECORDS 5

typedef struct {
    int id;
    char *name;
    double value;
} Record;

// Global pointer for demonstration
Record *global_record = NULL;

// Function to allocate a buffer with potential issues
char* allocate_buffer(int size) {
    char *buffer = (char*)malloc(size);
    if (buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return buffer;
}

// Function that causes heap buffer overflow
void process_data(char *input) {
    printf("Processing data: %s\n", input);
    
    // Issue 1: Heap buffer overflow
    char *local_buffer = allocate_buffer(BUFFER_SIZE);
    
    // This will overflow if input is longer than BUFFER_SIZE
    strcpy(local_buffer, input);
    
    printf("Processed: %s\n", local_buffer);
    free(local_buffer);
}

// Function that demonstrates use-after-free
void use_after_free_demo() {
    printf("\n--- Use After Free Demo ---\n");
    
    Record *temp_record = (Record*)malloc(sizeof(Record));
    temp_record->id = 100;
    temp_record->name = allocate_buffer(20);
    strcpy(temp_record->name, "TemporaryRecord");
    temp_record->value = 99.99;
    
    printf("Record ID: %d, Name: %s\n", temp_record->id, temp_record->name);
    
    // Free the record
    free(temp_record->name);
    free(temp_record);
    
    // Issue 2: Use after free - accessing freed memory
    printf("Accessing freed memory - ID: %d\n", temp_record->id);
    printf("Value: %.2f\n", temp_record->value);
}

// Function with memory leak
Record* create_record(int id, const char *name, double value) {
    Record *rec = (Record*)malloc(sizeof(Record));
    rec->id = id;
    rec->name = allocate_buffer(strlen(name) + 1);
    strcpy(rec->name, name);
    rec->value = value;
    
    return rec;
}

// Function that leaks memory
void memory_leak_demo() {
    printf("\n--- Memory Leak Demo ---\n");
    
    // Issue 3: Memory leak - allocate but never free
    for (int i = 0; i < 3; i++) {
        char *leaked_buffer = allocate_buffer(100);
        sprintf(leaked_buffer, "Leaked buffer number %d", i);
        printf("Created: %s\n", leaked_buffer);
        // Intentionally not freeing leaked_buffer
    }
}

// Function to process multiple records
void process_records() {
    printf("\n--- Processing Records ---\n");
    
    Record *records[MAX_RECORDS];
    
    for (int i = 0; i < MAX_RECORDS; i++) {
        char name[50];
        sprintf(name, "Record_%d", i);
        records[i] = create_record(i + 1, name, (i + 1) * 10.5);
        printf("Created record: ID=%d, Name=%s, Value=%.2f\n", 
               records[i]->id, records[i]->name, records[i]->value);
    }
    
    // Free some but not all records (partial memory leak)
    for (int i = 0; i < 3; i++) {
        free(records[i]->name);
        free(records[i]);
    }
    // records[3] and records[4] are leaked
}

// Another buffer overflow scenario
void copy_data(const char *source) {
    printf("\n--- Copy Data Demo ---\n");
    
    char *dest = allocate_buffer(8);
    
    // Issue 4: Another buffer overflow
    strcpy(dest, source);
    printf("Copied data: %s\n", dest);
    
    free(dest);
}

int main() {
    printf("=== Data Processor Starting ===\n\n");
    
    // Test 1: Buffer overflow with long string
    char long_string[] = "This is a very long string that will overflow the buffer";
    process_data(long_string);
    
    // Test 2: Use after free
    use_after_free_demo();
    
    // Test 3: Memory leaks
    memory_leak_demo();
    
    // Test 4: Process records with partial cleanup
    process_records();
    
    // Test 5: Another overflow scenario
    copy_data("AnotherLongStringHere");
    
    printf("\n=== Data Processor Finished ===\n");
    
    return 0;
}