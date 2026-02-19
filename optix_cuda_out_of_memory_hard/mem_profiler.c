#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ALLOCATIONS 10000
#define LINE_BUFFER_SIZE 256

typedef struct {
    int id;
    long size;
    int active;
} Allocation;

// Global allocation tracking array - BUG #2: This is allocated but never freed
Allocation *allocation_table = NULL;
int table_size = 0;

// BUG #1: Using int instead of long for memory tracking - will overflow with large values
int current_memory = 0;
long peak_memory = 0;
int total_allocs = 0;
int total_frees = 0;

void init_allocation_table() {
    // BUG #2: Memory leak - allocating memory that will never be freed
    allocation_table = (Allocation*)malloc(MAX_ALLOCATIONS * sizeof(Allocation));
    if (allocation_table == NULL) {
        fprintf(stderr, "Failed to allocate memory for allocation table\n");
        exit(1);
    }
    
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        allocation_table[i].id = -1;
        allocation_table[i].size = 0;
        allocation_table[i].active = 0;
    }
    table_size = 0;
}

int find_allocation(int id) {
    for (int i = 0; i < table_size; i++) {
        if (allocation_table[i].id == id && allocation_table[i].active) {
            return i;
        }
    }
    return -1;
}

void process_alloc(int id, long size) {
    total_allocs++;
    
    // BUG #3: No bounds checking - could overflow the allocation_table array
    // if table_size reaches MAX_ALLOCATIONS
    int index = table_size;
    allocation_table[index].id = id;
    allocation_table[index].size = size;
    allocation_table[index].active = 1;
    table_size++;
    
    // BUG #1: current_memory is int, will overflow with large allocations
    current_memory += (int)size;
    
    if (current_memory > peak_memory) {
        peak_memory = current_memory;
    }
}

void process_free(int id) {
    total_frees++;
    
    int index = find_allocation(id);
    if (index >= 0) {
        // BUG #1: Subtracting from int current_memory
        current_memory -= (int)allocation_table[index].size;
        allocation_table[index].active = 0;
    }
}

void parse_line(char *line) {
    char operation[16];
    int id;
    long size;
    
    // Try to parse as ALLOC operation
    if (sscanf(line, "%s %d %ld", operation, &id, &size) == 3) {
        if (strcmp(operation, "ALLOC") == 0) {
            process_alloc(id, size);
            return;
        }
    }
    
    // Try to parse as FREE operation
    if (sscanf(line, "%s %d", operation, &id) == 2) {
        if (strcmp(operation, "FREE") == 0) {
            process_free(id);
            return;
        }
    }
}

void write_stats(const char *output_file) {
    FILE *fp = fopen(output_file, "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open output file: %s\n", output_file);
        exit(1);
    }
    
    // Convert peak memory from bytes to megabytes (1 MB = 1048576 bytes)
    long peak_mb = peak_memory / 1048576;
    
    fprintf(fp, "peak_mb=%ld\n", peak_mb);
    fprintf(fp, "total_allocs=%d\n", total_allocs);
    fprintf(fp, "total_frees=%d\n", total_frees);
    
    fclose(fp);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <process_log_file>\n", argv[0]);
        return 1;
    }
    
    const char *log_file = argv[1];
    
    // Initialize the allocation tracking table
    init_allocation_table();
    
    // Open the log file
    FILE *fp = fopen(log_file, "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open log file: %s\n", log_file);
        return 1;
    }
    
    // BUG #2: Another memory leak - this buffer is allocated but never freed
    char *line_buffer = (char*)malloc(LINE_BUFFER_SIZE * sizeof(char));
    if (line_buffer == NULL) {
        fprintf(stderr, "Failed to allocate line buffer\n");
        fclose(fp);
        return 1;
    }
    
    // Process each line in the log file
    while (fgets(line_buffer, LINE_BUFFER_SIZE, fp) != NULL) {
        // Remove newline if present
        size_t len = strlen(line_buffer);
        if (len > 0 && line_buffer[len - 1] == '\n') {
            line_buffer[len - 1] = '\0';
        }
        
        // Skip empty lines
        if (strlen(line_buffer) == 0) {
            continue;
        }
        
        parse_line(line_buffer);
    }
    
    fclose(fp);
    
    // Write the statistics to output file
    write_stats("memory_stats.txt");
    
    // BUG #2: Memory leak - we never free allocation_table or line_buffer
    // free(allocation_table);
    // free(line_buffer);
    
    return 0;
}