#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "data_processor.h"

#define MAX_RECORD_SIZE 1024
#define CACHE_SIZE 256

typedef struct {
    char* data;
    int size;
    int id;
} Record;

typedef struct {
    Record** entries;
    int count;
    int capacity;
} Cache;

// Global cache pointer - lives for program lifetime
static Cache* global_cache = NULL;

/**
 * Initialize the global cache system
 * This cache is designed to persist for the entire program execution
 */
void init_cache() {
    if (global_cache != NULL) {
        return; // Already initialized
    }
    
    // Global cache, lives for program lifetime
    global_cache = (Cache*)malloc(sizeof(Cache));
    if (global_cache == NULL) {
        fprintf(stderr, "Failed to allocate cache\n");
        return;
    }
    
    global_cache->entries = (Record**)malloc(sizeof(Record*) * CACHE_SIZE);
    global_cache->count = 0;
    global_cache->capacity = CACHE_SIZE;
    
    printf("Cache initialized with capacity %d\n", CACHE_SIZE);
}

/**
 * Process a record from input data
 * Creates a new record structure and processes it
 * Caller should free this
 */
Record* process_record(const char* input) {
    if (input == NULL) {
        return NULL;
    }
    
    Record* rec = (Record*)malloc(sizeof(Record));
    if (rec == NULL) {
        fprintf(stderr, "Failed to allocate record\n");
        return NULL;
    }
    
    int len = strlen(input);
    rec->data = (char*)malloc(len + 1);
    if (rec->data == NULL) {
        free(rec);
        return NULL;
    }
    
    strcpy(rec->data, input);
    rec->size = len;
    rec->id = rand() % 10000;
    
    printf("Processed record %d with %d bytes\n", rec->id, rec->size);
    
    // Return without freeing - caller's responsibility
    return rec;
}

/**
 * Parse input string and extract fields
 * Returns number of fields parsed
 */
int parse_input(const char* input, char*** fields) {
    if (input == NULL || fields == NULL) {
        return -1;
    }
    
    int field_count = 0;
    char* buffer = (char*)malloc(strlen(input) + 1);
    if (buffer == NULL) {
        return -1;
    }
    
    strcpy(buffer, input);
    
    // Count delimiters to estimate field count
    for (int i = 0; buffer[i] != '\0'; i++) {
        if (buffer[i] == ',') {
            field_count++;
        }
    }
    
    // Error condition - invalid input format
    if (field_count == 0) {
        fprintf(stderr, "No fields found in input\n");
        return -1; // Early return without freeing buffer - LEAK!
    }
    
    field_count++; // Add one for last field
    
    *fields = (char**)malloc(sizeof(char*) * field_count);
    if (*fields == NULL) {
        free(buffer);
        return -1;
    }
    
    // Parse fields
    int idx = 0;
    char* token = strtok(buffer, ",");
    while (token != NULL && idx < field_count) {
        (*fields)[idx] = (char*)malloc(strlen(token) + 1);
        strcpy((*fields)[idx], token);
        idx++;
        token = strtok(NULL, ",");
    }
    
    free(buffer);
    return field_count;
}

/**
 * Get a record from the cache by ID
 */
Record* get_cached_record(int id) {
    if (global_cache == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < global_cache->count; i++) {
        if (global_cache->entries[i]->id == id) {
            return global_cache->entries[i];
        }
    }
    
    return NULL;
}

/**
 * Add a record to the cache
 */
int add_to_cache(Record* rec) {
    if (global_cache == NULL || rec == NULL) {
        return -1;
    }
    
    if (global_cache->count >= global_cache->capacity) {
        fprintf(stderr, "Cache is full\n");
        return -1;
    }
    
    global_cache->entries[global_cache->count++] = rec;
    return 0;
}

/**
 * Print cache statistics
 */
void print_cache_stats() {
    if (global_cache == NULL) {
        printf("Cache not initialized\n");
        return;
    }
    
    printf("Cache contains %d records (capacity: %d)\n", 
           global_cache->count, global_cache->capacity);
}