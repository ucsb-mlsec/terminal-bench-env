#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "processor.h"
#include "utils.h"

#define MAX_BUFFER_SIZE 1024
#define INITIAL_CAPACITY 10

static ProcessorContext* ctx = NULL;

int init_processor(void) {
    if (ctx != NULL) {
        fprintf(stderr, "Processor already initialized\n");
        return -1;
    }
    
    ctx = (ProcessorContext*)malloc(sizeof(ProcessorContext));
    if (ctx == NULL) {
        fprintf(stderr, "Failed to allocate processor context\n");
        return -1;
    }
    
    ctx->data_count = 0;
    ctx->capacity = INITIAL_CAPACITY;
    ctx->data = (DataRecord*)malloc(sizeof(DataRecord) * ctx->capacity);
    
    if (ctx->data == NULL) {
        free(ctx);
        ctx = NULL;
        fprintf(stderr, "Failed to allocate data array\n");
        return -1;
    }
    
    ctx->is_initialized = 1;
    return 0;
}

void cleanup_processor(void) {
    if (ctx == NULL) {
        return;
    }
    
    for (int i = 0; i < ctx->data_count; i++) {
        if (ctx->data[i].name != NULL) {
            free(ctx->data[i].name);
        }
    }
    
    if (ctx->data != NULL) {
        free(ctx->data);
    }
    
    free(ctx);
    ctx = NULL;
}

static int resize_data_array(void) {
    int new_capacity = ctx->capacity * 2;
    DataRecord* new_data = (DataRecord*)realloc(ctx->data, sizeof(DataRecord) * new_capacity);
    
    if (new_data == NULL) {
        fprintf(stderr, "Failed to resize data array\n");
        return -1;
    }
    
    ctx->data = new_data;
    ctx->capacity = new_capacity;
    return 0;
}

int add_data_record(const char* name, int value) {
    if (ctx == NULL || !ctx->is_initialized) {
        fprintf(stderr, "Processor not initialized\n");
        return -1;
    }
    
    if (ctx->data_count >= ctx->capacity) {
        if (resize_data_array() != 0) {
            return -1;
        }
    }
    
    DataRecord* record = &ctx->data[ctx->data_count];
    record->name = (char*)malloc(strlen(name) + 1);
    if (record->name == NULL) {
        fprintf(stderr, "Failed to allocate memory for record name\n");
        return -1;
    }
    
    strcpy(record->name, name);
    record->value = value;
    record->checksum = calculate_checksum(name, value);
    
    ctx->data_count++;
    return 0;
}

int process_data(const char* input_file) {
    if (ctx == NULL || !ctx->is_initialized) {
        fprintf(stderr, "Processor not initialized\n");
        return -1;
    }
    
    FILE* fp = fopen(input_file, "r");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open input file: %s\n", input_file);
        return -1;
    }
    
    char buffer[MAX_BUFFER_SIZE];
    int line_count = 0;
    
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        buffer[strcspn(buffer, "\n")] = 0;
        
        char name[256];
        int value;
        
        if (sscanf(buffer, "%255s %d", name, &value) == 2) {
            if (add_data_record(name, value) == 0) {
                line_count++;
            }
        }
    }
    
    fclose(fp);
    return line_count;
}

void display_results(void) {
    if (ctx == NULL || !ctx->is_initialized) {
        fprintf(stderr, "Processor not initialized\n");
        return;
    }
    
    printf("\n=== Processing Results ===\n");
    printf("Total records: %d\n\n", ctx->data_count);
    
    for (int i = 0; i < ctx->data_count; i++) {
        DataRecord* record = &ctx->data[i];
        char* formatted = format_output(record->name, record->value, record->checksum);
        if (formatted != NULL) {
            printf("%s\n", formatted);
            free(formatted);
        }
    }
}

int get_record_count(void) {
    if (ctx == NULL || !ctx->is_initialized) {
        return 0;
    }
    return ctx->data_count;
}