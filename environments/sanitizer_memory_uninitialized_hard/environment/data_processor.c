#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    float value;
    char category[32];
} DataRecord;

typedef struct {
    int total_records;
    float sum_values;
    int error_count;
} Statistics;

typedef struct {
    char operation[64];
    int batch_size;
    float threshold;
} ProcessingContext;

int initialize_data_reader(const char* filename, FILE** file_ptr) {
    int record_count;
    
    *file_ptr = fopen(filename, "r");
    if (*file_ptr == NULL) {
        printf("Error: Could not open file %s\n", filename);
        return -1;
    }
    
    // BUG 1: Using uninitialized variable in calculation
    // record_count is declared but never initialized before use
    printf("Starting data processing with initial count offset: %d\n", record_count);
    record_count = 0;
    
    return record_count;
}

DataRecord* process_record_line(char* line, ProcessingContext* context) {
    DataRecord* record = (DataRecord*)malloc(sizeof(DataRecord));
    
    if (record == NULL) {
        return NULL;
    }
    
    // BUG 2: Reading from malloc'd struct before initialization
    // record->id is accessed before any value is assigned to it
    if (record->id > 0) {
        printf("Processing existing record with ID: %d\n", record->id);
    }
    
    // Now actually initialize the record
    char category[32];
    float value;
    int id;
    
    if (sscanf(line, "%d,%f,%s", &id, &value, category) == 3) {
        record->id = id;
        record->value = value;
        strncpy(record->category, category, 31);
        record->category[31] = '\0';
    } else {
        record->id = 0;
        record->value = 0.0;
        strcpy(record->category, "unknown");
    }
    
    return record;
}

void calculate_statistics(DataRecord* records[], int count, Statistics* stats) {
    float intermediate_results[10];
    
    stats->total_records = count;
    stats->sum_values = 0.0;
    stats->error_count = 0;
    
    // BUG 3: Reading from uninitialized array element
    // intermediate_results[0] is never written to before being read
    float baseline = intermediate_results[0];
    printf("Using baseline value: %.2f\n", baseline);
    
    // Now initialize and use the array properly
    for (int i = 0; i < count && i < 10; i++) {
        intermediate_results[i] = records[i]->value * 1.1;
        stats->sum_values += records[i]->value;
    }
    
    printf("Calculated sum: %.2f\n", stats->sum_values);
}

void print_record(DataRecord* record) {
    if (record != NULL) {
        printf("Record ID: %d, Value: %.2f, Category: %s\n", 
               record->id, record->value, record->category);
    }
}

void process_data_batch(DataRecord* records[], int count, ProcessingContext* context) {
    printf("\nProcessing batch of %d records\n", count);
    printf("Operation: %s\n", context->operation);
    printf("Threshold: %.2f\n", context->threshold);
    
    for (int i = 0; i < count; i++) {
        if (records[i]->value > context->threshold) {
            records[i]->value *= 1.05;
        }
    }
}

int main() {
    FILE* input_file;
    char line[256];
    DataRecord* records[100];
    int record_count = 0;
    
    printf("Data Processing Application v1.0\n");
    printf("=================================\n\n");
    
    ProcessingContext* context = (ProcessingContext*)malloc(sizeof(ProcessingContext));
    if (context == NULL) {
        printf("Failed to allocate processing context\n");
        return 1;
    }
    
    strcpy(context->operation, "standard_processing");
    context->batch_size = 10;
    context->threshold = 50.0;
    
    int init_result = initialize_data_reader("sample_data.txt", &input_file);
    if (init_result < 0) {
        free(context);
        return 1;
    }
    
    printf("\nReading data from file...\n");
    
    while (fgets(line, sizeof(line), input_file) != NULL && record_count < 100) {
        if (strlen(line) > 1) {
            DataRecord* record = process_record_line(line, context);
            if (record != NULL) {
                records[record_count] = record;
                print_record(record);
                record_count++;
            }
        }
    }
    
    fclose(input_file);
    
    printf("\nTotal records read: %d\n", record_count);
    
    if (record_count > 0) {
        process_data_batch(records, record_count, context);
        
        Statistics stats;
        calculate_statistics(records, record_count, &stats);
        
        printf("\n=================================\n");
        printf("Processing Statistics:\n");
        printf("Total Records: %d\n", stats.total_records);
        printf("Sum of Values: %.2f\n", stats.sum_values);
        printf("Average Value: %.2f\n", stats.sum_values / stats.total_records);
        printf("Error Count: %d\n", stats.error_count);
    }
    
    for (int i = 0; i < record_count; i++) {
        free(records[i]);
    }
    free(context);
    
    printf("\nProcessing complete.\n");
    
    return 0;
}