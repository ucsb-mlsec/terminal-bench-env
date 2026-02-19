#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define FIELD_SIZE 50

typedef struct {
    char name[FIELD_SIZE];
    char email[FIELD_SIZE];
    char address[FIELD_SIZE];
    char phone[FIELD_SIZE];
    int age;
} UserRecord;

void parse_csv_line(char *line, UserRecord *record) {
    char *token;
    int field_count = 0;
    
    // Remove newline if present
    line[strcspn(line, "\n")] = 0;
    
    // Parse name field
    token = strtok(line, ",");
    if (token != NULL) {
        strcpy(record->name, token);  // VULNERABILITY: No bounds checking
        field_count++;
    }
    
    // Parse email field
    token = strtok(NULL, ",");
    if (token != NULL) {
        strcpy(record->email, token);  // VULNERABILITY: No bounds checking
        field_count++;
    }
    
    // Parse address field
    token = strtok(NULL, ",");
    if (token != NULL) {
        strcpy(record->address, token);  // VULNERABILITY: No bounds checking
        field_count++;
    }
    
    // Parse phone field
    token = strtok(NULL, ",");
    if (token != NULL) {
        strcpy(record->phone, token);  // VULNERABILITY: No bounds checking
        field_count++;
    }
    
    // Parse age field
    token = strtok(NULL, ",");
    if (token != NULL) {
        record->age = atoi(token);
        field_count++;
    }
}

void print_record(UserRecord *record) {
    printf("Name: %s, Email: %s, Address: %s, Phone: %s, Age: %d\n",
           record->name, record->email, record->address, 
           record->phone, record->age);
}

int process_csv_file(const char *filename) {
    FILE *input_file;
    FILE *output_file;
    char line[MAX_LINE_LENGTH];
    UserRecord record;
    int records_processed = 0;
    int is_header = 1;
    
    // Open input CSV file
    input_file = fopen(filename, "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Cannot open input file '%s'\n", filename);
        return -1;
    }
    
    printf("Processing CSV file: %s\n", filename);
    
    // Process each line
    while (fgets(line, sizeof(line), input_file) != NULL) {
        // Skip header line
        if (is_header) {
            is_header = 0;
            continue;
        }
        
        // Skip empty lines
        if (strlen(line) <= 1) {
            continue;
        }
        
        // Initialize record
        memset(&record, 0, sizeof(UserRecord));
        
        // Parse the CSV line
        parse_csv_line(line, &record);
        
        // Print record for debugging
        print_record(&record);
        
        records_processed++;
    }
    
    fclose(input_file);
    
    // Write results to output file
    output_file = fopen("/tmp/processing_result.txt", "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error: Cannot create output file\n");
        return -1;
    }
    
    fprintf(output_file, "records_processed=%d\n", records_processed);
    fprintf(output_file, "status=success\n");
    fclose(output_file);
    
    printf("\nProcessing complete. Total records processed: %d\n", records_processed);
    
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <csv_filename>\n", argv[0]);
        return 1;
    }
    
    const char *filename = argv[1];
    
    int result = process_csv_file(filename);
    
    if (result != 0) {
        fprintf(stderr, "Error: Processing failed\n");
        return 1;
    }
    
    return 0;
}