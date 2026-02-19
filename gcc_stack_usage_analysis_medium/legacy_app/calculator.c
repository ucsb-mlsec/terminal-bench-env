#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_READINGS 100
#define BUFFER_SIZE 512
#define TEMP_ARRAY_SIZE 200

// Simple calculation function with minimal stack usage
int calculate_sum(int a, int b) {
    int result;
    result = a + b;
    return result;
}

// Function with small stack usage
int validate_input(int value) {
    int temp1, temp2, temp3;
    int range_min = 0;
    int range_max = 1000;
    
    temp1 = value - range_min;
    temp2 = range_max - value;
    temp3 = temp1 + temp2;
    
    if (value >= range_min && value <= range_max) {
        return temp3 > 0 ? 1 : 0;
    }
    return 0;
}

// Function with moderate stack usage (around 160 bytes)
int process_sensor_data(int sensor_id) {
    int readings[40];
    int i;
    int sum = 0;
    int average;
    
    for (i = 0; i < 40; i++) {
        readings[i] = sensor_id * i + (i % 10);
        sum += readings[i];
    }
    
    average = sum / 40;
    return average;
}

// Function with moderate-high stack usage (around 220 bytes)
void convert_format(int input_value) {
    char buffer[50];
    int temp_values[40];
    int i;
    
    sprintf(buffer, "Converting value: %d", input_value);
    
    for (i = 0; i < 40; i++) {
        temp_values[i] = input_value * (i + 1);
    }
    
    printf("%s\n", buffer);
}

// Function exceeding 256 bytes - large character buffer
void generate_report(int report_id) {
    char report_buffer[BUFFER_SIZE];
    char header[100];
    char footer[100];
    int i;
    
    sprintf(header, "=== Report ID: %d ===\n", report_id);
    sprintf(footer, "=== End Report ===\n");
    
    strcpy(report_buffer, header);
    
    for (i = 0; i < 10; i++) {
        char line[50];
        sprintf(line, "Entry %d: Value %d\n", i, i * report_id);
        strcat(report_buffer, line);
    }
    
    strcat(report_buffer, footer);
    printf("%s", report_buffer);
}

// Function exceeding 256 bytes - large integer array
int analyze_data_stream(int stream_id) {
    int data_buffer[TEMP_ARRAY_SIZE];
    int i;
    int max_value = 0;
    int min_value = 9999;
    int sum = 0;
    
    for (i = 0; i < TEMP_ARRAY_SIZE; i++) {
        data_buffer[i] = (stream_id * i) % 1000;
        
        if (data_buffer[i] > max_value) {
            max_value = data_buffer[i];
        }
        if (data_buffer[i] < min_value) {
            min_value = data_buffer[i];
        }
        
        sum += data_buffer[i];
    }
    
    return sum / TEMP_ARRAY_SIZE;
}

// Function with highest stack usage - multiple large arrays
void process_large_dataset(int dataset_id) {
    int primary_data[250];
    int secondary_data[150];
    char metadata[200];
    int results[50];
    int i, j;
    int temp_sum;
    
    sprintf(metadata, "Processing dataset: %d", dataset_id);
    
    for (i = 0; i < 250; i++) {
        primary_data[i] = dataset_id * i;
    }
    
    for (i = 0; i < 150; i++) {
        secondary_data[i] = dataset_id + i * 2;
    }
    
    for (i = 0; i < 50; i++) {
        temp_sum = 0;
        for (j = 0; j < 5; j++) {
            if ((i * 5 + j) < 250) {
                temp_sum += primary_data[i * 5 + j];
            }
        }
        results[i] = temp_sum / 5;
    }
    
    printf("%s - Processed %d elements\n", metadata, 250);
}

// Function exceeding 256 bytes - mixed large buffers
int calculate_statistics(int sample_count) {
    double samples[120];
    int histogram[50];
    char status_msg[150];
    int i;
    double mean = 0.0;
    double variance = 0.0;
    
    for (i = 0; i < 120; i++) {
        samples[i] = (double)(sample_count * i) / 100.0;
        mean += samples[i];
    }
    mean = mean / 120.0;
    
    for (i = 0; i < 120; i++) {
        double diff = samples[i] - mean;
        variance += diff * diff;
    }
    variance = variance / 120.0;
    
    for (i = 0; i < 50; i++) {
        histogram[i] = 0;
    }
    
    sprintf(status_msg, "Statistics calculated for %d samples", sample_count);
    printf("%s\n", status_msg);
    
    return (int)mean;
}

// Small helper function
void print_header(void) {
    printf("Legacy Calculator System v1.0\n");
    printf("=============================\n");
}

// Another small function
int multiply_values(int x, int y) {
    int result = x * y;
    return result;
}

// Function with moderate stack usage
void format_output(int value1, int value2) {
    char output_line[80];
    int intermediate[20];
    int i;
    
    for (i = 0; i < 20; i++) {
        intermediate[i] = value1 + value2 + i;
    }
    
    sprintf(output_line, "Formatted: %d, %d", value1, value2);
    printf("%s\n", output_line);
}

int main(int argc, char *argv[]) {
    int test_value = 42;
    int result;
    
    print_header();
    
    result = calculate_sum(10, 20);
    printf("Sum: %d\n", result);
    
    if (validate_input(test_value)) {
        printf("Input validated\n");
    }
    
    result = process_sensor_data(5);
    printf("Sensor average: %d\n", result);
    
    convert_format(test_value);
    
    generate_report(1);
    
    result = analyze_data_stream(3);
    printf("Stream analysis result: %d\n", result);
    
    process_large_dataset(2);
    
    result = calculate_statistics(100);
    printf("Statistics result: %d\n", result);
    
    result = multiply_values(7, 8);
    printf("Product: %d\n", result);
    
    format_output(15, 25);
    
    printf("Processing complete\n");
    
    return 0;
}