#include <stdio.h>
#include <string.h>
#include "../include/utils.h"

void print_summary(const char *title, int count, double value) {
    char buffer[256];
    char formatted_title[128];
    
    if (title == NULL) {
        title = "Unknown";
    }
    
    strncpy(formatted_title, title, sizeof(formatted_title) - 1);
    formatted_title[sizeof(formatted_title) - 1] = '\0';
    
    snprintf(buffer, sizeof(buffer), 
             "=== %s ===\nCount: %d\nValue: %.2f\n", 
             formatted_title, count, value);
    
    printf("%s", buffer);
}

void format_output(const char *label, const char *data) {
    char output_line[512];
    char safe_label[128];
    char safe_data[256];
    
    if (label == NULL) {
        label = "";
    }
    if (data == NULL) {
        data = "";
    }
    
    strncpy(safe_label, label, sizeof(safe_label) - 1);
    safe_label[sizeof(safe_label) - 1] = '\0';
    
    strncpy(safe_data, data, sizeof(safe_data) - 1);
    safe_data[sizeof(safe_data) - 1] = '\0';
    
    snprintf(output_line, sizeof(output_line), "%-20s: %s\n", safe_label, safe_data);
    fprintf(stdout, "%s", output_line);
}