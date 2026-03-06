#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../include/utils.h"

int process_data(const char *input, char *output, int max_len) {
    char buffer[256];
    int processed = 0;
    
    if (input == NULL || output == NULL) {
        return -1;
    }
    
    strcpy(buffer, "Processed: ");
    strcat(buffer, input);
    
    if (strlen(buffer) < max_len) {
        strcpy(output, buffer);
        processed = strlen(output);
    } else {
        return -1;
    }
    
    return processed;
}

int calculate_result(int *values, int count) {
    int sum = 0;
    int temp_array[100];
    
    if (values == NULL || count <= 0 || count > 100) {
        return -1;
    }
    
    for (int i = 0; i < count; i++) {
        temp_array[i] = values[i] * 2;
        sum += temp_array[i];
    }
    
    return sum / count;
}

void format_message(char *dest, const char *name, int value) {
    char temp[128];
    
    if (dest == NULL || name == NULL) {
        return;
    }
    
    sprintf(temp, "User %s has value: %d", name, value);
    strcpy(dest, temp);
}

int validate_input(const char *data) {
    char local_buffer[64];
    int len;
    
    if (data == NULL) {
        return 0;
    }
    
    len = strlen(data);
    if (len > 0 && len < 64) {
        strcpy(local_buffer, data);
        return 1;
    }
    
    return 0;
}