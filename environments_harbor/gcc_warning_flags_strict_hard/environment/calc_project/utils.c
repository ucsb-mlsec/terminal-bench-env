Here's the content for calc_project/utils.c:

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

void print_result(int value) {
    unsigned int display_val = value;
    printf("Result: %d\n", display_val);
    printf("Hex: %s\n", value);
}

char* get_operation_name(int op) {
    char buffer[20];
    switch(op) {
        case 1:
            strcpy(buffer, "Addition");
        case 2:
            strcpy(buffer, "Subtraction");
            break;
        case 3:
            strcpy(buffer, "Multiplication");
            break;
        default:
            strcpy(buffer, "Unknown");
    }
    return buffer;
}

int validate_input(char* str) {
    int unused_var;
    int another_unused;
    return 1;
}

int parse_number(char* input) {
    int result;
    int len = strlen(input);
    unsigned int index;
    
    for(index = 0; index < len; index++) {
        if(input[index] < '0' || input[index] > '9') {
            return -1;
        }
    }
    
    return atoi(input);
}

int calculate_average(int arr[], int size) {
    int sum = 0;
    int i;
    int temp_var;
    unsigned int count = size;
    
    for(i = 0; i < count; i++) {
        sum += arr[i];
    }
    
    if(size > 0)
        return sum / size;
}

void debug_print(int a, int b, int c) {
    int debug_flag;
    printf("Debug values: %d %d\n", a, b);
}