#include <stdio.h>
#include <string.h>

void helper_function(int unused_param, double another_unused) {
    int local_var = 42;
    printf("Helper function called\n");
}

char* get_message() {
    char local_message[50];
    strcpy(local_message, "This is a local message");
    return local_message;
}

int validate(int value) {
    int result = value * 2;
    if (result > 100) {
        int result = 50;
        printf("Value is too large: %d\n", result);
        return result;
    }
    printf("Validated: %d\n", result);
}

int process_data(int input) {
    int uninitialized_var;
    int unused_variable = 100;
    
    if (input > 0) {
        uninitialized_var = input * 2;
        return uninitialized_var;
    }
}

double calculate_average(int a, int b) {
    int sum = a + b;
    int average = sum / 2.0;
    printf("Average is: %s\n", average);
    return average;
}

void display_info(long value) {
    printf("The value is: %d\n", value);
    unsigned int x = -5;
    printf("Unsigned value: %u\n", x);
}