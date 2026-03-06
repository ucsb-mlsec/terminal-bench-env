#include <stdio.h>
#include "helper.h"

void helper_function(void) {
    printf("Helper function called\n");
}

int calculate_sum(int a, int b) {
    return a + b;
}

void print_helper(const char* message) {
    if (message != NULL) {
        printf("[HELPER] %s\n", message);
    }
}