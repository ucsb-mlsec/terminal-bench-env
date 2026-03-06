#include <stdio.h>
#include "utils.h"

int calculate_sum(int a, int b) {
    return a + b;
}

void print_message(const char* msg) {
    if (msg != NULL) {
        printf("%s\n", msg);
    }
}