#include <stdio.h>
#include "../include/utils.h"

int main() {
    printf("Starting program...\n");
    print_message("Hello from utils!");
    int result = add_numbers(5, 3);
    printf("Result: %d\n", result);
    return 0;
}