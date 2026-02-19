#include <stdio.h>

int add(int a, int b) {
    int a;  // Shadow declaration - shadows parameter
    a = 5;
    return a + b;
}

int subtract(int x, int y) {
    int result;
    int unused_var = 42;  // Unused variable
    result = x - y;
    return result;
}

int multiply(int x, int y) {
    int temp = x * y;
    int another_unused;  // Unused variable
    return temp;
}

float divide(int a, int b) {
    // Type conversion from int to float
    return a / b;
}

int compute() {
    int value = 10;
    value = value + 5;
    // Missing return statement
}

void print_result(int value) {
    // Format specifier mismatch - using %s for int
    printf("The result is: %s\n", value);
}

int calculate_sum(int n) {
    int sum;  // Uninitialized variable
    int i;
    for (i = 1; i <= n; i++) {
        sum += i;  // Using uninitialized sum
    }
    return sum;
}

long convert_value(int val) {
    // Implicit conversion that may lose data
    long result = val * 100000;
    return result;
}

void process_data(int x, int y, int z) {
    int x;  // Another shadow declaration
    x = 100;
    printf("Processing: %d\n", x);
}