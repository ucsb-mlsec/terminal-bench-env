#include <stdio.h>
#include <stdlib.h>

// Function without return type (implicitly int - old C style)
calculate_sum(int a, int b) {
    return a + b;
}

// Function with missing return statement
int get_value(int x) {
    if (x > 10) {
        return x * 2;
    }
    // Missing return for other cases
}

// Function with uninitialized variable
int process_data(int input) {
    int result;
    int temp = input + 5;
    return result; // result used without initialization
}

// Function with signed/unsigned comparison
int check_bounds(int value, unsigned int limit) {
    if (value < limit) {
        return 1;
    }
    return 0;
}

// Function called before declaration
int compute_average(int a, int b, int c);

// Function with format string mismatch
void print_statistics(long int count, unsigned int total) {
    printf("Count: %d\n", count); // %d for long int
    printf("Total: %d\n", total); // %d for unsigned int
}

int main() {
    int unused_var;
    int another_unused = 42;
    char unused_char = 'x';
    
    int x = 10;
    int y = 20;
    unsigned int limit = 100;
    
    // Call function before it's declared
    int avg = compute_average(x, y, 30);
    
    // Signed/unsigned comparison warning
    if (x < limit) {
        printf("x is within limit\n");
    }
    
    // Call function without return type
    int sum = calculate_sum(x, y);
    printf("Sum: %d\n", sum);
    
    // Call function with missing return path
    int val = get_value(5);
    printf("Value: %d\n", val);
    
    // Call function with uninitialized variable
    int processed = process_data(15);
    printf("Processed: %d\n", processed);
    
    // Format string mismatches
    long int big_number = 1000000L;
    unsigned int count = 500;
    printf("Big number: %d\n", big_number); // Wrong format specifier
    printf("Count: %d\n", count);
    
    // More format mismatches
    print_statistics(big_number, count);
    
    // Signed/unsigned arithmetic
    unsigned int u_val = 50;
    int s_val = -10;
    int mixed_result = u_val + s_val;
    printf("Mixed result: %d\n", mixed_result);
    
    // Call check_bounds with signed/unsigned mix
    int boundary_check = check_bounds(s_val, u_val);
    printf("Boundary check: %d\n", boundary_check);
    
    // Implicit function call (function not declared)
    helper_function(x, y);
    
    printf("Program completed\n");
    
    return 0;
}

// Definition of compute_average (declared after use)
int compute_average(int a, int b, int c) {
    return (a + b + c) / 3;
}

// Helper function defined after being called
void helper_function(int p1, int p2) {
    int unused_local = 99;
    printf("Helper called with: %d, %d\n", p1, p2);
}

// Another function with multiple issues
get_max(int arr[], int size) {
    int i;
    int max;
    unsigned int index = 0;
    
    // Signed/unsigned comparison in loop
    for (i = 0; i < size; i++) {
        if (i < index) {
            continue;
        }
    }
    
    return max; // Uninitialized max
}