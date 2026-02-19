#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function declarations from libmath
int add(int a, int b);
int multiply(int a, int b);
double power(double base, double exponent);

// Function declarations from libstring
int str_length(const char* str);
char* str_reverse(const char* str);
char* str_upper(const char* str);

int main() {
    printf("=== Testing libmath functions ===\n");
    
    // Test add function
    int sum = add(5, 3);
    printf("add(5, 3) = %d\n", sum);
    if (sum != 8) {
        printf("ERROR: add function failed\n");
        return 1;
    }
    
    // Test multiply function
    int product = multiply(4, 7);
    printf("multiply(4, 7) = %d\n", product);
    if (product != 28) {
        printf("ERROR: multiply function failed\n");
        return 1;
    }
    
    // Test power function
    double result = power(2.0, 3.0);
    printf("power(2.0, 3.0) = %.2f\n", result);
    if (result != 8.0) {
        printf("ERROR: power function failed\n");
        return 1;
    }
    
    printf("\n=== Testing libstring functions ===\n");
    
    // Test str_length function
    const char* test_str1 = "hello";
    int length = str_length(test_str1);
    printf("str_length(\"%s\") = %d\n", test_str1, length);
    if (length != 5) {
        printf("ERROR: str_length function failed\n");
        return 1;
    }
    
    // Test str_reverse function
    const char* test_str2 = "world";
    char* reversed = str_reverse(test_str2);
    printf("str_reverse(\"%s\") = \"%s\"\n", test_str2, reversed);
    if (strcmp(reversed, "dlrow") != 0) {
        printf("ERROR: str_reverse function failed\n");
        free(reversed);
        return 1;
    }
    free(reversed);
    
    // Test str_upper function
    const char* test_str3 = "testing";
    char* upper = str_upper(test_str3);
    printf("str_upper(\"%s\") = \"%s\"\n", test_str3, upper);
    if (strcmp(upper, "TESTING") != 0) {
        printf("ERROR: str_upper function failed\n");
        free(upper);
        return 1;
    }
    free(upper);
    
    printf("\n=== All tests passed successfully ===\n");
    return 0;
}