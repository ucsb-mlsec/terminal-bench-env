#include <stdio.h>
#include "math_ops.h"
#include "utils.h"

int main() {
    printf("Testing library integration...\n");
    
    // Test math library functions
    int a = 10;
    int b = 5;
    int sum = add(a, b);
    int product = multiply(a, b);
    
    printf("Math operations:\n");
    printf("  %d + %d = %d\n", a, b, sum);
    printf("  %d * %d = %d\n", a, b, product);
    
    // Test utils library functions
    printf("\nUtils operations:\n");
    print_message("Hello from utils library!");
    
    char buffer[100];
    format_string(buffer, "Result", sum);
    printf("  %s\n", buffer);
    
    printf("\nAll tests completed successfully!\n");
    
    return 0;
}