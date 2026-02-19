#include <stdio.h>
#include "mathops.h"

int main() {
    int a = 5;
    int b = 3;
    int result_add;
    int result_mult;
    double result_div;
    int result_fact;
    
    /* Test addition */
    result_add = add(a, b);
    printf("%d + %d = %d\n", a, b, result_add);
    
    /* Test multiplication */
    result_mult = multiply(4, 7);
    printf("4 * 7 = %d\n", result_mult);
    
    /* Test division */
    result_div = divide(10.0, 2.0);
    printf("10.0 / 2.0 = %.2f\n", result_div);
    
    /* Test factorial */
    result_fact = factorial(5);
    printf("5! = %d\n", result_fact);
    
    printf("All tests completed successfully.\n");
    
    return 0;
}