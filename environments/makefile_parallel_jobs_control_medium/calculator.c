#include <stdio.h>
#include "utils.h"

int main() {
    int a = 5;
    int b = 3;
    
    printf("Calculator Program\n");
    printf("==================\n\n");
    
    int sum_result = add_numbers(a, b);
    printf("Addition: %d + %d = %d\n", a, b, sum_result);
    
    int square_result = calculate_square(a);
    printf("Square of %d: %d\n", a, square_result);
    
    int power_result = calculate_power(b, 2);
    printf("Power: %d^2 = %d\n", b, power_result);
    
    printf("\nCalculations completed successfully!\n");
    
    return 0;
}