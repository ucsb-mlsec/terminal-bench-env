#include <stdio.h>
#include "add.h"
#include "subtract.h"
#include "multiply.h"
#include "divide.h"

int main() {
    int a = 10;
    int b = 5;
    int result;

    // Test add function
    result = add(a, b);
    printf("Testing add: %d + %d = %d\n", a, b, result);

    // Test subtract function
    result = subtract(a, b);
    printf("Testing subtract: %d - %d = %d\n", a, b, result);

    // Test multiply function
    result = multiply(a, b);
    printf("Testing multiply: %d * %d = %d\n", a, b, result);

    // Test divide function
    result = divide(a, b);
    printf("Testing divide: %d / %d = %d\n", a, b, result);

    return 0;
}