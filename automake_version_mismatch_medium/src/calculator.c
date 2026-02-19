#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    double num1 = 15.5;
    double num2 = 4.5;
    double sum, difference, product, quotient;

    printf("Simple Calculator v1.0\n");
    printf("======================\n\n");

    sum = num1 + num2;
    difference = num1 - num2;
    product = num1 * num2;
    quotient = num1 / num2;

    printf("Performing calculations with %.2f and %.2f:\n", num1, num2);
    printf("Addition: %.2f + %.2f = %.2f\n", num1, num2, sum);
    printf("Subtraction: %.2f - %.2f = %.2f\n", num1, num2, difference);
    printf("Multiplication: %.2f * %.2f = %.2f\n", num1, num2, product);
    printf("Division: %.2f / %.2f = %.2f\n", num1, num2, quotient);

    return 0;
}