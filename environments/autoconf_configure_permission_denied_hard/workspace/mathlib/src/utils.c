#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

void print_result(double value)
{
    printf("Result: %.6f\n", value);
}

int is_even(int n)
{
    return (n % 2 == 0) ? 1 : 0;
}

int max(int a, int b)
{
    return (a > b) ? a : b;
}