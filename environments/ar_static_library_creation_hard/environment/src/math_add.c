#include <stdio.h>

int add(int a, int b) {
    return a + b;
}

int math_add(int a, int b) {
    printf("Math library: Adding %d + %d\n", a, b);
    return a + b;
}