#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int num1, num2, result;
    
    if (argc != 3) {
        fprintf(stderr, "Error: Expected exactly 2 arguments\n");
        fprintf(stderr, "Usage: %s <number1> <number2>\n", argv[0]);
        return 1;
    }
    
    num1 = atoi(argv[1]);
    num2 = atoi(argv[2]);
    
    result = num1 + num2;
    
    printf("%d\n", result);
    
    return 0;
}