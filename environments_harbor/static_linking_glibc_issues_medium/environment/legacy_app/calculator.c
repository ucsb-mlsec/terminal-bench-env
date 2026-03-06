#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <number> <operation>\n", argv[0]);
        fprintf(stderr, "Operations: sqrt, log, sin, cos, tan\n");
        return 1;
    }

    double number = atof(argv[1]);
    char *operation = argv[2];
    double result;

    if (strcmp(operation, "sqrt") == 0) {
        if (number < 0) {
            fprintf(stderr, "Error: Cannot calculate square root of negative number\n");
            return 1;
        }
        result = sqrt(number);
    } else if (strcmp(operation, "log") == 0) {
        if (number <= 0) {
            fprintf(stderr, "Error: Cannot calculate logarithm of non-positive number\n");
            return 1;
        }
        result = log(number);
    } else if (strcmp(operation, "sin") == 0) {
        result = sin(number);
    } else if (strcmp(operation, "cos") == 0) {
        result = cos(number);
    } else if (strcmp(operation, "tan") == 0) {
        result = tan(number);
    } else {
        fprintf(stderr, "Error: Unknown operation '%s'\n", operation);
        fprintf(stderr, "Supported operations: sqrt, log, sin, cos, tan\n");
        return 1;
    }

    printf("%.10f\n", result);
    return 0;
}