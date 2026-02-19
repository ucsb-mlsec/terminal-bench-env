#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "utils.h"

double vector_magnitude(double x, double y, double z) {
    return sqrt(x * x + y * y + z * z);
}

void initialize_array(double *array, int size) {
    if (array == NULL) {
        fprintf(stderr, "Error: NULL pointer passed to initialize_array\n");
        return;
    }
    
    for (int i = 0; i < size; i++) {
        array[i] = 0.0;
    }
}

void print_header(void) {
    printf("====================================\n");
    printf("  Computational Physics Project\n");
    printf("  HPC Simulation System v1.0\n");
    printf("====================================\n");
}

double compute_distance(double x1, double y1, double z1, 
                       double x2, double y2, double z2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;
    return vector_magnitude(dx, dy, dz);
}

void array_statistics(double *array, int size, double *mean, double *sum) {
    if (array == NULL || mean == NULL || sum == NULL) {
        return;
    }
    
    *sum = 0.0;
    for (int i = 0; i < size; i++) {
        *sum += array[i];
    }
    *mean = *sum / (double)size;
}