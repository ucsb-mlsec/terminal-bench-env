#include <stdio.h>
#include "../utils/utils.h"

int main(int argc, char** argv) {
    // Visualization tool - uses utils library but NOT MPI
    
    printf("===========================================\n");
    printf("Physics Visualization Tool\n");
    printf("===========================================\n");
    
    // Use utility functions from the utils library
    print_header();
    
    // Test vector magnitude calculation
    double x = 3.0;
    double y = 4.0;
    double z = 0.0;
    
    double magnitude = vector_magnitude(x, y, z);
    printf("\nVector magnitude test:\n");
    printf("Vector (%.1f, %.1f, %.1f) has magnitude: %.2f\n", x, y, z, magnitude);
    
    printf("\nVisualization tool running (no MPI)\n");
    printf("This component does NOT use MPI parallelization\n");
    
    return 0;
}