#include <stdio.h>
#include <math.h>
#include "../src/vector.h"

#define EPSILON 1e-6

int main() {
    // Create test vectors
    Vector3 v1 = {3.0, 4.0, 0.0};
    Vector3 v2 = {1.0, 0.0, 0.0};
    
    // Test dot product
    double dot = vector_dot(&v1, &v2);
    if (fabs(dot - 3.0) > EPSILON) {
        printf("Error: Dot product test failed. Expected 3.0, got %f\n", dot);
        return 1;
    }
    
    // Test magnitude
    double mag = vector_magnitude(&v1);
    if (fabs(mag - 5.0) > EPSILON) {
        printf("Error: Magnitude test failed. Expected 5.0, got %f\n", mag);
        return 1;
    }
    
    // Test normalize
    Vector3 normalized = vector_normalize(&v1);
    double expected_x = 3.0 / 5.0;
    double expected_y = 4.0 / 5.0;
    double expected_z = 0.0;
    
    if (fabs(normalized.x - expected_x) > EPSILON ||
        fabs(normalized.y - expected_y) > EPSILON ||
        fabs(normalized.z - expected_z) > EPSILON) {
        printf("Error: Normalize test failed. Expected (%f, %f, %f), got (%f, %f, %f)\n",
               expected_x, expected_y, expected_z,
               normalized.x, normalized.y, normalized.z);
        return 1;
    }
    
    // Test that normalized vector has magnitude 1
    double norm_mag = vector_magnitude(&normalized);
    if (fabs(norm_mag - 1.0) > EPSILON) {
        printf("Error: Normalized vector magnitude test failed. Expected 1.0, got %f\n", norm_mag);
        return 1;
    }
    
    printf("Vector tests passed\n");
    return 0;
}