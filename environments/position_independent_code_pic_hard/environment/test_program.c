#include <stdio.h>
#include <stdlib.h>

// Function declarations from the data processing library
extern int transform_data(int* arr, int size);
extern int filter_values(int* arr, int size, int* out);
extern int validate_input(int* arr, int size);

int main() {
    // Create a test array with mixed positive and negative values
    int test_array[] = {5, -3, 8, -1, 12, -7, 3, 9};
    int size = 8;
    int filtered_array[8];
    int i;
    
    printf("Original array: ");
    for (i = 0; i < size; i++) {
        printf("%d ", test_array[i]);
    }
    printf("\n");
    
    // Validate the input array
    if (validate_input(test_array, size) == 0) {
        printf("Error: Input validation failed\n");
        return 1;
    }
    printf("Input validation passed\n");
    
    // Transform the data
    int transform_result = transform_data(test_array, size);
    printf("Transform operation returned: %d\n", transform_result);
    
    printf("Transformed array: ");
    for (i = 0; i < size; i++) {
        printf("%d ", test_array[i]);
    }
    printf("\n");
    
    // Filter out negative values
    int filtered_count = filter_values(test_array, size, filtered_array);
    printf("Filter operation found %d positive values\n", filtered_count);
    
    printf("Filtered array (positives only): ");
    for (i = 0; i < filtered_count; i++) {
        printf("%d ", filtered_array[i]);
    }
    printf("\n");
    
    printf("All library functions executed successfully\n");
    return 0;
}