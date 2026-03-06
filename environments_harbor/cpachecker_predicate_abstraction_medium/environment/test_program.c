#include <stdlib.h>
#include <stdio.h>

// Test program for CPAchecker predicate abstraction analysis
// Contains intentional memory safety issues for verification testing

int main() {
    int *array;
    int size = 10;
    int index;
    int *ptr;
    
    // Dynamic memory allocation
    array = (int *)malloc(size * sizeof(int));
    
    if (array == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Initialize array elements
    for (int i = 0; i < size; i++) {
        array[i] = i * 2;
    }
    
    // Read index from symbolic input
    // In verification, this could be any value
    index = size + 5; // potential buffer overflow here
    
    // Unsafe array access - could access out of bounds
    if (index >= 0) {
        int value = array[index]; // potential buffer overflow here
        printf("Value at index %d: %d\n", index, value);
    }
    
    // Another potential issue with pointer arithmetic
    ptr = array + 15; // pointer goes beyond allocated memory
    *ptr = 100; // potential buffer overflow here - writing out of bounds
    
    // Conditional branch with unsafe operations
    if (size > 5) {
        array[size] = 99; // off-by-one error - array[10] when size is 10
    } else {
        array[0] = 0;
    }
    
    // Memory leak risk - allocated memory not freed in all paths
    if (size < 5) {
        free(array);
    }
    // memory leak risk - when size >= 5, memory is never freed
    
    return 0;
}