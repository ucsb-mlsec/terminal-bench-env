#include <stdio.h>
#include <stdlib.h>

int main() {
    int array1[20];
    int array2[20];
    int results[20];
    int i;
    int sum = 0;
    int product = 1;
    
    // Initialize first array with sequential values
    for (i = 0; i < 20; i++) {
        array1[i] = i + 1;
    }
    
    // Initialize second array with multiples of 2
    for (i = 0; i < 20; i++) {
        array2[i] = (i + 1) * 2;
    }
    
    // Perform operations: add corresponding elements
    for (i = 0; i < 20; i++) {
        results[i] = array1[i] + array2[i];
    }
    
    // Calculate sum of results
    for (i = 0; i < 20; i++) {
        sum += results[i];
    }
    
    // Calculate product of first 5 elements of array1
    for (i = 0; i < 5; i++) {
        product *= array1[i];
    }
    
    // Perform some additional mixed operations
    for (i = 0; i < 10; i++) {
        array1[i] = array1[i] * 2;
        array2[i] = results[i] + array1[i];
    }
    
    // Calculate final average
    int average = sum / 20;
    
    // Print results
    printf("Sum: %d\n", sum);
    printf("Product: %d\n", product);
    printf("Average: %d\n", average);
    
    return 0;
}